
import os
import numpy as np
import math
# import random
import types
import itertools
from tqdm import tqdm
from functools import partial
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
from mmseg.ops import resize

import dinov2.eval.segmentation.models
from dinov2.hub.backbones import dinov2_vitg14

import logging
from mmcv.utils import get_logger

logger = get_logger('mmcv')
logger.setLevel(logging.WARNING)    # Disable mmcv info print

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def fast_hist(label, prediction, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n).reshape(n, n)


def per_class_miou(hist):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iou


def head_decode(self, crop_feature_list, img_metas, backbone_model, shape):
    """Decode features and generate segmentation output.

    Args:
        crop_feature_list (List[torch.Tensor]): List of cropped feature maps.
        img_metas (List[dict]): Metadata for the image(s).
        backbone_model (torch.nn.Module): Backbone model for feature normalization.
        shape (Tuple[int, int]): Shape of the original image.

    Returns:
        torch.Tensor: Resized segmentation output.
    """
    # Normalize feature maps using the backbone model's normalization, moved it from backbone to head
    outputs = [backbone_model.norm(out) for out in crop_feature_list]

    # Remove class tokens and retain patch tokens
    outputs = [out[:, 1 + backbone_model.num_register_tokens:] for out in outputs]

    # Reshape and reorder dimensions for compatibility, moved it from backbone to head
    B, w, h = outputs[0].shape[0], shape[0], shape[1]
    outputs = [
        out.reshape(B, math.ceil(w / backbone_model.patch_size), math.ceil(h / backbone_model.patch_size), -1)
        .permute(0, 3, 1, 2)
        .contiguous()
        for out in outputs
    ]

    # Process through neck if applicable
    x = tuple(outputs)
    if self.with_neck:
        x = self.neck(x)

    # Forward through decode head and resize output
    out = self._decode_head_forward_test(x, img_metas)
    out = resize(input=out, size=shape, mode='bilinear', align_corners=self.align_corners)

    return out


def slide_inference_encode(self, img, img_meta, rescale=True):
    """Extract features using sliding-window inference.

    Args:
        img (torch.Tensor): Input image tensor.
        img_meta (List[dict]): Metadata for the image(s).
        rescale (bool, optional): Whether to rescale features to the original size. Defaults to True.

    Returns:
        List[torch.Tensor]: List of extracted features for sliding-window crops.
    """
    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()

    feature_list = []
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Crop and extract features
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_feature_list = self.backbone(crop_img)
            feature_list.append(crop_feature_list)

    return feature_list


def slide_inference_decode(self, feature_list, img_meta, rescale=True, backbone_model=None):
    """Decode features using sliding-window inference.

    Args:
        feature_list (List[List[torch.Tensor]]): List of extracted feature maps for each crop.
        img_meta (List[dict]): Metadata for the image(s).
        rescale (bool, optional): Whether to rescale the output to the original size. Defaults to True.
        backbone_model (torch.nn.Module, optional): Backbone model for decoding. Defaults to None.

    Returns:
        torch.Tensor: Decoded segmentation predictions.
    """
    device = next(backbone_model.parameters()).device
    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size = feature_list[0][0].shape[0]
    h_img, w_img = img_meta[0]['img_shape'][0], img_meta[0]['img_shape'][1]
    num_classes = self.num_classes

    # Initialize predictions and counting matrix
    preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=device)
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=device)

    i = 0
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Decode crop and update predictions
            crop_seg_logit = self.head_decode(feature_list[i], img_meta, backbone_model, (h_crop, w_crop))
            preds += F.pad(crop_seg_logit, (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2))
            count_mat[:, :, y1:y2, x1:x2] += 1
            i += 1

    assert (count_mat == 0).sum() == 0, "Zero count in count matrix detected"

    preds = preds / count_mat

    if rescale:
        # Resize predictions to original image size
        resize_shape = img_meta[0]['img_shape'][:2]
        preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
        preds = resize(
            preds,
            size=img_meta[0]['ori_shape'][:2],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False
        )

    return preds


def simple_test_encode(self, img, img_meta, rescale=True):
    """Perform simple encoding during testing for a single image.

    Args:
        img (torch.Tensor): Input image tensor.
        img_meta (List[dict]): Metadata for the image, including original shape.
        rescale (bool, optional): Whether to rescale the features to the original image size. Defaults to True.

    Returns:
        List[torch.Tensor]: Extracted features from the input image.
    """
    # Ensure the test mode is valid
    assert self.test_cfg.mode in ['slide', 'whole'], "Invalid test mode"

    # Check that all images have the same original shape
    ori_shape = img_meta[0]['ori_shape']
    assert all(meta['ori_shape'] == ori_shape for meta in img_meta), "Mismatch in original shapes across metadata"

    # Extract features based on the test mode
    if self.test_cfg.mode == 'slide':
        feature_list = self.slide_inference_encode(img, img_meta, rescale)
    else:
        raise ValueError("Only 'slide' mode is currently supported")

    return feature_list


def simple_test_decode(self, feature_list, img_meta, backbone_model, rescale=True):
    """Perform simple decoding during testing for a single image.

    Args:
        feature_list (List[torch.Tensor]): Extracted features from the model's backbone.
        img_meta (dict): Metadata for the image, including flip information.
        backbone_model (torch.nn.Module): Backbone model used in the segmentation process.
        rescale (bool, optional): Whether to rescale the output to the original image size. Defaults to True.

    Returns:
        List[np.ndarray]: Decoded segmentation prediction for the input image.
    """
    # Perform sliding window inference for decoding
    seg_logit = self.slide_inference_decode(feature_list, img_meta, rescale, backbone_model=backbone_model)

    # Apply softmax to generate probabilities
    output = F.softmax(seg_logit, dim=1)

    # Handle image flipping if applicable
    flip = img_meta[0]['flip']
    if flip:
        flip_direction = img_meta[0]['flip_direction']
        assert flip_direction in ['horizontal', 'vertical'], "Invalid flip direction"
        if flip_direction == 'horizontal':
            output = output.flip(dims=(3,))
        elif flip_direction == 'vertical':
            output = output.flip(dims=(2,))

    # Generate segmentation prediction
    seg_logit = output
    seg_pred = seg_logit.argmax(dim=1)

    # Handle ONNX export compatibility
    if torch.onnx.is_in_onnx_export():
        # Ensure the inference backend receives 4D output
        seg_pred = seg_pred.unsqueeze(0)
        return seg_pred

    # Convert predictions to numpy and unravel batch dimension
    seg_pred = seg_pred.cpu().numpy()
    seg_pred = list(seg_pred)

    return seg_pred


def setup_backbone(backbone_checkpoint_path: str) -> torch.nn.Module:
    """Setup and return the DINOv2 backbone model."""   
    backbone_model = dinov2_vitg14(pretrained=True, weights=backbone_checkpoint_path)
    backbone_model.eval()
    backbone_model.cuda()
    
    return backbone_model


def create_segmenter(cfg: mmcv.Config, backbone_model: torch.nn.Module) -> torch.nn.Module:
    """Create and initialize a segmentation model.

    Args:
        cfg (mmcv.Config): Configuration object defining the segmentation model's structure and parameters.
        backbone_model (torch.nn.Module): Backbone model used for feature extraction.

    Returns:
        torch.nn.Module: Initialized segmentation model with modified backbone.
    """
    # Initialize the segmentation model using the configuration
    model = init_segmentor(cfg)

    # Modify the backbone forward function for feature extraction
    model.backbone.forward = partial(
        backbone_model.seg_backbone,
        n=cfg.model.backbone.out_indices,  # Specify output layer indices
        reshape=False,  # Set to False, move the reshape operation to head
        norm=cfg.model.backbone.final_norm, # Set to False in cfg, move the norm operation to head
    )

    # Add center padding if the backbone model has a patch_size attribute
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )

    # Initialize model weights
    model.init_weights()

    return model


def build_segmentation_model(cfg: mmcv.Config, backbone_model: torch.nn.Module, head_checkpoint_path: str) -> torch.nn.Module:
    """Build and initialize a segmentation model with custom methods.

    Args:
        cfg (mmcv.Config): Configuration object for the model.
        head_checkpoint_path (str): Path to the checkpoint file for the model's head.

    Returns:
        torch.nn.Module: Initialized segmentation model ready for inference.
    """

    # Create the segmentation model
    model = create_segmenter(cfg, backbone_model)

    # Attach custom methods to the model
    model.simple_test_encode = types.MethodType(simple_test_encode, model)
    model.simple_test_decode = types.MethodType(simple_test_decode, model)
    model.slide_inference_encode = types.MethodType(slide_inference_encode, model)
    model.slide_inference_decode = types.MethodType(slide_inference_decode, model)
    model.head_decode = types.MethodType(head_decode, model)

    # Load pre-trained weights for the head
    load_checkpoint(model, head_checkpoint_path, map_location="cpu")

    # Move the model to GPU and set to evaluation mode
    model.cuda()
    model.eval()

    return model


def extract_features(model: torch.nn.Module, source_img_path: str, org_feature_path: str, image_list: list):
    """Extract and save features for given images.

    Args:
        model (torch.nn.Module): Segmentation model.
        source_img_path (str): Path to source images.
        org_feature_path (str): Path to save extracted features.
        image_list (List[str]): List of image names to process.
    """
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + model.cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    for image_name in tqdm(image_list):
        # Load and preprocess image
        image = Image.open(f'{source_img_path}/JPEGImages/{image_name}.jpg')
        img = np.array(image)[:, :, ::-1]  # BGR

        # Prepare  data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]   # Scatter to specified GPU
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        # Extract features
        with torch.no_grad():
            org_feature_list = model.simple_test_encode(data['img'][0], data['img_metas'][0], rescale=True)
            org_feature_list = [torch.cat(feature_list) for feature_list in org_feature_list]
            org_feature_list = torch.stack(org_feature_list)
            
            # Save features
            np.save(f'{org_feature_path}/{image_name}.npy', org_feature_list.cpu().detach().numpy())


def seg_evaluate(model: torch.nn.Module, source_img_path: str, org_feature_path: str, rec_feature_path: str, image_list: list, backbone_model: torch.nn.Module):
    """Evaluate segmentation performance.

    Args:
        model (torch.nn.Module): Segmentation model.
        source_img_path (str): Path to source images.
        org_feature_path (str): Path to original features.
        rec_feature_path (str): Path to reconstructed features.
        image_list (List[str]): List of image names to evaluate.
        backbone_model (torch.nn.Module): Backbone model.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + model.cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    hist = np.zeros((model.decode_head.num_classes, model.decode_head.num_classes))   # 20 classes + 1 background
    mse_list = []

    for image_name in tqdm(image_list):
        # Load image and label
        img = Image.open(f'{source_img_path}/JPEGImages/{image_name}.jpg')
        label = Image.open(f'{source_img_path}/SegmentationClass/{image_name}.png')
        img = np.array(img)[:, :, ::-1]  # BGR

        # Preprocess data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]   # Scatter to specified GPU
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        with torch.no_grad():
            # Load features and move to the specified device
            rec_features_numpy = np.load(f'{rec_feature_path}/{image_name}.npy')
            rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)
            
            # Convert features to the required format for the model
            rec_feature_list = [
                [rec_features_tensor[i, j].unsqueeze(0) for j in range(rec_features_tensor.shape[1])]
                for i in range(rec_features_tensor.shape[0])
            ]
            
            # Perform segmentation prediction
            pred = model.simple_test_decode(
                rec_feature_list, 
                data['img_metas'][0], 
                backbone_model, 
                rescale=True
            )

        array_label = np.array(label)
        hist += fast_hist(array_label, pred[0], model.decode_head.num_classes)

        # Calculate MSE
        org_feature = np.load(f'{org_feature_path}/{image_name}.npy')
        mse = (np.square(org_feature - rec_features_numpy)).mean()
        mse_list.append(mse)

    # Calculate metrics
    all_iou = per_class_miou(hist)
    all_miou = np.nanmean(all_iou)

    return all_iou, all_miou, mse_list


def main(config_path: str, backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_split_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the depth estimation pipeline."""
    # Load configuration
    cfg = mmcv.Config.fromfile(config_path)
    
    # Setup source image list
    with open(source_split_name) as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()

    # Setup models
    backbone_model = setup_backbone(backbone_checkpoint_path)
    model = build_segmentation_model(cfg, backbone_model, head_checkpoint_path)
    
    # Extract features
    extract_features(model, source_img_path, org_feature_path, image_list)
    
    # Evaluate and print results
    # all_iou, all_miou, mse_list = seg_evaluate(model, source_img_path, org_feature_path, rec_feature_path, image_list, backbone_model)
    
    # print(f"IoU: ", end=" ")
    # for iou in all_iou: print(f"{iou*100:.4f}", end=" ") 
    # print(f"\nmIoU: {all_miou*100:.4f}")
    # print(f"Feature MSE: {np.mean(mse_list):.8f}")


if __name__ == "__main__":
    config_path = f"cfg/dinov2_vitg14_voc2012_linear_config.py"
    backbone_checkpoint_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/pretrained_head/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/pretrained_head/dinov2_vitg14_voc2012_linear_head.pth'
    
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/VOC2012_Train100'
    source_split_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/train_100.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/feature_train'
    rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/feature_test'
    
    main(config_path, backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_split_name, org_feature_path, rec_feature_path)