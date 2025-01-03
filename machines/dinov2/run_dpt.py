import os
import math
import itertools
import numpy as np
from PIL import Image
from functools import partial
from typing import Tuple, List


import torch
import torch.nn.functional as F
import mmcv
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmseg.datasets.pipelines import Compose
from dinov2.hub.backbones import dinov2_vitg14
from dinov2.eval.depth.models import build_depther
from depth.datasets import build_dataloader, build_dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size: int) -> Tuple[int, int]:
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)


def create_depther_model(cfg: mmcv.Config, backbone_model: torch.nn.Module, head_checkpoint_path: str, backbone_size: str) -> torch.nn.Module: #gcs
    """Create and configure the depth estimation model."""
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, test_cfg=test_cfg)

    # Change the default backbone "forward" function to "dpt_backbone" function
    depther.backbone.forward = partial(
        backbone_model.dpt_backbone, 
        n=cfg.model.backbone.out_indices,
        reshape=False,  # Set to False, move the reshape operation to head, to obtain original feature in the shape of 1611x1536
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm, # Set to False in cfg. Dpt do not perfrom norm
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )

    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    
    load_checkpoint(depther, head_checkpoint_path, map_location="cpu")
    depther.eval()
    depther.cuda()

    return depther

def setup_backbone(backbone_checkpoint_path: str) -> torch.nn.Module:
    """Setup and return the DINOv2 backbone model."""   
    backbone_model = dinov2_vitg14(pretrained=True, weights=backbone_checkpoint_path)
    backbone_model.eval()
    backbone_model.cuda()
    
    return backbone_model

def extract_features(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, org_feature_path: str):
    """Extract features from images and save them."""
    device = next(model.parameters()).device
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for batch_indices, data in zip(data_loader.batch_sampler, data_loader):
        with torch.no_grad():
            data['img'] = [_.cuda() for _ in data['img']]
            aug_feature_list = model.aug_test_encode(data['img'], data["img_metas"])         
            aug_feature_list = [torch.cat(feature_list) for feature_list in aug_feature_list]
            aug_feature_list = torch.stack(aug_feature_list)
            
            file_path = data['img_metas'][0].data[0][0]['filename'].split('/')
            file_name = f"{file_path[-2]}_{file_path[-1].split('.')[0]}"
            
            np.save(f'{org_feature_path}/{file_name}.npy', aug_feature_list.cpu().detach().numpy())
        
        prog_bar.update()

def evaluate_depth(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                  org_feature_path: str, rec_feature_path: str, backbone_model: torch.nn.Module) -> Tuple[np.ndarray, float]:
    """Evaluate depth estimation using extracted features."""
    device = next(model.parameters()).device
    results = np.zeros((1, 9))
    mse_eval = 0.0
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))

    for batch_indices, data in zip(data_loader.batch_sampler, data_loader):
        with torch.no_grad():
            # Load features
            file_path = data['img_metas'][0].data[0][0]['filename'].split('/')
            file_name = f"{file_path[-2]}_{file_path[-1].split('.')[0]}"
            rec_feature_numpy = np.load(f'{rec_feature_path}/{file_name}.npy')
            rec_feature_tensor = torch.from_numpy(rec_feature_numpy).to(device)
            
            # Convert features to the required format for the model           
            rec_feature_list = [
                [rec_feature_tensor[i, j].unsqueeze(0) for j in range(rec_feature_tensor.shape[1])]
                for i in range(rec_feature_tensor.shape[0])
            ]

            # Generate depth estimation
            result_depth = model.aug_test_decode(
                rec_feature_list, 
                backbone_model, 
                data["img_metas"],
                shape=data['img'][0].shape
            )

            # Calculate MSE
            org_feat = np.load(f'{org_feature_path}/{file_name}.npy')
            mse = np.square(org_feat - rec_feature_numpy).mean()
            mse_eval += mse

            # Process results
            result, _ = data_loader.dataset.pre_eval(result_depth, indices=batch_indices)
            results += np.array(result, dtype='float32')
            
        prog_bar.update()

    return results / len(data_loader), mse_eval / len(data_loader)

def main(config_path: str, backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_split_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the depth estimation pipeline."""
    # Load configuration
    cfg = mmcv.Config.fromfile(config_path)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # Setup models
    backbone_model = setup_backbone(backbone_checkpoint_path)
    model = create_depther_model(cfg, backbone_model, head_checkpoint_path, "giant")
    
    # Setup data loader
    cfg.data.test.data_root = source_img_path
    cfg.data.test.split = source_split_name
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1, 
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )
    
    # Extract features
    extract_features(model, data_loader, org_feature_path)
    
    # Evaluate and print results
    # results, feat_mse = evaluate_depth(model, data_loader, org_feature_path, rec_feature_path, backbone_model)
    
    # # print("a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel")
    # print(f"\nRMSE: {results[0][4]:.8f}")
    # print(f"Feature MSE: {feat_mse:.8f}")

if __name__ == "__main__":
    config_path = "cfg/dinov2_vitg14_nyu_linear4_config.py"
    backbone_checkpoint_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/pretrained_head/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/pretrained_head/dinov2_vitg14_nyu_linear4_head.pth"
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/source/NYU_Test16'
    source_split_name = 'nyu_test.txt'
    org_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test"
    rec_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test_copy"
    
    main(config_path, backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_split_name, org_feature_path, rec_feature_path)