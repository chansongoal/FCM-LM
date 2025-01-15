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
            #gcs, modify the features
            # rec_feature_numpy = rec_feature_numpy + 0.1*np.random.rand(*rec_feature_numpy.shape).astype(np.float32)
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

def dpt_pipeline(config_path: str, backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_split_name: str, org_feature_path: str, rec_feature_path: str):
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
    # extract_features(model, data_loader, org_feature_path)
    
    # Evaluate and print results
    results, feat_mse = evaluate_depth(model, data_loader, org_feature_path, rec_feature_path, backbone_model)
    
    # print("a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel")
    print(f"\nRMSE: {results[0][4]:.8f}")
    print(f"Feature MSE: {feat_mse:.8f}")


def vtm_baseline_evaluation():
    # Set up paths
    config_path = 'cfg/dinov2_vitg14_nyu_linear4_config.py'
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = "/home/gaocs/models/dinov2/dinov2_vitg14_nyu_linear4_head.pth"
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/source/NYU_Test'
    source_split_name = 'nyu_test.txt'  # put it at the same folder as the source_img_path
    org_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test"
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/vtm_baseline'; print('vtm_root_path: ', vtm_root_path)

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

    # Evaluate and print results
    max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]
    trun_high = [1, 2, 10, 20]; trun_low = [-1, -2, -10, -20]

    trun_flag = True
    samples = 0; bit_depth = 10; quant_type = 'uniform'

    if trun_flag == False: trun_high = max_v; trun_low = min_v

    QPs = [42]
    for QP in QPs:
        print(trun_low, trun_high, samples, bit_depth, quant_type, QP)
        rec_feature_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP{QP}"
        results, feat_mse = evaluate_depth(model, data_loader, org_feature_path, rec_feature_path, backbone_model)

        # # print("a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel")
        print(f"\nRMSE: {results[0][4]:.8f}")
        print(f"Feature MSE: {feat_mse:.8f}")

def hyperprior_baseline_evaluation():
    # Set up paths
    config_path = 'cfg/dinov2_vitg14_nyu_linear4_config.py'
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = "/home/gaocs/models/dinov2/dinov2_vitg14_nyu_linear4_head.pth"
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/source/NYU_Test'
    source_split_name = 'nyu_test.txt'  # put it at the same folder as the source_img_path
    org_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test"
    root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/hyperprior'; print('root_path: ', root_path)

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

    # Evaluate and print results
    max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1,2,10,10]; trun_low = [-1,-2,-10,-10]
    lambda_value_all = [0.001, 0.005, 0.02, 0.05, 0.12]
    epochs = 200; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    trun_flag = True
    samples = 0; bit_depth = 1; quant_type = 'uniform'

    if trun_flag == False: trun_high = max_v; trun_low = min_v

    for lambda_value in lambda_value_all:
        print(trun_low, trun_high, samples, bit_depth, quant_type, lambda_value)
        if isinstance(trun_low, list):
            trun_low = '[' + ','.join(map(str, trun_low)) + ']'
            trun_high = '[' + ','.join(map(str, trun_high)) + ']'
        rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
        results, feat_mse = evaluate_depth(model, data_loader, org_feature_path, rec_feature_path, backbone_model)

        # # print("a1, a2, a3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel")
        print(f"\nRMSE: {results[0][4]:.8f}")
        print(f"Feature MSE: {feat_mse:.8f}")

# # run below to evaluate the reconstructed features
# if __name__ == "__main__":
#     vtm_baseline_evaluation()
#     # hyperprior_baseline_evaluation()

# run below to extract original features as the dataset. 
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    config_path = 'cfg/dinov2_vitg14_nyu_linear4_config.py'
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = "/home/gaocs/models/dinov2/dinov2_vitg14_nyu_linear4_head.pth"
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/source/NYU_Test'
    source_split_name = 'nyu_test_16.txt'
    org_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test"
    rec_feature_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/feature_test"
    
    dpt_pipeline(config_path, backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_split_name, org_feature_path, rec_feature_path)