import os
import json
# import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from dinov2.hub.classifiers import dinov2_vitg14_lc

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs


class DataFolder(datasets.ImageFolder):
    """Custom dataset class that includes the file path in the returned sample."""

    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def get_label_from_file(filename, file_path):
    with open(file_path, 'r') as f:
        file_lines = f.readlines()
    
    for line in file_lines:
        parts = line.strip().split()
        if parts[0] == filename:
            return int(parts[-1])  # Return the last element (the number)
    
    return None  # Return None if the file name is not found

def get_label_from_dataset(test_dataset):
    err_list = [22, 30, 32, 36, 54, 59, 60, 67, 68, 120, 123, 134, 137, 138, 161, 170, 173, 184, 186, 193, 196, 204, 213, 231, 236, \
                240, 241, 249, 257, 259, 265, 272, 281, 288, 292, 297, 304, 319, 343, 353, 356, 358, 369, 381, 385, 392, 397, 424, 435, \
                444, 445, 460, 463, 470, 479, 480, 482, 484, 493, 501, 508, 519, 526, 527, 531, 534, 541, 544, 550, 561, 580, 582, 601, \
                605, 608, 616, 619, 620, 630, 639, 650, 651, 664, 673, 675, 676, 691, 702, 724, 733, 742, 743, 747, 750, 754, 776, 778, \
                782, 784, 787, 789, 790, 799, 814, 815, 826, 832, 834, 835, 836, 841, 848, 851, 857, 858, 876, 880, 885, 890, 892, 908, \
                911, 925, 928, 947, 952, 961, 966, 967, 970, 972, 983, 987]
    label_for_correct = []
    total_class = 1000
    required_class = 500
    label_idx = 0
    
    image_names = [sample[0] for sample in test_dataset.samples]
    for idx, image_name in enumerate(image_names):
        img_split = image_name.split('/')
        path_name, img_name = img_split[-2], img_split[-1][:-5]
        if not idx in err_list:
            # print(img_name, idx)    # for imagenet_selected_label500.txt
            # print(path_name, img_name)  # for imagenet_selected_pathname500.txt
            print(img_name) # for cpu cluster
            label_for_correct.append(idx)
        if len(label_for_correct)==required_class:
            break
    return label_for_correct

def build_dataset(source_img_path: str, batch_size: int, transform: str ='test'):
    # Define data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load test dataset and DataLoader
    dataset = DataFolder(source_img_path, transform=data_transform["test"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataset, dataloader


def extract_features(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, org_feature_path: str):
    """Extract features from backbone"""
    model.eval()
    device = next(model.parameters()).device

    for batch_x, batch_y, img_name in tqdm(data_loader):
        batch_x = batch_x.to(device)
        feature_list = model.forward_backbone(batch_x)
        feature = feature_list[0].unsqueeze(0)

        feat_name = img_name[0].split('/')[-1].split('.')[0]    # get the img name without '.JPEG'
        np.save(f'{org_feature_path}/{feat_name}.npy', feature.cpu().detach().numpy())


def evaluate_cls(model: torch.nn.Module, org_feature_path: str, rec_feature_path: str, source_label_name: str):
    """Evaluate image classification accuracy and feature reconstruction error.

    Args:
        model (torch.nn.Module): Classification model for evaluation.
        org_feature_path (str): Path to the original features.
        rec_feature_path (str): Path to the reconstructed features.
        source_label_name (str): Path to the source label file.

    Returns:
        Accuracy, feature MSE
    """
    model.eval()
    device = next(model.parameters()).device

    eval_acc = 0.0
    eval_mse = 0.0

    # Retrieve reconstructed feature filenames
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith('.npy')]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_numpy = np.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)

        with torch.no_grad():
            # Decode features and make predictions
            pred = torch.argmax(model.forward_head(rec_features_tensor), dim=1)

            # Compute accuracy using labels
            label = get_label_from_file(rec_feat_name.split('.')[0], source_label_name)
            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            mse = np.mean(np.square(org_feat - rec_features_numpy))
            eval_mse += mse

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc*100 / num_samples, eval_mse / num_samples


def cls_pipeline(backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_label_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the evaluation."""

    batch_size = 1
    test_dataset, test_dataloader = build_dataset(source_img_path, batch_size, 'test')
    
    labels = get_label_from_dataset(test_dataset)   # comment this, only used in the first time to generate source label file

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Extract features
    # extract_features(model, test_dataloader, org_feature_path)

    # Evaluate and print results
    acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Feature MSE: {feat_mse:.8f}")


def vtm_baseline_evaluation():
    # Set up paths
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/ImageNet_Selected100'
    source_label_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/feature_test'
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/cls/vtm_baseline'; print('vtm_root_path: ', vtm_root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Evaluate and print results
    max_v = 104.1752; min_v = -552.451; trun_high = 20; trun_low = -20

    trun_flag = True; samples = 0; bit_depth = 10; quant_type = 'uniform'
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    QPs = [22]
    for QP in QPs:
        print(trun_low, trun_high, samples, bit_depth, quant_type, QP)
        rec_feature_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP{QP}"
        acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
        print(f"Classification Accuracy: {acc:.4f}")
        # print(f"Feature MSE: {feat_mse:.8f}")

def hyperprior_baseline_evaluation():
    # Set up paths
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/ImageNet_Selected100'
    source_label_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/feature_test'
    root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/cls/hyperprior'; print('root_path: ', root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Evaluate and print results
    max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5
    epochs = 800; learning_rate="1e-4"; batch_size = 128; patch_size = "256 256"
    lambda_value_all = [0.001, 0.0017, 0.003, 0.0035, 0.01]

    trun_flag = True
    samples = 0; bit_depth = 1; quant_type = 'uniform'

    if trun_flag == False: trun_high = max_v; trun_low = min_v

    for lambda_value in lambda_value_all:
        print(trun_low, trun_high, samples, bit_depth, quant_type, lambda_value)
        rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
        acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
        print(f"Classification Accuracy: {acc:.4f}")
        # print(f"Feature MSE: {feat_mse:.8f}")

# # run below to evaluate the reconstructed features
# if __name__ == "__main__":
#     # vtm_baseline_evaluation()
#     hyperprior_baseline_evaluation()

# run below to extract original features as the dataset. 
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/ImageNet_Selected100'
    source_label_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/feature_test'
    rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/feature_test'

    cls_pipeline(backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_label_name, org_feature_path, rec_feature_path)