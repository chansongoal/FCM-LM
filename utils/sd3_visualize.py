import os
import numpy as np
from PIL import Image


def uniform_quantization(feat, min_v, max_v, bit_depth):
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        quant_feat = ((feat-min_v) * scale)

    quant_feat = quant_feat.astype(np.uint16) if bit_depth==10 else quant_feat.astype(np.uint8)
    return quant_feat


def packing(feat, model_type):
    N, C, H, W = feat.shape
    if model_type == 'llama3':
        feat = feat[0,0,:,:]
    elif model_type == 'dinov2':
        feat = feat.transpose(0,2,1,3).reshape(N*H,C*W)
    elif model_type == 'sd3':
        feat = feat.reshape(int(C/4), int(C/4), H, W).transpose(0, 2, 1, 3).reshape(int(C/4*H), int(C/4*W)) 
    return feat

    
# Function to compute statistics for .npy files in a directory
def visualize(feat_path, visualize_path):
    filenames = os.listdir(feat_path)
    filenames = filenames[:10]
    for file_name in filenames:
        if file_name.endswith('.npy'):
            file_path = os.path.join(feat_path, file_name)
            feat = np.load(file_path)
            feat = packing(feat, 'sd3')
            feat = uniform_quantization(feat, 3.975, -5.754, 8)   
            img = Image.fromarray(feat)
            img_name = os.path.join(visualize_path, f"{os.path.splitext(file_name)[0]}.png")
            img.save(img_name)   


# Main workflow
if __name__ == "__main__":
    feat_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test"  # Replace with your directory path
    visualize_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test_visualize"
    os.makedirs(visualize_path, exist_ok=True)
    visualize(feat_path, visualize_path)