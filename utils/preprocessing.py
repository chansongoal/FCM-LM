import os
import numpy as np
import subprocess as subp
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt 
import json
import time


def truncation(feat, trun_low, trun_high):
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)
    
    return trun_feat


def load_quantization_points(file_path: str or list[str]):
    """
    Load quantization points from a file or a list of files.
    
    Parameters:
        file_path (Union[str, List[str]]): Path to load the quantization points from.
            Can be a single file path (str) or a list of file paths (List[str]).
    
    Returns:
        Union[numpy.ndarray, List[numpy.ndarray]]: Loaded quantization points. If `file_path`
            is a single path, returns a single numpy.ndarray. If `file_path` is a list of paths,
            returns a list of numpy.ndarray.
    """
    def load_file(path):
        with open(path, 'r') as f:
            quantization_points = np.array(json.load(f))
        # print(f"Quantization points loaded from {path}")
        return quantization_points

    if isinstance(file_path, list):
        # Load quantization points from each file in the list
        return [load_file(path) for path in file_path]
    elif isinstance(file_path, str):
        # Load quantization points from a single file
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")


def uniform_quantization(feat, min_v, max_v, bit_depth):
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        quant_feat = ((feat-min_v) * scale)

    quant_feat = quant_feat.astype(np.uint16) if bit_depth>8 else quant_feat.astype(np.uint8)
    return quant_feat

def uniform_dequantization(feat, min_v, max_v, bit_depth):
    feat = feat.astype(np.float32)
    dequant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            dequant_feat[:,idx,:,:] = feat[:,idx,:,:] / scale + min_v[idx]
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        dequant_feat = feat / scale + min_v
    return dequant_feat

def uniform_scaling(feat, min_v, max_v, bit_depth):
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
            quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
    else:
        scale = ((2**bit_depth) -1) / (max_v - min_v)
        quant_feat = ((feat-min_v) * scale)

    return quant_feat

def nonlinear_quantization(data, quantization_points, bit_depth):
    """
    Apply quantization to data using a single or multiple sets of quantization points.
    
    Parameters:
        data (numpy.ndarray): Original floating-point array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).
    
    Returns:
        numpy.ndarray: Quantized integer array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        num_levels = len(quantization_points)
        data_flat = data.flatten()
        quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
        quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)
        quantized_data = quantized_data_flat.reshape(data.shape)
    elif isinstance(quantization_points, list):
        if len(quantization_points) != data.shape[1]:
            raise ValueError("Length of quantization_points list must match the number of channels (C) in data.")
        
        quantized_data = np.zeros_like(data, dtype=int)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            num_levels = len(qp)
            channel_data = data[:, i, :, :]
            channel_data_flat = channel_data.flatten()
            quantized_channel_flat = np.digitize(channel_data_flat, qp) - 1
            quantized_channel_flat = np.clip(quantized_channel_flat, 0, num_levels - 1)
            quantized_data[:, i, :, :] = quantized_channel_flat.reshape(channel_data.shape)
    else:
        raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
    
    quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)
    return quantized_data

def nonlinear_dequantization(quantized_data, quantization_points):
    """
    Dequantize quantized data back to its approximate original floating-point values.
    
    Parameters:
        quantized_data (numpy.ndarray): Quantized integer array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]): 
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).
    
    Returns:
        numpy.ndarray: Dequantized floating-point array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        quantization_points = np.sort(quantization_points)  # Ensure points are sorted
        dequantized_data = quantization_points[quantized_data]
    elif isinstance(quantization_points, list):
        if len(quantization_points) != quantized_data.shape[1]:
            raise ValueError("Length of quantization_points list must match the number of channels (C) in quantized_data.")
        
        dequantized_data = np.zeros_like(quantized_data, dtype=np.float32)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            qp = np.sort(qp)  # Ensure points are sorted
            channel_data = quantized_data[:, i, :, :]
            dequantized_data[:, i, :, :] = qp[channel_data]
    else:
        raise ValueError("quantization_points must be a numpy array or a list of numpy arrays.")
    
    # print(dequantized_data.dtype)
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data


def packing(feat, model_type):
    N, C, H, W = feat.shape
    if model_type == 'llama3':
        feat = feat[0,0,:,:]
    elif model_type == 'dinov2':
        feat = feat.transpose(0,2,1,3).reshape(N*H,C*W)
    elif model_type == 'sd3':
        feat = feat.reshape(int(C/4), int(C/4), H, W).transpose(0, 2, 1, 3).reshape(int(C/4*H), int(C/4*W)) 
    return feat

def unpacking(feat, shape, model_type):
    N, C, H, W = shape
    if model_type == 'llama3':
        feat = np.expand_dims(feat, axis=0); feat = np.expand_dims(feat, axis=0)
    elif model_type == 'dinov2':
        feat = feat.reshape(N,H,C,W).transpose(0, 2, 1, 3) 
    elif model_type == 'sd3':
        feat = feat.reshape(int(C/4), H, int(C/4), W).transpose(0,2,1,3).reshape(N,C,H,W)
    return feat


def trun_quant_pipeline(org_feat_path, org_sample_name, root_path, model_type, trun_flag, samples, max_v, min_v, trun_high, trun_low, quant_type, bit_depth):
    # Set related paths
    preprocessed_yuv_path = f"{root_path}/preprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP0"; os.makedirs(postprocessed_feat_path, exist_ok=True)

    # feat_names = os.listdir(org_feat_path)
    feat_names = []
    with open(org_sample_name, 'r', encoding='utf-8') as file:
        # Read the entire file content
        for content in file:
            # Split the content by spaces
            feat_names.append(content.strip())
    # print(feat_names)
    # feat_names = feat_names[:1]
    mse_all = []
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        org_feat_name = os.path.join(org_feat_path, f"{feat_name}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{feat_name}.yuv"); #print(preprocessed_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{feat_name}.npy"); #print(postprocessed_feat_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(idx, feat_name, N,C,H,W)

        # Truncation
        if trun_flag == True:
            trun_feat = truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        # Quantization
        quant_feat = uniform_quantization(trun_feat, trun_low, trun_high, bit_depth)      

        # Dequantization
        dequant_feat = uniform_dequantization(quant_feat, trun_low, trun_high, bit_depth)      

        # Save features
        if model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        elif model_type == 'dinov2': dequant_feat = dequant_feat.astype(np.float32)
        elif model_type == 'llama3': dequant_feat = dequant_feat.astype(np.float32)
        np.save(postprocessed_feat_name, dequant_feat)
        mse = np.mean((org_feat-dequant_feat)**2)
        mse_all.append(mse)
    print('Average MSE: ', np.mean(mse_all))

def trun_scale_pipeline(org_feat_path, org_sample_name, root_path, model_type, trun_flag, samples, max_v, min_v, trun_high, trun_low, quant_type, bit_depth):
    # Set related paths
    preprocessed_yuv_path = f"{root_path}/preprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}"; os.makedirs(postprocessed_feat_path, exist_ok=True)

    # feat_names = os.listdir(org_feat_path)
    feat_names = []
    with open(org_sample_name, 'r', encoding='utf-8') as file:
        # Read the entire file content
        for content in file:
            # Split the content by spaces
            feat_names.append(content.strip())
    # feat_names = feat_names[:1]
    mse_all = []
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        org_feat_name = os.path.join(org_feat_path, f"{feat_name}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{feat_name}.yuv"); #print(preprocessed_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{feat_name}.npy"); #print(postprocessed_feat_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(idx, feat_name, N,C,H,W)

        # Truncation
        if trun_flag == True:
            trun_feat = truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        # Scaling
        quant_feat = uniform_scaling(trun_feat, trun_low, trun_high, bit_depth)      

        # Dequantization
        dequant_feat = uniform_dequantization(quant_feat, trun_low, trun_high, bit_depth)      

        # Save features
        if model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        elif model_type == 'dinov2': dequant_feat = dequant_feat.astype(np.float32)
        elif model_type == 'llama3': dequant_feat = dequant_feat.astype(np.float32)
        np.save(postprocessed_feat_name, dequant_feat)
        mse = np.mean((org_feat-dequant_feat)**2)
        mse_all.append(mse)
    print('Average MSE: ', np.mean(mse_all))

if __name__ == "__main__":
    model_type = 'llama3'; task = 'csr'
    max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5  # Remember to set different truncation regions for vtm and hyperprior!

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1,2,10,10]; trun_low = [-1,-2,-10,-10]
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176

    trun_flag = True
    samples = 0; bit_depth = 10; quant_type = 'uniform'
    
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    org_sample_name = fr'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/source/arc_challenge_test_longest100_name.txt'
    # org_sample_name = fr'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/source/captions_val2017_select100_vtm.txt'

    encoder = 'vtm_baseline'
    org_feat_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/feature_test_all'; print('org_feat_path: ', org_feat_path)
    root_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/{encoder}'; print('root_path: ', root_path)
    print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)
    trun_quant_pipeline(org_feat_path, org_sample_name, root_path, model_type, trun_flag, samples, max_v, min_v, trun_high, trun_low, quant_type, bit_depth)

    # Remember to set different truncation regions for vtm and hyperprior!
    bit_depth = 1
    encoder = 'hyperprior'
    org_feat_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/feature_test_all'; print('org_feat_path: ', org_feat_path)
    root_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/{encoder}'; print('root_path: ', root_path)
    print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)
    trun_scale_pipeline(org_feat_path, org_sample_name, root_path, model_type, trun_flag, samples, max_v, min_v, trun_high, trun_low, quant_type, bit_depth)