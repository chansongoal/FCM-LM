import os
import numpy as np
import subprocess as subp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import json
import time 
from typing import Union, List


def truncation(feat, trun_low, trun_high):
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)
    
    return trun_feat

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

def kmeans_quantization(data, bit_depth=10):
    """
    Non-uniform quantization: maps floating-point data to integers with the specified bit depth.
    
    Parameters:
        data (numpy.ndarray): Original floating-point array.
        bit_depth (int): Number of bits for quantization (default is 10).
        
    Returns:
        quantized_data (numpy.ndarray): Quantized integer array.
        quantization_points (numpy.ndarray): Quantization points.
    """
    num_levels = 2 ** bit_depth  # Number of quantization levels
    original_shape = data.shape  # Save the original shape of the data
    data_flat = data.flatten().reshape(-1,1)  # Flatten and reshape data to a column vector
    kmeans = KMeans(n_clusters=num_levels, random_state=42)
    kmeans.fit(data_flat)
    
    # Get quantization points (cluster centers)
    quantization_points = kmeans.cluster_centers_.flatten()
    quantization_points.sort()
    
    # Map data to the nearest quantization points
    quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
    quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)  # Prevent overflow
    
    # Reshape quantized data to match the original data shape
    quantized_data = quantized_data_flat.reshape(original_shape)
    quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)

    return quantized_data, quantization_points

def density_quantization(data, bit_depth=10):
    """
    Density-based non-uniform quantization: maps floating-point data to integers with the specified bit depth.
    
    Parameters:
        data (numpy.ndarray): Original floating-point array.
        bit_depth (int): Number of bits for quantization (default is 10).
        
    Returns:
        quantized_data (numpy.ndarray): Quantized integer array.
        quantization_points (numpy.ndarray): Quantization points.
    """
    # Step 1: Compute the number of quantization levels
    num_levels = 2 ** bit_depth  # Number of quantization levels

    # Step 2: Flatten data and sort
    data_flat = data.flatten()
    sorted_data = np.sort(data_flat)

    # Step 3: Compute CDF and determine quantization points
    cdf = np.linspace(0, 1, len(sorted_data))  # CDF values for the sorted data
    quantization_indices = np.linspace(0, 1, num_levels)  # Target CDF levels for quantization points
    quantization_points = np.interp(quantization_indices, cdf, sorted_data)  # Map target CDF levels to data values

    # Step 4: Map data to the nearest quantization points
    quantized_indices = np.digitize(data_flat, quantization_points) - 1
    quantized_indices = np.clip(quantized_indices, 0, num_levels - 1)  # Ensure indices are within valid range

    # Step 5: Reshape quantized data back to the original shape
    quantized_data = quantized_indices.reshape(data.shape)
    quantized_data = quantized_data.astype(np.uint16) if bit_depth>8 else quantized_data.astype(np.uint8)
    
    return quantized_data, quantization_points

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
    
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data

def save_quantization_points(quantization_points, file_path):
    """
    Save quantization points to a file.
    
    Parameters:
        quantization_points (numpy.ndarray): Array of quantization points.
        file_path (str): Path to save the quantization points.
    """
    with open(file_path, 'w') as f:
        json.dump(quantization_points.tolist(), f)
    # print(f"Quantization points saved to {file_path}")

def load_quantization_points(file_path: Union[str, list[str]]):
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
        print(f"Quantization points loaded from {path}")
        return quantization_points

    if isinstance(file_path, list):
        # Load quantization points from each file in the list
        return [load_file(path) for path in file_path]
    elif isinstance(file_path, str):
        # Load quantization points from a single file
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")

def sample_data(data, max_points=5000):
    num_elements = np.prod(data.shape)
    if num_elements > max_points:
        sampled_indices = np.random.choice(num_elements, size=max_points, replace=False)
        sampled_data = data.flatten()[sampled_indices]
        sampled_data.reshape(-1)
        return sampled_data
    return data

def plot_quantized_data_hist(quantized_data, log_flag, bit_depth, pdf_name):
    # Set global font
    fontsize=26
    font = {'family': 'Times New Roman', 'size': fontsize}
    plt.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot histograms
    color = '#018f52'
    quantized_data = quantized_data.flatten()
    ax1.hist(quantized_data, bins=2**bit_depth, log=log_flag, edgecolor=color, alpha=0.5)
    ax1.set_xlabel('Quantized Feature Value')
    ax1.set_ylabel('Frequency', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis and plot CDF
    color = 'r'
    ax2 = ax1.twinx()

    sorted_data = np.sort(quantized_data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax2.plot(sorted_data, cdf, color=color, linewidth=3)

    ax2.set_ylabel('CDF', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Customize plot
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf', bbox_inches='tight')

def plot_quantization_mapping(data, uniform_points, density_points, kmeans_points, pdf_name):
    """
    Plot the mapping of original data values to quantized integer values for uniform and non-uniform quantization.

    Parameters:
        data (numpy.ndarray): Original data array.
        uniform_points (numpy.ndarray): Uniform quantization points.
        non_uniform_points (numpy.ndarray): Non-uniform quantization points.
    """
    fontsize=26
    font = {'family': 'Times New Roman', 'size': fontsize}
    plt.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    # Compute mappings
    uniform_mapping = np.digitize(data, uniform_points) - 1
    uniform_mapping = np.clip(uniform_mapping, 0, len(uniform_points) - 1)

    density_mapping = np.digitize(data, density_points) - 1
    density_mapping = np.clip(density_mapping, 0, len(density_points) - 1)

    kmeans_mapping = np.digitize(data, kmeans_points) - 1
    kmeans_mapping = np.clip(kmeans_mapping, 0, len(kmeans_points) - 1)

    # Plot the mappings
    plt.scatter(data, uniform_mapping, s=5, alpha=0.3, color='red', label='Uniform')
    plt.scatter(data, density_mapping, s=5, alpha=0.3, color='blue', label='Density')
    plt.scatter(data, kmeans_mapping, s=5, alpha=0.3, color='green', label='KMeans')

    # Customize plot
    # plt.title('Quantization Mapping: Original to Quantized Values')
    plt.xlabel('Original Feature Value')
    plt.ylabel('Quantized Feature Value')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf')

def nonlinear_quantization(org_feat_path, vtm_root_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth):
    """
    Combined function to handle quantization for different tasks based on task type.
    
    Parameters:
        org_feat_path (str): Path to the original features.
        vtm_root_path (str): Path to the VTM root directory.
        quant_mapping_path (str): Path to save quantization mappings.
        model_type (str): Model type.
        task (str): Task name ('dpt', 'csr', etc.).
        samples (int): Number of samples to process.
        trun_flag (bool): Whether to apply truncation.
        trun_high (float or List[float]): High truncation limit.
        trun_low (float or List[float]): Low truncation limit.
        quant_type (str): Quantization type.
        bit_depth (int): Bit depth for quantization.
    """
    feat_names = os.listdir(org_feat_path)[:samples]
    feat_list_all = []

    # Load and preprocess features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")

        if trun_flag:
            trun_feat = truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        if task == 'csr':
            trun_feat = trun_feat[:, :, :64, :]  # Crop features for 'csr' task
        feat_list_all.append(trun_feat)

    feat_list_all = np.asarray(feat_list_all)
    # print(f"Loaded features shape: {feat_list_all.shape}")

    # Task-specific processing
    if task == 'dpt':
        # Process each channel separately for 'dpt' task
        for ch in range(feat_list_all.shape[2]):
            feat_list = feat_list_all[:, :, ch, :, :]
            process_quantization(feat_list, quant_mapping_path, task, bit_depth, trun_low[ch], trun_high[ch], samples, trun_flag, ch)
    else:
        # Process all features together for other tasks
        feat_list = feat_list_all
        process_quantization(feat_list, quant_mapping_path, task, bit_depth, trun_low, trun_high, samples, trun_flag)

def process_quantization(feat_list, quant_mapping_path, task, bit_depth, trun_low, trun_high, samples, trun_flag, ch=None):
    """
    Helper function to process quantization for a given feature list.
    
    Parameters:
        feat_list (numpy.ndarray): Feature list to process.
        quant_mapping_path (str): Path to save quantization mappings.
        task (str): Task name.
        bit_depth (int): Bit depth for quantization.
        trun_low (float): Low truncation limit.
        trun_high (float): High truncation limit.
        samples (int): Number of samples.
        trun_flag (bool): Whether truncation is applied.
        ch (int or None): Channel index for naming, if applicable.
    """
    # Compute uniform quantization mapping
    quant_time = time.time()
    uniform_quant_feat = uniform_quantization(feat_list, trun_low, trun_high, bit_depth)
    # print(f'Uniform_quant_time: {(time.time()-quant_time):.4f}', end=' ')
    uniform_time = time.time()-quant_time
    uniform_dequant_feat = uniform_dequantization(uniform_quant_feat, trun_low, trun_high, bit_depth)
    uniform_mse = np.mean((feat_list-uniform_dequant_feat)**2)
    # print(f"Uniform Feature MSE: {feat_mse:.8f}")
    # print(f"{feat_mse:.8f}", end=' ')

    # Compute density quantization mapping
    quant_time = time.time()
    density_quant_feat, density_points = density_quantization(feat_list, bit_depth)
    # print(f'Density_quant_time: {(time.time()-quant_time):.4f}', end=' ')
    density_time = time.time() - quant_time
    density_dequant_feat = nonlinear_dequantization(density_quant_feat, density_points)
    density_mse = np.mean((feat_list-density_dequant_feat)**2)
    # print(f"Density Feature MSE: {feat_mse:.8f}")
    # print(f"{feat_mse:.8f}", end=' ')

    # Compute kmeans quantization mapping
    quant_time = time.time()
    kmeans_quant_feat, kmeans_points = kmeans_quantization(feat_list, bit_depth)
    # print(f'KMeans_quant_time: {(time.time()-quant_time):.4f}', end=' ')
    kmeans_time = time.time() - quant_time
    kmeans_dequant_feat = nonlinear_dequantization(kmeans_quant_feat, kmeans_points)
    kmeans_mse = np.mean((feat_list-kmeans_dequant_feat)**2)
    # print(f"KMeans Feature MSE: {feat_mse:.8f}")
    # print(f"{feat_mse:.8f}")

    print(f"{uniform_mse:.8f} {density_mse:.8f} {kmeans_mse:.8f} {uniform_time:.4f} {density_time:.4f} {kmeans_time:.4f}")
    # Save quantization mapping
    suffix = f"_ch{ch}" if ch is not None else ""
    save_quantization_points(density_points, f'{quant_mapping_path}/quantization_mapping_{task}{suffix}_trunl{trun_low}_trunh{trun_high}_density_{samples}_bitdepth{bit_depth}.json')
    save_quantization_points(kmeans_points, f'{quant_mapping_path}/quantization_mapping_{task}{suffix}_trunl{trun_low}_trunh{trun_high}_kmeans_{samples}_bitdepth{bit_depth}.json')
    

if __name__ == "__main__":
    # model_type = 'llama3'; task = 'csr'
    # max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.4510; trun_high = 50; trun_low = -50

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 50; trun_low = -50

    model_type = 'dinov2'; task = 'dpt'
    max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1, 2, 10, 20]; trun_low = [-1, -2, -10, -20]
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 3.0527; min_v = -4.0938; trun_high = 3.0527; trun_low = -4.0938 

    samples = 10; quant_type = 'kmeans'; bit_depth = 8

    org_feat_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/feature_train'
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/vtm'
    quant_mapping_path = f'/home/gaocs/projects/FCM-LM/Code/vtm_coding/quantization_mapping/{task}'
    
    # print(model_type, task, trun_flag, quant_type, samples, trun_high, trun_low)

    # trun_high_all = [50, 10, 2]; trun_low_all = [-50, -10, -2]
    bit_depths = [10]
    # for idx in range(len(trun_high_all)):
    #     trun_high = trun_high_all[idx]; trun_low = trun_low_all[idx]
    #     for bit_depth in bit_depths:
    #         trun_flag = True
    #         print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)
    #         nonlinear_quantization(org_feat_path, vtm_root_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth)

    for bit_depth in bit_depths:       
        trun_flag = False
        if trun_flag == False: trun_high = max_v; trun_low = min_v
        print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)
        nonlinear_quantization(org_feat_path, vtm_root_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth)