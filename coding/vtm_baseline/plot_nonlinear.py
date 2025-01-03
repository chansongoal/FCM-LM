import os
import numpy as np
import subprocess as subp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import json
from typing import Union, List
from collections import Counter


def truncation(feat, trun_low, trun_high):
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)
    
    return trun_feat

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
    
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data
   

def sample_data(data, max_points=5000):
    num_elements = np.prod(data.shape)
    if num_elements > max_points:
        sampled_indices = np.random.choice(num_elements, size=max_points, replace=False)
        sampled_data = data.flatten()[sampled_indices]
        sampled_data.reshape(-1)
        return sampled_data
    return data

def plot_quantized_data_hist(quantized_data, log_flag, pdf_name, bit_depth):
    # Set global font
    fontsize=26
    font = {'family': 'Times New Roman', 'size': fontsize}
    plt.rc('font', **font)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Count the frequency of each unique value
    quantized_data = quantized_data.flatten()
    value_counts = Counter(quantized_data)
    unique_values = list(value_counts.keys())
    frequencies = list(value_counts.values())
    
    # Sort values and frequencies for better visualization
    sorted_indices = np.argsort(unique_values)
    sorted_values = np.array(unique_values)[sorted_indices]
    sorted_frequencies = np.array(frequencies)[sorted_indices]
    

    # Plot histograms
    color = '#018f52'
    ax1.bar(sorted_values, sorted_frequencies, color=color, alpha=0.5)
    ax1.set_xlabel('Quantized Feature Value')
    ax1.set_ylabel('Frequency', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis and plot CDF
    color = 'r'
    ax2 = ax1.twinx()
    cdf = np.cumsum(sorted_frequencies) / np.sum(sorted_frequencies)
    ax2.plot(sorted_values, cdf, color=color, linewidth=3)
    ax2.set_ylabel('CDF', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    max_value = 2 ** bit_depth - 1
    if bit_depth == 8: x_ticks = np.linspace(0, 250, 6)
    elif bit_depth == 10: x_ticks = np.linspace(0, 1000, 21) 
    ax1.set_xticks(x_ticks)
    ax1.set_xlim([0-10, max_value+10])

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

def plot_quantization_intervals(quantization_mapping_name, task, trun_low, trun_high, quant_type, samples, bit_depth):
    if quant_type == 'uniform': return
    quantization_mapping = load_quantization_points(quantization_mapping_name)
    if task == 'dpt':
        for ch in range(len(quantization_mapping)):
            suffix = f"ch{ch}" 
            mapping = quantization_mapping[ch]
            pdf_name = f'./quantization_visualization/{task}/quantization_mapping_{task}_{suffix}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.pdf'
            visualize_quantization_intervals(mapping, pdf_name)
    else:
        pdf_name = f'./quantization_visualization/{task}/quantization_mapping_{task}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.pdf'
        visualize_quantization_intervals(quantization_mapping, pdf_name)

def visualize_quantization_intervals(quantization_points, pdf_name):
    """Visualize the non-uniform quantization intervals """
    quantization_intervals = np.diff(quantization_points)

    # quantization_intervals = np.log(quantization_intervals + 1e-12)
    # quantization_intervals = np.sqrt(quantization_intervals)
    quantization_intervals = np.tanh(5*quantization_intervals)
    
    fontsize=26
    font = {'family': 'Times New Roman', 'size': fontsize}
    plt.rc('font', **font)

    colors = plt.cm.viridis(np.linspace(0, 1, len(quantization_intervals)))
    plt.figure(figsize=(9, 6))
    plt.bar(range(len(quantization_intervals)), quantization_intervals, color=colors, alpha=0.5)
    plt.xlabel('Quantization Interval Index')
    plt.ylabel('Interval Width')
    # plt.title(f'Visualization of Quantization Intervals (Bit Depth: {bit_depth})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(pdf_name, dpi=600, format='pdf')


def quant_visualization(org_feat_path, vtm_root_path, quantization_mapping_name, model_type, trun_flag, samples, trun_high, trun_low, quant_type, bit_depth):
    # Set related paths
    feat_names = os.listdir(org_feat_path)
    feat_names = feat_names[:1]
    trun_feat_list_all = []
    quant_feat_list_all = []; dequant_feat_list_all = []
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        org_feat_name = os.path.join(org_feat_path, feat_name); #print(org_feat_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        # print(feat_name, N,C,H,W)
        if task == 'csr':
            org_feat = org_feat[:, :, :64, :]  # Crop features for 'csr' task

        # Truncation
        if trun_flag == True:
            trun_feat = truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat
        trun_feat_list_all.append(trun_feat)

        # Quantization
        if quant_type == 'uniform': 
            quant_feat = uniform_quantization(trun_feat, trun_low, trun_high, bit_depth)      
        elif quant_type == 'density' or quant_type == 'kmeans':
            quantization_mapping = load_quantization_points(quantization_mapping_name)
            quant_feat = nonlinear_quantization(trun_feat, quantization_mapping, bit_depth) 
        quant_feat_list_all.append(quant_feat)
        
        # Dequantization
        if quant_type == 'uniform': 
            dequant_feat = uniform_dequantization(quant_feat, trun_low, trun_high, bit_depth)      
        elif quant_type == 'density' or quant_type == 'kmeans':
            quantization_mapping = load_quantization_points(quantization_mapping_name)
            dequant_feat = nonlinear_dequantization(quant_feat, quantization_mapping) 
        dequant_feat_list_all.append(dequant_feat)

    trun_feat_list_all = np.asarray(trun_feat_list_all)
    quant_feat_list_all = np.asarray(quant_feat_list_all)
    dequant_feat_list_all = np.asarray(dequant_feat_list_all)
    # print(trun_feat_list_all.shape, np.max(trun_feat_list_all), np.min(trun_feat_list_all), np.mean(trun_feat_list_all))
    # print(quant_feat_list_all.shape, np.max(quant_feat_list_all), np.min(quant_feat_list_all), np.mean(quant_feat_list_all))
    # print(dequant_feat_list_all.shape, np.max(dequant_feat_list_all), np.min(dequant_feat_list_all), np.mean(dequant_feat_list_all))

    # Task-specific processing
    if task == 'dpt':
        # Process each channel separately for 'dpt' task
        for ch in range(quant_feat_list_all.shape[2]):
            trun_feat_list = trun_feat_list_all[:, :, ch, :, :]
            quant_feat_list = quant_feat_list_all[:, :, ch, :, :]
            dequant_feat_list = dequant_feat_list_all[:, :, ch, :, :]
            suffix = f"ch{ch}" 
            plot_quantized_data_hist(quant_feat_list, not trun_flag, f'./quantization_visualization/{task}/quantized_data_{task}_{suffix}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.pdf', bit_depth)
            feat_mse = np.mean((trun_feat_list-dequant_feat_list)**2)
            print(f"Feature MSE: {feat_mse:.8f}")
            # break
    else:
        # Process all features together for other tasks
        plot_quantized_data_hist(quant_feat_list_all, not trun_flag, f'./quantization_visualization/{task}/quantized_data_{task}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.pdf', bit_depth)
        feat_mse = np.mean((trun_feat_list_all-dequant_feat_list_all)**2)
        print(f"Feature MSE: {feat_mse:.8f}")



if __name__ == "__main__":
    # model_type = 'llama3'; task = 'csr'
    # max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5

    model_type = 'dinov2'; task = 'cls'
    max_v = 104.1752; min_v = -552.4510; trun_high = 30; trun_low = -30

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 20; trun_low = -20

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1, 2, 10, 20]; trun_low = [-1, -2, -10, -20]
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 3.0527; min_v = -4.0938; trun_high = 3.0527; trun_low = -4.0938 

    trun_flag = False; quant_type = 'kmeans'; samples = 10; bit_depth = 8
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    quant_type_all = ['uniform', 'density', 'kmeans']
    # quant_type_all = ['density']
    bit_depth_all = [8]

    org_feat_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/feature_test'; print('org_feat_path: ', org_feat_path)
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/vtm'; print('vtm_root_path: ', vtm_root_path)

    for bit_depth in bit_depth_all:
        for quant_type in quant_type_all:  
            if task == 'dpt':
                quantization_mapping_name = []
                for ch in range(len(trun_high)):
                    quant_mapping_name = f'/home/gaocs/projects/FCM-LM/Code/vtm_coding/quantization_mapping/{task}/quantization_mapping_{task}_ch{ch}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.json'
                    quantization_mapping_name.append(quant_mapping_name)
            else:
                quantization_mapping_name = f'/home/gaocs/projects/FCM-LM/Code/vtm_coding/quantization_mapping/{task}/quantization_mapping_{task}_trunl{trun_low}_trunh{trun_high}_{quant_type}_{samples}_bitdepth{bit_depth}.json'
            # print('quantization_mapping_name: ', quantization_mapping_name)
            
            print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)
            quant_visualization(org_feat_path, vtm_root_path, quantization_mapping_name, model_type, trun_flag, samples, trun_high, trun_low, quant_type, bit_depth)

            # Plot quantization mapping
            plot_quantization_intervals(quantization_mapping_name, task, trun_low, trun_high, quant_type, samples, bit_depth)