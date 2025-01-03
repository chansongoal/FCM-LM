import threading
import os
from PIL import Image
import subprocess as subp
import numpy as np
import glob
from tqdm import tqdm
import time


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

    quant_feat = quant_feat.astype(np.uint16) if bit_depth==10 else quant_feat.astype(np.uint8)
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


def vtm_encoding(preprocessed_yuv_name, bitstream_name, compress_log_name, wdt, hgt, qp, quant_bits):
    stdout_vtm = open(f"{compress_log_name}", 'w')
    preprocessed_yuv_name = "\"" + preprocessed_yuv_name + "\""
    bitstream_name = "\"" + bitstream_name + "\"" 
    subp.run(f"../vtm_coding/EncoderAppStatic -c ../vtm_coding/encoder_intra_vtm.cfg -i {preprocessed_yuv_name} -o \"\" -b {bitstream_name} -q {qp} --ConformanceWindowMode=1 -wdt {wdt} -hgt {hgt} -f 1 -fr 1 --InternalBitDepth={quant_bits} --InputBitDepth={quant_bits} --InputChromaFormat=400 --OutputBitDepth={quant_bits}",
            stdout=stdout_vtm, shell=True)


def vtm_decoding(bitstream_name, decoded_yuv_name, decompress_log_name):
    stdout_vtm = open(f"{decompress_log_name}", 'w')
    bitstream_name = "\"" + bitstream_name + "\"" 
    decoded_yuv_name = "\"" + decoded_yuv_name + "\"" 
    subp.run(f"../vtm_coding/DecoderAppStatic -b {bitstream_name} -o {decoded_yuv_name}", stdout=stdout_vtm, shell=True)


def vtm_pipeline(org_feat_path, vtm_root_path, max_v, min_v, trun_high, trun_low, QP, quant_bits, model_type):
    # Set related paths
    preprocessed_yuv_path = f"{vtm_root_path}/preprocessed/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    bitstream_path = f"{vtm_root_path}/bitstream/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(bitstream_path, exist_ok=True)
    decoded_yuv_path = f"{vtm_root_path}/decoded/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(decoded_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(postprocessed_feat_path, exist_ok=True)
    encoding_log_path = f"{vtm_root_path}/encoding_log/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(encoding_log_path, exist_ok=True)
    decoding_log_path = f"{vtm_root_path}/decoding_log/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(decoding_log_path, exist_ok=True)

    feat_names = os.listdir(org_feat_path)
    # feat_names = feat_names[:1]
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        org_feat_name = os.path.join(org_feat_path, f"{feat_name[:-4]}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{feat_name[:-4]}.yuv"); #print(preprocessed_yuv_name)
        bitstream_name = os.path.join(bitstream_path, f"{feat_name[:-4]}.bin"); #print(bitstream_name)
        decoded_yuv_name = os.path.join(decoded_yuv_path, f"{feat_name[:-4]}.yuv"); #print(decoded_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{feat_name[:-4]}.npy"); #print(postprocessed_feat_name)
        encoding_log_name = os.path.join(encoding_log_path, f"{feat_name[:-4]}.txt"); #print(encoding_log_name)
        decoding_log_name = os.path.join(decoding_log_path, f"{feat_name[:-4]}.txt"); #print(decoding_log_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        print(feat_name, N,C,H,W)

        # Preprocessing
        trun_feat = truncation(org_feat, trun_low, trun_high); #print(trun_feat[0,0])
        quant_feat = uniform_quantization(trun_feat, trun_low, trun_high, quant_bits); #print(quant_feat[0,0])
        pack_feat = packing(quant_feat, model_type)
        with open(preprocessed_yuv_name, 'wb') as f:
            pack_feat.tofile(f)

        # VTM encoding
        vtm_encoding(preprocessed_yuv_name, bitstream_name, encoding_log_name, pack_feat.shape[1], pack_feat.shape[0], QP, quant_bits)

        # VTM decoding
        vtm_decoding(bitstream_name, decoded_yuv_name, decoding_log_name)

        # Load decoded YUV
        decoded_yuv = np.zeros_like(quant_feat)
        with open(decoded_yuv_name, 'rb') as f:
            decoded_yuv = np.fromfile(f, dtype=np.uint16 if quant_bits==10 else np.uint8) # save converted YUV file to dist 
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape) #(H,W)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, [N,C,H,W], model_type)
        dequant_feat = uniform_dequantization(unpack_feat, trun_low, trun_high, quant_bits)
        if model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        np.save(postprocessed_feat_name, dequant_feat)

def vtm_decode_only(org_feat_path, vtm_root_path, max_v, min_v, trun_high, trun_low, QP, quant_bits, model_type):
    # Set related paths
    preprocessed_yuv_path = f"{vtm_root_path}/preprocessed/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}"; os.makedirs(preprocessed_yuv_path, exist_ok=True)
    bitstream_path = f"{vtm_root_path}/bitstream/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(bitstream_path, exist_ok=True)
    decoded_yuv_path = f"{vtm_root_path}/decoded/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(decoded_yuv_path, exist_ok=True)
    postprocessed_feat_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(postprocessed_feat_path, exist_ok=True)
    encoding_log_path = f"{vtm_root_path}/encoding_log/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(encoding_log_path, exist_ok=True)
    decoding_log_path = f"{vtm_root_path}/decoding_log/trunl{trun_low}_trunh{trun_high}_uniquant{quant_bits}/QP{QP}"; os.makedirs(decoding_log_path, exist_ok=True)

    feat_names = os.listdir(bitstream_path)
    feat_names = feat_names[:1]
    for idx, feat_name in enumerate(feat_names):
        # Set related names
        org_feat_name = os.path.join(org_feat_path, f"{feat_name[:-4]}.npy"); #print(org_feat_name)
        preprocessed_yuv_name = os.path.join(preprocessed_yuv_path, f"{feat_name[:-4]}.yuv"); #print(preprocessed_yuv_name)
        bitstream_name = os.path.join(bitstream_path, f"{feat_name[:-4]}.bin"); #print(bitstream_name)
        decoded_yuv_name = os.path.join(decoded_yuv_path, f"{feat_name[:-4]}.yuv"); #print(decoded_yuv_name)
        postprocessed_feat_name = os.path.join(postprocessed_feat_path, f"{feat_name[:-4]}.npy"); #print(postprocessed_feat_name)
        encoding_log_name = os.path.join(encoding_log_path, f"{feat_name[:-4]}.txt"); #print(encoding_log_name)
        decoding_log_name = os.path.join(decoding_log_path, f"{feat_name[:-4]}.txt"); #print(decoding_log_name)
        
        # Load original feature
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        print(feat_name, N,C,H,W)

        # Preprocessing
        pack_feat = packing(org_feat, model_type)

        # VTM decoding
        vtm_decoding(bitstream_name, decoded_yuv_name, decoding_log_name)

        # Load decoded YUV
        decoded_yuv = np.zeros_like(pack_feat)
        with open(decoded_yuv_name, 'rb') as f:
            decoded_yuv = np.fromfile(f, dtype=np.uint16 if quant_bits==10 else np.uint8) # save converted YUV file to dist 
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape) #(H,W)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, [N,C,H,W], model_type)
        dequant_feat = dequantization(unpack_feat, trun_low, trun_high, quant_bits)
        if model_type == 'sd3': dequant_feat = dequant_feat.astype(np.float16)
        np.save(postprocessed_feat_name, dequant_feat)


if __name__ == "__main__":
    # model_type = 'llama3'; task = 'csr'
    # max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1, 2, 10, 20]; trun_low = [-1, -2, -10, -20]
    
    model_type = 'sd3'; task = 'tti'
    max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176

    quant_bits = 10
    org_feat_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/feature_test_all'
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/{model_type}/{task}/vtm_baseline'
    QPs = [22]
    for QP in QPs:
        vtm_pipeline(org_feat_path, vtm_root_path, max_v, min_v, trun_high, trun_low, QP, quant_bits, model_type)