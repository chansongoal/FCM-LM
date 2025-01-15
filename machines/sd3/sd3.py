import os
import re
import time 
import math
import random
import numpy as np
from PIL import Image
from functools import partial
import torch
from diffusers import StableDiffusion3Pipeline
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance


random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
np.random.seed(3407)


def get_captions(source_captions_name):
    image_names = []
    captions = []

    with open(source_captions_name, 'r') as f:
        for line in f:
            image_name, caption = line.split(" ", 1)
            image_names.append(image_name)
            captions.append(caption.strip())

    return captions, image_names

def extract_features_parallel(sd3_pipeline, source_captions, image_names, org_feature_path, org_image_path):
    # source_captions = source_captions[:16]
    batch_size = 16; num_batch = math.ceil(len(source_captions)/batch_size) # please keep the batch size as 16 to extract identical test features. different batch size or total number source_captions results in different features
    for bs_idx in range(num_batch):
        start = bs_idx * batch_size
        end = (bs_idx+1) * batch_size if (bs_idx+1) * batch_size < len(source_captions) else len(source_captions)
        batch_capations = source_captions[start:end]
        batch_images = image_names[start:end]

        # Generate original features when output_type="latent" 
        feats = sd3_pipeline(
                            prompt=batch_capations,
                            negative_prompt="",
                            num_inference_steps=28,
                            height=1024,
                            width=1024,
                            guidance_scale=7.0,
                            output_type="latent"    # output features rather than images
        )
        feats = feats[0].cpu().numpy(); print(feats.shape)
        for idx, caption in enumerate(batch_capations):
            feat_name = os.path.join(org_feature_path, batch_images[idx][:-4] + '.npy')
            feat = feats[idx,:,:,:]; feat = np.expand_dims(feat, axis=0)
            np.save(feat_name, feat)


        # # Generate images when output_type="pil" 
        # imgs = sd3_pipeline(
        #                     prompt=batch_capations,
        #                     negative_prompt="",
        #                     num_inference_steps=28,
        #                     height=1024,
        #                     width=1024,
        #                     guidance_scale=7.0,
        #                     output_type="pil"    # output features rather than images
        # )
        # imgs = imgs[0]
        # for idx, caption in enumerate(batch_capations):
        #     img_name = os.path.join(org_image_path, batch_images[idx][:-4] + '.png')
        #     img = imgs[idx]
        #     img.save(img_name)

def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().detach().permute(0, 2, 3, 1).float().numpy()
    return images

def numpy_to_pil(images: np.ndarray) -> list[Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def feat_to_image_parallel(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    image_names = image_names[:10]
    batch_size = 4; num_batch = math.ceil(len(source_captions)/batch_size)
    num_batch = 1
    for bs_idx in range(num_batch):
        start = bs_idx * batch_size
        end = (bs_idx+1) * batch_size if (bs_idx+1) * batch_size < len(source_captions) else len(source_captions)
        batch_capations = source_captions[start:end]
        batch_images = image_names[start:end]
        
        batch_rec_feat = []
        for idx, image_name in enumerate(batch_images):
            rec_feat_name = os.path.join(rec_feature_path, image_name[:-4]+'.npy')
            rec_feat = np.load(rec_feat_name)
            batch_rec_feat.append(rec_feat)
        batch_rec_feat = np.asarray(batch_rec_feat)[:,0,:,:,:]    
        batch_rec_feat = torch.from_numpy(batch_rec_feat).to('cuda')

        batch_pt_image = sd3_pipeline.generate_image_from_latents(latents=batch_rec_feat, output_type="pt")

        for idx, image_name in enumerate(batch_images):
            rec_img_name = os.path.join(rec_image_path, image_name[:-4]+'.png')
            pt_image = batch_pt_image[idx]
            numpy_image = pt_to_numpy(pt_image)

            pil_image = numpy_to_pil(numpy_image)[0]
            pil_image.save(rec_img_name)

def feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:16]
    for idx, image_name in enumerate(image_names): 
        rec_feat_name = os.path.join(rec_feature_path, image_name[:-4]+'.npy')
        rec_img_name = os.path.join(rec_image_path, image_name[:-4]+'.png')

        rec_feat = np.load(rec_feat_name)
        # print(idx, image_name, rec_feat.shape)
        rec_feat = torch.from_numpy(rec_feat).to('cuda')
        
        caption = source_captions[idx]

        # Generate the image
        pt_image = sd3_pipeline.generate_image_from_latents(latents=rec_feat, output_type="pil")
        # numpy_image = pt_to_numpy(pt_image)
        # pil_image = numpy_to_pil(numpy_image)[0]
        pil_image = pt_image[0]
        pil_image.save(rec_img_name)
    print(f"Feat to image time: {(time.time()-start_time):.2f}")

def tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:10]
    rec_image_all = []
    for idx, caption in enumerate(source_captions): 
        rec_img_name = os.path.join(rec_image_path, image_names[idx][:-4]+'.png')
        rec_image = np.asarray(Image.open(rec_img_name))
        # rec_image = (rec_image*255).astype('uint8')   # DO NOT PERFORM THIS
        rec_image_all.append(rec_image)
        
    # Compute CLIP Score
    rec_image_all = np.asarray(rec_image_all)   # already in [0, 255], no need further conversion
    clip_score = clip_score_fn(torch.from_numpy(rec_image_all).permute(0, 3, 1, 2), source_captions[:rec_image_all.shape[0]]).detach()
    # print(clip_score)
    clip_score = round(float(clip_score), 4)
    print(f"CLIP Score: {clip_score:.4f}")
    # print(f"Feature MSE: {np.mean(mse_list):.8f}")
    print(f"CLIP Score evaluation time: {(time.time()-start_time):.2f}")

# please refer to https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html for more details
def tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path):
    start_time = time.time()
    # image_names = image_names[:10]
    rec_image_all = []
    for idx, caption in enumerate(source_captions): 
        rec_img_name = os.path.join(rec_image_path, image_names[idx][:-4]+'.png')
        rec_image = np.asarray(Image.open(rec_img_name).resize((299,299)))  # resize to 299x299
        rec_image_all.append(rec_image)
    
    org_image_all = []
    for idx, caption in enumerate(source_captions): 
        org_img_name = os.path.join(org_image_path, image_names[idx][:-4]+'.png')
        org_image = np.asarray(Image.open(org_img_name).resize((299,299)))  # resize to 299x299
        org_image_all.append(org_image)
        
    rec_image_all = np.asarray(rec_image_all)   # already in [0, 255], no need further conversion
    org_image_all = np.asarray(org_image_all)   # already in [0, 255], no need further conversion
    rec_image_tensor = torch.tensor(rec_image_all).permute(0,3,1,2)
    org_image_tensor = torch.tensor(org_image_all).permute(0,3,1,2)

    # Compute FID
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=False, input_img_size=(3, 299, 299))
    fid.update(org_image_tensor, real=True)
    fid.update(rec_image_tensor, real=False)

    print(f"FID: {float(fid.compute()):.4f}")
    print(f"FID evaluation time: {(time.time()-start_time):.2f}")

def tti_pipeline(source_captions_name, org_feature_path, org_image_path, rec_feature_path, rec_image_path, vae_checkpoint_path, sd3_checkpoint_path):
    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    # Extract features
    os.makedirs(org_feature_path, exist_ok=True)
    os.makedirs(org_image_path, exist_ok=True)
    # extract_features_parallel(sd3_pipeline, source_captions, image_names, org_feature_path, org_image_path)

    # Generate images and evaluate 
    os.makedirs(rec_image_path, exist_ok=True)
    feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
    
    tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

    clip_score_fn = partial(clip_score, model_name_or_path=vae_checkpoint_path)
    tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
    
def vtm_baseline_evaluation():
    # Setup related path
    source_captions_name = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select100.txt"
    vae_checkpoint_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/pretrained_head/clip-vit-base-patch16"
    sd3_checkpoint_path = "/home/gaocs/models/StableDiffusion/stable-diffusion-3-medium-diffusers"

    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test_all'
    org_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/image_test'  # use the images generated from original features as the anchor
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/sd3/tti/vtm_baseline'; print('vtm_root_path: ', vtm_root_path)

    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176
    QPs = [22, 27, 32, 37, 42]
    
    trun_flag = False
    samples = 0; bit_depth = 10; quant_type = 'uniform'
    
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    for QP in QPs:
        print(trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth, QP)
        rec_feature_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP{QP}"
        rec_image_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP{QP}_image"

        # Generate images 
        os.makedirs(rec_image_path, exist_ok=True)
        feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
        
        # Evaluation
        tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

        clip_score_fn = partial(clip_score, model_name_or_path=vae_checkpoint_path)
        tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)

def hyperprior_baseline_evaluation():
    # Setup related path
    source_captions_name = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select100.txt"
    vae_checkpoint_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/pretrained_head/clip-vit-base-patch16"
    sd3_checkpoint_path = "/home/gaocs/models/StableDiffusion/stable-diffusion-3-medium-diffusers"

    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test'
    org_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/image_test'  # use the images generated from original features as the anchor
    root_path = f'/home/gaocs/projects/FCM-LM/Data/sd3/tti/hyperprior'; print('root_path: ', root_path)

    # Obtain source captions
    source_captions, image_names = get_captions(source_captions_name)

    # Setup models
    sd3_pipeline = StableDiffusion3Pipeline.from_pretrained(sd3_checkpoint_path, torch_dtype=torch.float16)
    sd3_pipeline.enable_model_cpu_offload()

    max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176
    lambda_value_all = [0.005, 0.01, 0.02, 0.05, 0.2]
    epochs = 60; learning_rate = "1e-4"; batch_size = 32; patch_size = "512 512"   # height first, width later

    trun_flag = False
    samples = 0; bit_depth = 1; quant_type = 'uniform'
    
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    for lambda_v in lambda_value_all:
        print(trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth, lambda_v)
        rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/" \
                           f"lambda{lambda_v}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
        rec_image_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/" \
                         f"lambda{lambda_v}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_image"

        # # for scaling only 
        # rec_feature_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/hyperprior/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth1"
        # rec_image_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/hyperprior/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth1_image"

        # Generate images 
        os.makedirs(rec_image_path, exist_ok=True)
        feat_to_image(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)
        
        # Evaluation
        tti_evaluate_fid(sd3_pipeline, source_captions, image_names, org_feature_path, rec_feature_path, org_image_path, rec_image_path)

        clip_score_fn = partial(clip_score, model_name_or_path=vae_checkpoint_path)
        tti_evaluate_clip_score(sd3_pipeline, clip_score_fn, source_captions, image_names, org_feature_path, rec_feature_path, rec_image_path)

# if __name__ == "__main__":
#     # vtm_baseline_evaluation()
#     hyperprior_baseline_evaluation()


# run below to extract original features as the dataset. 
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    source_captions_name = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select100.txt"
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test'
    org_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/image_test'  # use the images generated from original features as the anchor
    rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test'
    rec_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/image_test'
    # rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/vtm_baseline/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth10/QP0'
    # rec_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/vtm_baseline/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth10/QP0_image'
    # rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/hyperprior/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth1'
    # rec_image_path = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/hyperprior/postprocessed/trunl-6.176_trunh4.668_uniform0_bitdepth1_image'
    vae_checkpoint_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/pretrained_head/clip-vit-base-patch16"
    sd3_checkpoint_path = "/home/gaocs/models/StableDiffusion/stable-diffusion-3-medium-diffusers"

    tti_pipeline(source_captions_name, org_feature_path, org_image_path, rec_feature_path, rec_image_path, vae_checkpoint_path, sd3_checkpoint_path)


