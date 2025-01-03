# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset

import numpy as np 

def crop_array_random(arr, crop_shape): # (hight, width)
        max_row = arr.shape[0] - crop_shape[0]
        max_col = arr.shape[1] - crop_shape[1]
        
        if max_row < 0 or max_col < 0:
            print(arr.shape[0], crop_shape[0])
            print(arr.shape[1], crop_shape[1])
            raise ValueError("crop_shape exceeds the feature shape")

        start_row = np.random.randint(0, max_row + 1)
        start_col = np.random.randint(0, max_col + 1)
        
        end_row = start_row + crop_shape[0]
        end_col = start_col + crop_shape[1]
        
        return arr[start_row:end_row, start_col:end_col]

@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load an feature folder database. Training and testing feature samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # load feature
        feat = np.load(self.samples[index])

        # crop, truncation, and normalize
        filename = str(self.samples[index])

        # default crop size and truncated value, adjust according to your dataset!
        crop_shape = (64, 4096)  #(height, width), must be the multiple of 64
        lower = -10; upper = 10

        # if 'openbook' in filename or 'arc' in filename:
        #     crop_shape = (64, 4096)  #(height, width), must be the multiple of 64
        #     lower = -10; upper = 10
        # elif 'nyu' in filename:
        #     crop_shape = (256, 256)  #(height, width)
        #     if filename[-6:-4] == '10': lower = -1; upper = 1
        #     elif filename[-6:-4] == '20': lower = -2; upper = 2
        #     elif filename[-6:-4] == '30': lower = -10; upper = 10
        #     elif filename[-6:-4] == '40': lower = -20; upper = 20
        # elif 'coco' in filename:
        #     crop_shape = (512, 512)
        #     lower = 0; upper = 0

        feat = crop_array_random(feat[0,0,:,:], crop_shape)
        feat = np.expand_dims(feat, axis=0) # (C,H,W)
        feat = np.clip(feat, lower, upper)
        feat = (feat - lower) / (upper - lower)
        # print(feat.shape)
        return feat

    def __len__(self):
        return len(self.samples)
    
