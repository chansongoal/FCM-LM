import os
import numpy as np

feat_path = '/home/gaocs/projects/FCM-LM/Data/llama3/csr/feature_test'
feat_files = os.listdir(feat_path)
shapes = []
for idx, feat_file in enumerate(feat_files):
    feat = np.load(os.path.join(feat_path, feat_file))
    shapes.append(feat.shape)
    print(feat_file, feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3])

    # print(feat.shape[2])
    # print(feat_file[:-4])

# print(np.min(shapes), np.max(shapes), np.mean(shapes)) # 91 252 111.98793787177793; arc:94 252 130.2675799086758; openbook:91 177 107.94996973976195