import shutil
import os

def image_copy():
    # Paths to the source and destination folders
    folder_a = '/Users/changshenggao/Library/CloudStorage/OneDrive-NanyangTechnologicalUniversity/ResearchOngoing/minVal2014'  # Update with the path to folder 'a'
    folder_b = '/Users/changshenggao/Downloads/coco_val2017_selected100'  # Update with the path to folder 'b'

    # Path to the captions file
    captions_file = '/Users/changshenggao/Downloads/captions_val2017_select100.txt'

    # Read the image names from the captions file
    with open(captions_file, 'r') as file:
        image_names = [line.split()[0] for line in file.readlines()]

    # Create folder 'b' if it doesn't exist
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    # Copy images from folder 'a' to folder 'b'
    for image_name in image_names:
        new_image_name = image_name.replace('val2017', 'val2014')
        src_image_path = os.path.join(folder_a, new_image_name)
        dst_image_path = os.path.join(folder_b, image_name)
        
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)
            # print(f'Copied: {src_image_path}')
        else:
            print(f'Image not found: {src_image_path}')

def feat_copy():
    # Paths to the source and destination folders
    folder_a = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test_all'  # Update with the path to folder 'a'
    folder_b = '/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test'  # Update with the path to folder 'b'

    # Path to the captions file
    captions_file = fr'/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select100_vtm.txt'

    # Read the image names from the captions file
    with open(captions_file, 'r') as file:
        image_names = [line.split()[0] for line in file.readlines()]

    # Create folder 'b' if it doesn't exist
    if not os.path.exists(folder_b):
        os.makedirs(folder_b)

    # Copy images from folder 'a' to folder 'b'
    for image_name in image_names:
        src_image_path = os.path.join(folder_a, image_name+'.npy')
        dst_image_path = os.path.join(folder_b, image_name+'.npy')
        
        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dst_image_path)
            # print(f'Copied: {src_image_path}')
        else:
            print(f'Image not found: {src_image_path}')

if __name__ == "__main__":
    feat_copy()