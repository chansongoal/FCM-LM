import os
import shutil

def copy_folders_based_on_file(file_path, source_dir, destination_dir):
    """
    Copy specific subfolders and their files from a source directory to a destination directory 
    based on the content of a text file.

    :param file_path: Path to the text file containing folder and file names.
    :param source_dir: Path to the source directory containing all subfolders.
    :param destination_dir: Path to the destination directory to store the copied subfolders.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Read the file and process each line
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            folder_name, file_name = line.split()  # Split folder and file name
            # folder_name = folder_name[2:-1]; file_name = file_name[2:-2]
            print(folder_name, file_name)
            
            source_folder_path = os.path.join(source_dir, folder_name)
            destination_folder_path = os.path.join(destination_dir, folder_name)
            
            # Check if the source folder exists
            if os.path.exists(source_folder_path):
                os.makedirs(destination_folder_path, exist_ok=True)  # Create destination folder
                
                source_file_path = os.path.join(source_folder_path, file_name+'.JPEG')
                destination_file_path = os.path.join(destination_folder_path, file_name+'.JPEG')
                print(source_file_path)
                print(destination_file_path)
                
                # Copy the specific file if it exists
                if os.path.exists(source_file_path):
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"Copied {source_file_path} to {destination_file_path}")
                else:
                    print(f"File {source_file_path} does not exist.")
            else:
                print(f"Folder {source_folder_path} does not exist.")

# Main function to run the script
if __name__ == "__main__":
    text_file_path = "/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/imagenet_selected_pathname100.txt"  # Replace with the actual path to the text file
    source_directory = "/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/ImageNet_Val1000"  # Replace with the actual path to the source directory
    destination_directory = "/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/ImageNet_Selected100"  # Replace with the desired destination directory
    
    copy_folders_based_on_file(text_file_path, source_directory, destination_directory)
