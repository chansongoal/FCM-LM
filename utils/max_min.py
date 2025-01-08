import os
import numpy as np

# Function to compute statistics for .npy files in a directory
def compute_statistics(directory):
    max_values = []
    min_values = []
    mean_values = []

    # Iterate through all .npy files in the directory
    filenames = os.listdir(directory)
    filenames = filenames[:10]
    for file_name in filenames:
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory, file_name)
            data = np.load(file_path)
            print(data.dtype)

            # Compute statistics for the current file
            max_values.append(np.max(data))
            min_values.append(np.min(data))
            mean_values.append(np.mean(data))

    # Overall statistics
    global_max = max(max_values) if max_values else None
    global_min = min(min_values) if min_values else None
    global_mean = np.mean(mean_values) if mean_values else None

    print("Overall statistics:")
    print("Global max:", global_max)
    print("Global min:", global_min)
    print("Global mean:", global_mean)


# Main workflow
if __name__ == "__main__":
    directory = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/feature_test_all"  # Replace with your directory path
    compute_statistics(directory)