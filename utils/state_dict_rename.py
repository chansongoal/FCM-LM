import torch


# Load the model file
model_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/hyperprior/training_models/trunl[-1,-2,-10,-10]_trunh[1,2,10,10]_uniform0_bitdepth1/hyperprior_lambda0.12_epoch200_lr1e-4_bs128_patch256-256_checkpoint.pth.tar'
new_model_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/dpt/hyperprior/training_models/trunl[-1,-2,-10,-10]_trunh[1,2,10,10]_uniform0_bitdepth1/hyperprior_lambda0.12_epoch200_lr1e-4_bs128_patch256-256_checkpoint.pth.tar'
model_data = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the state_dict
state_dict = model_data['state_dict']

# Remove the 'module.' prefix from state_dict keys
new_state_dict = {}
for key in state_dict.keys():
    if key.startswith("module."):
        new_key = key[len("module."):]  # Remove 'module.' prefix
    else:
        new_key = key
    new_state_dict[new_key] = state_dict[key]

# Replace the original state_dict with the updated one
model_data['state_dict'] = new_state_dict

# # Print original and modified keys (optional)
# print("Original Keys:")
# print(list(state_dict.keys())[:10])  # Print the first 10 original keys
# print("\nModified Keys:")
# print(list(new_state_dict.keys())[:10])  # Print the first 10 modified keys

# Save the updated model
torch.save(model_data, new_model_path)

print(f"New model saved as '{new_model_path}'")
