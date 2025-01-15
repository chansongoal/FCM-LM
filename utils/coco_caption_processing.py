import json

# Step 1: Read the JSON file
def read_annotations(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Step 2: Process the first 100 images and find the longest caption
def process_annotations(data, prefix, samples=100):
    image_captions = {}

    # Organize captions by image_id
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        # caption = annotation['caption'].strip()
        caption = annotation['caption'].replace('\n', '').strip()  # Remove all newlines and strip whitespace

        if caption.strip():  # Ensure the caption is not empty
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(caption)

    # Get the longest caption for the specified number of images
    processed_data = []
    for image in data['images']:
        image_id = image['id']
        file_name = prefix + image['file_name']

        if image_id in image_captions and image_captions[image_id]:
            longest_caption = max(image_captions[image_id], key=len)
            processed_data.append((file_name, longest_caption))

    return processed_data[:samples]

# Step 3: Write the results to a file
def write_to_file(processed_data, output_path):
    with open(output_path, 'w') as f:
        for file_name, caption in processed_data:
            if file_name and caption:  # Check for non-empty entries
                f.write(f"{file_name} {caption}\n")
                # f.write(f"{file_name[:-4]}\n")

# Step 4: Read the output file and return file names and captions
def get_captions(source_captions_name):
    image_names = []
    captions = []

    with open(source_captions_name, 'r') as f:
        for line in f:
            image_name, caption = line.split(" ", 1)
            image_names.append(image_name)
            captions.append(caption.strip())

    return captions, image_names

# Main workflow
def main(annotations_path, output_file, prefix, samples):
    data = read_annotations(annotations_path)
    processed_data = process_annotations(data, prefix, samples)
    write_to_file(processed_data, output_file)
    # captions, image_names = get_captions(output_file)

if __name__ == "__main__":
    samples = 100
    prefix = 'COCO_val2017_'
    annotations_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017.json"
    # output_file = f"/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select{samples}.txt"
    output_file = f"/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_select{samples}_vtm2.txt"
    # output_file = f"/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_val2017_all.txt"
    main(annotations_path, output_file, prefix, samples)

    # samples = 10000
    # prefix = 'COCO_train2017_'
    # annotations_path = "/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_train2017.json"
    # # output_file = f"/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_train2017_select{samples}.txt"
    # output_file = f"/home/gaocs/projects/FCM-LM/Data/sd3/tti/source/captions_train2017_all.txt"
    # main(annotations_path, output_file, prefix, samples)