# import kagglehub
import os
import shutil
import pandas as pd

# path = kagglehub.dataset_download("nih-chest-xrays/data")

# Set paths
root_path = "/Users/ycy/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3"
csv_path = os.path.join(root_path, "Data_Entry_2017.csv")
subset_dir = "/Users/ycy/Desktop/821project/input_images"
os.makedirs(subset_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(csv_path)

# Filter labels and sample
labels_of_interest = ["Cardiomegaly", "Effusion", "Infiltration", "No Finding"]
df_filtered = df[df["Finding Labels"].isin(labels_of_interest)]
df_sampled = df_filtered.groupby("Finding Labels").sample(n=200, random_state=42)
df_sampled.to_csv("selected_metadata.csv", index=False)

# Search image folders
image_folders = [f"images_{i:03d}" for i in range(1, 13)]
not_found = []

copied_count = 0
for image_name in df_sampled["Image Index"]:
    found = False
    for folder in image_folders:
        src_img_path = os.path.join(root_path, folder, "images", image_name)
        if os.path.exists(src_img_path):
            dst_img_path = os.path.join(subset_dir, image_name)
            shutil.copy(src_img_path, dst_img_path)
            found = True
            copied_count += 1
            break
    if not found:
        not_found.append(image_name)
