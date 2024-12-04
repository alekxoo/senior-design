import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Split the dataset into training and validation sets.
    
    Args:
        source_dir (str): Path to the source dataset (containing class subfolders).
        train_dir (str): Path where the training data will be saved.
        val_dir (str): Path where the validation data will be saved.
        split_ratio (float): Fraction of data to use for training (0.8 = 80% for training).
    """
    # Create train and val directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Loop through each class folder in the source dataset
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        if os.path.isdir(class_path):
            # Create the same class folder structure inside train and val directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # Get list of all images in the class folder
            all_images = os.listdir(class_path)
            # Shuffle the images for random splitting
            random.shuffle(all_images)

            # Calculate the split index based on the provided split ratio
            split_index = int(len(all_images) * split_ratio)

            # Split the images into training and validation sets
            train_images = all_images[:split_index]
            val_images = all_images[split_index:]

            # Move the images to the respective folders
            for image in train_images:
                src_image = os.path.join(class_path, image)
                dest_image = os.path.join(train_class_dir, image)
                shutil.copy(src_image, dest_image)

            for image in val_images:
                src_image = os.path.join(class_path, image)
                dest_image = os.path.join(val_class_dir, image)
                shutil.copy(src_image, dest_image)

            print(f"Class '{class_name}' split: {len(train_images)} for training, {len(val_images)} for validation.")

if __name__ == "__main__":
    # Define the source dataset path and the directories for train and validation
    source_dataset_dir = '../dataset/vehicle_images_vault'  # e.g., 'data'
    train_dataset_dir = '../dataset/train'     # e.g., 'data/train'
    val_dataset_dir = '../dataset/val'         # e.g., 'data/val'

    # Split the dataset
    split_dataset(source_dataset_dir, train_dataset_dir, val_dataset_dir)
