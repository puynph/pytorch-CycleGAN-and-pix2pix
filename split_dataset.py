import os
import shutil
import random

def split_dataset(train_dir, val_dir, split_ratio=0.2, seed=42):
    """Split dataset into training and validation sets.

    Parameters:
        train_dir (str) -- path to the training data directory
        val_dir (str) -- path to the validation data directory
        split_ratio (float) -- fraction of data to use for validation
        seed (int) -- random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Create validation directory
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Get all training data
    all_files = []
    for root, _, filenames in os.walk(train_dir):
        for filename in filenames:
            all_files.append(os.path.join(root, filename))

    random.shuffle(all_files)

    # Determine the number of validation files
    num_files = len(all_files)
    num_val_files = int(num_files * split_ratio)

    for i in range(num_val_files):
        file_path = all_files[i]
        relative_path = os.path.relpath(file_path, train_dir)
        target_path = os.path.join(val_dir, relative_path)
        
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(file_path, target_path)

    print(f"Moved {num_val_files} files from {train_dir} to {val_dir}")

def count_files(directory):
    """Count the number of files in a directory, including files in subdirectories."""
    return sum([len(files) for r, d, files in os.walk(directory)])

# Define directories
trainA_dir = './datasets/fundus/trainA'
valA_dir = './datasets/fundus/valA'
trainB_dir = './datasets/fundus/trainB'
valB_dir = './datasets/fundus/valB'

# Split datasets
# split_dataset(trainA_dir, valA_dir)
# split_dataset(trainB_dir, valB_dir)

trainA_count = count_files(trainA_dir)
valA_count = count_files(valA_dir)
trainB_count = count_files(trainB_dir)
valB_count = count_files(valB_dir)

# Print file counts
print(f"Number of files in trainA: {trainA_count}")
print(f"Number of files in valA: {valA_count}")
print(f"Number of files in trainB: {trainB_count}")
print(f"Number of files in valB: {valB_count}")