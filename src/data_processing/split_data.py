import json
import random
import os

def split_data(data_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data_file (str): Path to the encoded data JSON file.
        output_dir (str): Directory to save the split datasets.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        seed (int): Random seed for reproducibility.
    """
    # Load the encoded data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the data
    random.seed(seed)
    random.shuffle(data)

    # Calculate split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the splits
    with open(os.path.join(output_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(output_dir, 'val_data.json'), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(output_dir, 'test_data.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    # Print stats
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Data splits saved to {output_dir}")

if __name__ == "__main__":
    # Input encoded data file
    data_file = "venv/data/processed/encoded_data.json"

    # Output directory for split datasets
    output_dir = "venv/data/processed/splits"

    # Split the data
    split_data(data_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
