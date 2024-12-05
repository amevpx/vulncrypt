import json
import os

def encode_sequences(data_file, vocab_file, output_file, max_seq_length):
    """
    Encodes tokenized sequences into integer IDs using the vocabulary.

    Args:
        data_file (str): Path to the tokenized data JSON file.
        vocab_file (str): Path to the vocabulary JSON file.
        output_file (str): Path to save the encoded data JSON file.
        max_seq_length (int): Maximum sequence length for padding or truncation.
    """
    # Load the tokenized data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load the vocabulary
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    encoded_data = []

    # Encode each entry
    for entry in data:
        tokens = entry["tokens"]
        label = entry["label"]

        # Convert tokens to integer IDs
        sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

        # Pad or truncate to the maximum sequence length
        if len(sequence) < max_seq_length:
            sequence += [vocab["<PAD>"]] * (max_seq_length - len(sequence))
        else:
            sequence = sequence[:max_seq_length]

        # Add the encoded sequence and label
        encoded_data.append({"sequence": sequence, "label": label})

    # Save the encoded data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(encoded_data, f, indent=4)
    print(f"Encoded data saved to {output_file}")

if __name__ == "__main__":
    # Paths for input and output files
    data_file = "venv/data/processed/processed_data.json"
    vocab_file = "venv/data/processed/vocab.json"
    output_file = "venv/data/processed/encoded_data.json"
    max_seq_length = 512  # Adjust based on your model's input requirements

    # Encode sequences
    encode_sequences(data_file, vocab_file, output_file, max_seq_length)
