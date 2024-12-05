import json
from collections import Counter

def build_vocab(data_file, output_file, min_freq=1):
    """
    Builds a vocabulary from the tokenized data.

    Args:
        data_file (str): Path to the tokenized data JSON file.
        output_file (str): Path to save the vocabulary JSON file.
        min_freq (int): Minimum frequency for a token to be included in the vocabulary.
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Count token frequencies
    token_counter = Counter()
    for entry in data:
        tokens = entry["tokens"]
        token_counter.update(tokens)

    # Build the vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}  # Special tokens: padding and unknown
    for token, freq in token_counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    # Save the vocabulary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)
    print(f"Vocabulary saved to {output_file}")


    print(f"Total tokens: {len(token_counter)}")
    print(f"Vocabulary size: {len(vocab)} (tokens with frequency >= {min_freq})")

if __name__ == "__main__":

    data_file = "venv/data/processed/processed_data.json"
    output_file = "venv/data/processed/vocab.json"


    build_vocab(data_file, output_file, min_freq=1)
