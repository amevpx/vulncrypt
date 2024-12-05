import json
from collections import Counter

def build_vocabulary(tokenized_data_path, vocab_output_path):
    """
    Builds a vocabulary from tokenized data.
    """
    # Load the tokenized data
    with open(tokenized_data_path, 'r') as f:
        tokenized_data = json.load(f)

    # Count all tokens
    all_tokens = []
    for file_data in tokenized_data.values():
        all_tokens.extend(file_data['tokens'])

    # Build vocabulary with token frequencies
    token_counts = Counter(all_tokens)
    vocab = {token: idx for idx, (token, _) in enumerate(token_counts.items(), start=1)}  # Start IDs at 1
    vocab['<PAD>'] = 0  # Add padding token at index 0

    # Save the vocabulary
    with open(vocab_output_path, 'w') as f:
        json.dump(vocab, f, indent=4)

    print(f"Vocabulary saved to {vocab_output_path}. Total tokens: {len(vocab)}")

if __name__ == "__main__":
    tokenized_data_path = "venv/data/processed/tokenized_data.json"
    vocab_output_path = "venv/data/processed/vocab.json"

    build_vocabulary(tokenized_data_path, vocab_output_path)
