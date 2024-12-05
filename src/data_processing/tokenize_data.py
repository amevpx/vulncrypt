import os
import re
import json

def tokenize_code(code):
    """
    Tokenizes C code into meaningful tokens using a simple regex-based tokenizer.
    """
    token_specification = [
        ('COMMENT', r'//.*|/\*[\s\S]*?\*/'),  # Single-line or multi-line comments
        ('STRING', r'"(?:\\.|[^"\\])*"'),     # String literals
        ('NUMBER', r'\b\d+\b'),               # Numbers
        ('IDENTIFIER', r'\b[A-Za-z_][A-Za-z0-9_]*\b'),  # Identifiers
        ('OPERATOR', r'[+\-*/%=&|!<>^~]+'),   # Operators
        ('DELIMITER', r'[(),;{}[\]]'),        # Delimiters
        ('WHITESPACE', r'\s+'),               # Whitespace
        ('MISMATCH', r'.'),                   # Any other character
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
    get_token = re.compile(tok_regex).finditer
    tokens = []
    for match in get_token(code):
        kind = match.lastgroup
        value = match.group()
        if kind not in {'WHITESPACE', 'COMMENT'}:  # Skip whitespace and comments
            tokens.append(value)
    return tokens

def preprocess_file(file_path):
    """
    Reads and tokenizes a C code file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        return tokenize_code(code)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def preprocess_data(file_list_path, output_dir):
    """
    Processes the dataset, tokenizes code, and saves the results to JSON files.
    """
    # Load the file list
    with open(file_list_path, 'r', encoding='utf-8') as f:
        file_list = json.load(f)

    processed_data = []

    # Process each file in the list
    for file_name, file_info in file_list.items():
        file_path = file_info["file_path"]
        label = 1 if 'bad' in file_info["metadata"]["functions"] and file_info["metadata"]["functions"]["bad"] else 0
        print(f"Processing {file_name} (Label: {label})")
        tokens = preprocess_file(file_path)
        processed_data.append({"tokens": tokens, "label": label})

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'processed_data.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Input file list and output directory
    file_list_path = 'venv/data/processed/processed_functions.json'
    output_dir = 'venv/data/processed'

    # Preprocess the data
    preprocess_data(file_list_path, output_dir)
