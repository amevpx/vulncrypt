#!/usr/bin/env python3
import os
import sys
import json
import torch
import re
from train import LSTMModel, PAD_TOKEN, UNK_TOKEN

MODEL_PATH = "vulncrypt_model.pth"
VOCAB_FILE = "venv/data/vocab.json"

# Regex patterns for function and labels
FUNC_PATTERN = re.compile(
    r'(?:void|int|char|wchar_t|long|short|float|double)\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{([^}]*)\}',
    re.MULTILINE | re.DOTALL
)
TOKEN_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+|\S')

def tokenize_code(code_str):
    return TOKEN_PATTERN.findall(code_str)

def extract_functions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    matches = FUNC_PATTERN.findall(content)
    functions = []
    for m in matches:
        func_name = m[0]
        func_body = m[1]

        # Extract label from function name
        label = 'good' if 'good' in func_name.lower() else 'bad' if 'bad' in func_name.lower() else 'unknown'

        # Check for label in preceding comment
        comment_pattern = re.compile(r'//\s*Label:\s*(good|bad)', re.IGNORECASE)
        comment_match = comment_pattern.search(content, 0, content.find(func_body))
        if comment_match:
            label = comment_match.group(1).lower()

        tokens = tokenize_code(func_body)
        functions.append({'function_name': func_name, 'tokens': tokens, 'label': label})
    return functions

def load_model_and_vocab():
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    word2id = {w: i for i, w in enumerate(vocab)}
    pad_id = word2id[PAD_TOKEN]
    unk_id = word2id[UNK_TOKEN]

    model = LSTMModel(len(vocab), 128, 256, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, vocab, word2id, pad_id, unk_id, device

def predict_functions(model, vocab, word2id, pad_id, unk_id, device, functions, max_len=200):
    sequences = []
    labels = []
    for func in functions:
        tokens = func['tokens']
        token_ids = [word2id.get(t, unk_id) for t in tokens[:max_len]]
        if len(token_ids) < max_len:
            token_ids += [pad_id] * (max_len - len(token_ids))
        sequences.append(token_ids)
        labels.append(func['label'])

    if not sequences:
        return []

    x = torch.tensor(sequences, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().tolist()

    results = []
    correct = 0
    total = 0
    for f, p, actual in zip(functions, preds, labels):
        label = 'bad' if p == 1 else 'good'
        is_correct = (label == actual)
        if actual != 'unknown':
            total += 1
            if is_correct:
                correct += 1
        results.append({'function_name': f['function_name'], 'predicted_label': label, 'actual_label': actual})
    accuracy = (correct / total) * 100 if total > 0 else 0
    return results, accuracy, correct, total

def main():
    if len(sys.argv) < 2:
        print("Usage: predict.py <path_to_new_c_files...>")
        sys.exit(1)

    model, vocab, word2id, pad_id, unk_id, device = load_model_and_vocab()

    for file_path in sys.argv[1:]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing file: {file_path}")
        functions = extract_functions_from_file(file_path)
        if not functions:
            print(f"No functions found in {file_path}.")
            continue

        predictions, accuracy, correct, total = predict_functions(
            model, vocab, word2id, pad_id, unk_id, device, functions
        )
        print(f"Predictions for {file_path}:")
        for p in predictions:
            print(f"  Function: {p['function_name']} -> Predicted: {p['predicted_label']} (Actual: {p['actual_label']})")
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
