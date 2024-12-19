#!/usr/bin/env python3
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

# We will import classes and functions from train.py for convenience.
# Ensure that train.py is in the same directory or that it's in the Python path.
from train import LSTMModel, VulnDataset, collate_fn, PAD_TOKEN, UNK_TOKEN, MODEL_PATH

TEST_DATA = "venv/data/processed/splits/test_data.json"
VOCAB_FILE = "venv/data/vocab.json"

BATCH_SIZE = 32

def main():
    # Load vocab
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    word2id = {w: i for i, w in enumerate(vocab)}
    pad_id = word2id[PAD_TOKEN]
    unk_id = word2id[UNK_TOKEN]

    # Create test dataset and loader
    test_dataset = VulnDataset(TEST_DATA, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, word2id, pad_id, unk_id)
    )

    # Load model
    model = LSTMModel(len(vocab), 128, 256, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item()*x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_loss /= total
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
