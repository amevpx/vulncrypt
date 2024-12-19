#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

TRAIN_DATA = "venv/data/processed/splits/train_data.json"
VAL_DATA = "venv/data/processed/splits/val_data.json"
VOCAB_FILE = "venv/data/vocab.json"
MODEL_PATH = "vulncrypt_model.pth"

BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 10
LR = 0.001
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class VulnDataset(Dataset):
    def __init__(self, data_path, vocab):
        self.data = self.load_data(data_path)
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.pad_id = self.word2id[PAD_TOKEN]
        self.unk_id = self.word2id[UNK_TOKEN]

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        label = 1 if item['label'] == 'bad' else 0
        # Just return the tokens and label, padding will be done in collate_fn
        return tokens, label

def collate_fn(batch, word2id, pad_id, unk_id, max_len=200):
    # batch is list of (tokens, label)
    labels = []
    sequences = []
    for tokens, label in batch:
        token_ids = [word2id.get(t, unk_id) for t in tokens]
        # Truncate if longer than max_len
        token_ids = token_ids[:max_len]
        # Pad if shorter
        if len(token_ids) < max_len:
            token_ids += [pad_id]*(max_len - len(token_ids))
        sequences.append(token_ids)
        labels.append(label)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, (h, c) = self.lstm(emb)
        # Use the last hidden state
        logits = self.fc(h[-1])
        return logits

def main():
    # Load vocab
    with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    word2id = {w: i for i, w in enumerate(vocab)}
    pad_id = word2id[PAD_TOKEN]
    unk_id = word2id[UNK_TOKEN]

    # Create datasets
    train_dataset = VulnDataset(TRAIN_DATA, vocab)
    val_dataset = VulnDataset(VAL_DATA, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, word2id, pad_id, unk_id))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, word2id, pad_id, unk_id))

    # Model
    model = LSTMModel(len(vocab), EMBED_DIM, HIDDEN_DIM, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss /= total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()*x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_loss /= total
        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model saved.")

if __name__ == "__main__":
    main()
