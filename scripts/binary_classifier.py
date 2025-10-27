import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer
import statistics

class BinaryClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        pooled = x.sum(dim=1) / denom          
        pooled = self.relu(pooled)
        logits = self.classifier(pooled)
        return logits.squeeze(-1)


# -----------------------------
# Preprocessing -> tensors
# -----------------------------
class Preprocessing:
    def __init__(self, datapath: str, pretrained_tokenizer: str = 'zhihan1996/DNABERT-2-117M'):
        # Also tried to use gpt-2 getting very similar results, may need a deeper net
        self.data = pd.read_csv(datapath)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

    def to_tensors(self):
        df = pd.DataFrame(self.data)
        texts = df['sequence'].astype(str).tolist()
        labels = torch.tensor(df['label'].values, dtype=torch.float32)
        x = df['label'].tolist()
        print(statistics.mean(x))

        tok_out = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tok_out['input_ids'] 
        attention_mask = tok_out['attention_mask']
        return input_ids, attention_mask, labels


# -----------------------------
# DataLoaders
# -----------------------------

def make_dataloaders(input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: torch.Tensor,
                     batch_size: int = 32,
                     test_size: float = 0.2,
                     seed: int = 42):
    # Train/val split on indices to keep tensors aligned
    n = labels.shape[0]
    idx = torch.arange(n)
    train_idx, val_idx = train_test_split(idx.numpy(), test_size=test_size, random_state=seed, shuffle=True)
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    train_ds = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
    val_ds = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -----------------------------
# Train + Eval
# -----------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    # print(f"running on {device}")
    for input_ids, attention_mask, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return total_loss / len(dataloader.dataset), correct / max(total, 1)


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    prep = Preprocessing('./data/data_sequences.csv')
    input_ids, attention_mask, labels = prep.to_tensors()

    train_loader, val_loader = make_dataloaders(input_ids, attention_mask, labels, batch_size=32)

    vocab_size = prep.tokenizer.vocab_size
    model = BinaryClassifier(vocab_size=vocab_size, embedding_dim=128).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Train
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f}")

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # Gets around 64% accuracy at the moment with the DNABERT Tokenizer
    print(val_acc)

