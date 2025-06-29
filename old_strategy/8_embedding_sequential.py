import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

input_dir = 'results/subsets/'
output_dir = 'results/embeddings/'
os.makedirs(output_dir, exist_ok=True)

changed_file = os.path.join(input_dir, 'changed_variants.parquet')
if not os.path.exists(changed_file):
    print(f"File not found: {changed_file}")
    exit(1)

df = pd.read_parquet(changed_file)

bundled_cols = [col for col in df.columns if col.startswith('Bundled_ClinicalSignificance_')]
years = sorted([int(col.split('_')[-1]) for col in bundled_cols])

def validate_labels(df, bundled_cols):
    expected_labels = {'Pathogenic', 'Benign', 'VUS', 'Other', 'Unknown'}
    
    all_labels = set()
    for col in bundled_cols:
        if col in df.columns:
            unique_labels = set(df[col].dropna().unique())
            all_labels.update(unique_labels)
    unexpected_labels = all_labels - expected_labels
    
    if unexpected_labels:
        for col in bundled_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 'Other' if x in unexpected_labels else x)
    missing_labels = expected_labels - all_labels
    if missing_labels:
        print(f"Note: Missing labels: {missing_labels}")
    
    return df

df = validate_labels(df, bundled_cols)

sequence_data = []
for _, row in df.iterrows():
    sequence = []
    for year in years:
        col = f'Bundled_ClinicalSignificance_{year}'
        if col in df.columns:
            label = row[col]
            if pd.notna(label):
                sequence.append(label)
            else:
                sequence.append('Unknown')
    
    if len(sequence) == len(years):
        sequence_data.append({
            'VariantID': row['VariantID'],
            'sequence': sequence,
            'start_label': sequence[0],
            'end_label': sequence[-1]
        })
        
label_encoder = LabelEncoder()
all_labels = ['Pathogenic', 'Benign', 'VUS', 'Other', 'Unknown']
label_encoder.fit(all_labels)

print(f"Label encoding: {dict(zip(all_labels, label_encoder.transform(all_labels)))}")

class VariantSequenceDataset(Dataset):
    def __init__(self, sequences, label_encoder):
        self.sequences = sequences
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]['sequence']
        encoded_sequence = self.label_encoder.transform(sequence)
        
        
        
        # Input: Sequence WITHOUT final label
        input_sequence = torch.tensor(encoded_sequence[:-1], dtype=torch.long)
        
        # Target: ONLY final label
        target_label = torch.tensor(encoded_sequence[-1], dtype=torch.long)
        
        
        
        return {
            'sequence': input_sequence,
            'end_label': target_label, # This name is now a bit misleading, it's the "target"
            'variant_id': self.sequences[idx]['VariantID']
        }
class VariantLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(VariantLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        last_hidden = hidden[-1]
        dropped = self.dropout(last_hidden)
        output = self.classifier(dropped)
        
        return output, lstm_out

vocab_size = len(all_labels)
embedding_dim = 64
hidden_dim = 128
num_layers = 2
num_classes = len(all_labels)

model = VariantLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)
model = model.to(device)

print(f"Model architecture:")
print(f"  Vocab size: {vocab_size}")
print(f"  Embedding dim: {embedding_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Num layers: {num_layers}")
print(f"  Num classes: {num_classes}")

train_sequences, val_sequences = train_test_split(sequence_data, test_size=0.2, random_state=42)

train_dataset = VariantSequenceDataset(train_sequences, label_encoder)
val_dataset = VariantSequenceDataset(val_sequences, label_encoder)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        sequences = batch['sequence'].to(device)
        end_labels = batch['end_label'].to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(sequences)
        loss = criterion(outputs, end_labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += end_labels.size(0)
        correct += (predicted == end_labels).sum().item()
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            sequences = batch['sequence'].to(device)
            end_labels = batch['end_label'].to(device)
            
            outputs, _ = model(sequences)
            loss = criterion(outputs, end_labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += end_labels.size(0)
            correct += (predicted == end_labels).sum().item()
    
    return total_loss / len(val_loader), correct / total

num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print(f"\nStarting training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

model_path = os.path.join(output_dir, 'variant_lstm_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder,
    'model_config': {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': num_classes
    }
}, model_path)
print(f"Saved model to: {model_path}")

def extract_embeddings(model, dataset, device):
    model.eval()
    embeddings = []
    variant_ids = []
    labels = []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            sequences = batch['sequence'].to(device)
            end_labels = batch['end_label']
            variant_id_batch = batch['variant_id']
            
            _, lstm_out = model(sequences)
            
            last_hidden = lstm_out[:, -1, :].cpu().numpy()
            
            embeddings.extend(last_hidden)
            variant_ids.extend(variant_id_batch)
            labels.extend(end_labels.numpy())
    
    return np.array(embeddings), variant_ids, labels

train_embeddings, train_variant_ids, train_labels = extract_embeddings(model, train_dataset, device)
val_embeddings, val_variant_ids, val_labels = extract_embeddings(model, val_dataset, device)

embeddings_data = {
    'train': {
        'embeddings': train_embeddings,
        'variant_ids': train_variant_ids,
        'labels': train_labels
    },
    'val': {
        'embeddings': val_embeddings,
        'variant_ids': val_variant_ids,
        'labels': val_labels
    }
}

embeddings_path = os.path.join(output_dir, 'variant_embeddings.npz')
np.savez(embeddings_path, **{
    'train_embeddings': train_embeddings,
    'train_variant_ids': train_variant_ids,
    'train_labels': train_labels,
    'val_embeddings': val_embeddings,
    'val_variant_ids': val_variant_ids,
    'val_labels': val_labels
})
print(f"Saved embeddings to: {embeddings_path}")

training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'num_epochs': num_epochs,
    'final_train_acc': train_accuracies[-1],
    'final_val_acc': val_accuracies[-1]
}

history_path = os.path.join(output_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)

print(f"\n" + "="*50)
print("DONE")
print("="*50)
print(f"Training accuracy: {train_accuracies[-1]:.4f}")
print(f"Validation accuracy: {val_accuracies[-1]:.4f}")
print(f"Embedding dimension: {hidden_dim}")
print(f"Variants processed: {len(sequence_data)}")
