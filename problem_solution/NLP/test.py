import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Generate synthetic data
data = {
    'text': ["This movie is fantastic!", "I hated this movie.", "It was an average film."] * 100,
    'label': [1, 0, 0] * 100
}
df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Dataset class
class TextDataset(Dataset):
    def __init__(self, dataframe, vocab=None, max_length=100):
        self.data = dataframe
        self.max_length = max_length
        self.vocab = vocab or self.build_vocab()
    
    def build_vocab(self):
        vocab = set()
        for text in self.data['text']:
            tokens = preprocess(text)
            vocab.update(tokens)
        return {word: idx+1 for idx, word in enumerate(vocab)}  # Reserve index 0 for padding
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        tokens = preprocess(text)
        indices = [self.vocab.get(token, 0) for token in tokens]  # Convert tokens to indices
        padded_indices = indices[:self.max_length] + [0] * (self.max_length - len(indices))  # Padding
        return torch.tensor(padded_indices), torch.tensor(label, dtype=torch.float32)

# Load data
dataframe = pd.read_csv('sample_data.csv')
dataset = TextDataset(dataframe)

# Split data into training and validation sets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

# Model definition
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embedding(text)  # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

# Parameters
VOCAB_SIZE = len(dataset.vocab) + 1  # Add 1 for padding index
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = 0  # Index for padding

# Initialize the model
model = CNN(VOCAB_SIZE, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

# Training settings
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# Training loop
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# Train the model
N_EPOCHS = 500

train_losses = []
valid_losses = []

fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode

def update_plot(epoch):
    ax.clear()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(valid_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title(f'Epoch {epoch+1}/{N_EPOCHS}')
    plt.draw()
    plt.pause(0.1)

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc*100:.2f}%, Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc*100:.2f}%')
    
    update_plot(epoch)

# Save the model
torch.save(model.state_dict(), 'cnn_model.pth')

# Save the plot
plt.ioff()
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss_plot.png')
plt.show()
