# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 01:46:51 2024

@author: edgar
"""

import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# GLOBAL MODEL
# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=60): # Set to 60 max length because of the sequence's length
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Add spaces between amino acid residues
        sequence = " ".join(sequence)
        inputs = self.tokenizer(sequence, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(label, dtype=torch.float)

# Load data
df = pd.read_csv('EColi_MIC.csv') # Change the directory if needed
protein_sequences = df.iloc[:, 1].tolist()  # sequences are in the second column
mic_values = df.iloc[:, 2].tolist()  # MIC values are in the third column

# Normalize MIC values
scaler = StandardScaler()
mic_values = scaler.fit_transform(np.array(mic_values).reshape(-1, 1)).flatten()

# Train-test split
train_sequences, test_sequences, train_labels, test_labels = train_test_split(protein_sequences, mic_values, test_size=0.2, random_state=42)

# Model name
model_name = "Rostlab/prot_bert_bfd"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
bert_model = BertModel.from_pretrained(model_name)

# Dataset and DataLoader
train_dataset = ProteinDataset(train_sequences, train_labels, tokenizer)
test_dataset = ProteinDataset(test_sequences, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Model with a fully connected neural network
class ProBERTMICModel(nn.Module):
    def __init__(self, bert_model):
        super(ProBERTMICModel, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        prediction = self.fc3(x)  # [batch_size, 1]
        return prediction

# Initialize model
model = ProBERTMICModel(bert_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Training function
def train_model(model, train_loader, test_loader, epochs=15, learning_rate=1e-04):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels_list = []
        
        for batch in train_loader:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            train_preds.extend(outputs.cpu().detach().numpy())
            train_labels_list.extend(labels.cpu().detach().numpy())

        # Testing Dataset
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_labels_list = []
        total_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                total_batches += 1
                
                test_preds.extend(outputs.cpu().detach().numpy())
                test_labels_list.extend(labels.cpu().detach().numpy())

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / total_batches if total_batches > 0 else float('inf')

        # Calculate RMSE, MAE, and R2
        train_preds = np.array(train_preds).flatten()
        train_labels_list = np.array(train_labels_list).flatten()
        train_rmse = np.sqrt(mean_squared_error(train_labels_list, train_preds))
        train_mae = mean_absolute_error(train_labels_list, train_preds)
        train_r2 = r2_score(train_labels_list, train_preds)

        test_preds = np.array(test_preds).flatten()
        test_labels_list = np.array(test_labels_list).flatten()
        test_rmse = np.sqrt(mean_squared_error(test_labels_list, test_preds))
        test_mae = mean_absolute_error(test_labels_list, test_preds)
        test_r2 = r2_score(test_labels_list, test_preds)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}")
        print(f"Train RMSE: {train_rmse}, Train MAE: {train_mae}, Train R2: {train_r2}")
        print(f"Test RMSE: {test_rmse}, Test MAE: {test_mae}, Test R2: {test_r2}")

    # Plotting the regression results
    plt.figure(figsize=(10, 5))

    # Plot train predictions vs true values
    plt.subplot(1, 2, 1)
    sns.regplot(x=train_labels_list, y=train_preds, scatter_kws={"s": 5})
    plt.xlabel("True MIC Values (Train)")
    plt.ylabel("Predicted MIC Values")
    plt.title(f"Train R2: {train_r2:.4f}")

    # Plot test predictions vs true values
    plt.subplot(1, 2, 2)
    sns.regplot(x=test_labels_list, y=test_preds, scatter_kws={"s": 5})
    plt.xlabel("True MIC Values (Test)")
    plt.ylabel("Predicted MIC Values")
    plt.title(f"Test R2: {test_r2:.4f}")

    plt.tight_layout()
    plt.show()

    return avg_test_loss, test_rmse, test_mae, test_r2  # Return the metrics after all epochs


# Train the model
train_model(model, train_loader, test_loader)

# GRID SEARCH

# Define lists for the values of the hyperparameters that will be tested
batch_sizes = [4, 8, 16]
learning_rates = [1e-4, 1e-3, 1e-2]
epochs_list = [10, 15, 20]

best_test_loss = float('inf')
best_params = {}
best_metrics = {}

# Iterate over the combinations of hyperparameters
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for epochs in epochs_list:
            print(f"Testing batch size: {batch_size}, learning rate: {learning_rate}, epochs: {epochs}")
            
            # Initialize a new model for each combination of hyperparameters
            model = ProBERTMICModel(bert_model)
            model.to(device)
            
            # Train the model for the current combination of hyperparameters
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            test_loss, test_rmse, test_mae, test_r2 = train_model(model, train_loader, test_loader, learning_rate=learning_rate, epochs=epochs)
            
            print(f"Test Loss Value (MSE): {test_loss}")
            
            # Save the best hyperparameters
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_params = {'Batch Size': batch_size, 'Learning Rate': learning_rate, 'Epochs': epochs}
                best_metrics = {'MSE': test_loss, 'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2}

print("Best Hyperparameters:", best_params)
print("Best Metrics:", best_metrics)