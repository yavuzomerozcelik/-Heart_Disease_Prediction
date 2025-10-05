# Heart Disease Prediction with PyTorch
# Dataset: UCI Cleveland Heart Disease


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
data = pd.read_csv(url, header=None)
data.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','DISEASE']

# Replace '?' with NaN and drop those rows
data = data.replace('?', np.nan).dropna()

# Convert all columns to numeric
data = data.apply(pd.to_numeric)

# Recode DISEASE: 0 = absence, 1 = presence
data['DISEASE'] = (data['DISEASE'] > 0).astype(int)

# Z-score normalization for non-categorical columns
num_cols = ['age','cp','trestbps','chol','restecg','thalach','oldpeak','slope','ca','thal']
for col in num_cols:
    data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)

# -------------------------------
# 2. Split dataset
# -------------------------------
X = torch.tensor(data.drop('DISEASE', axis=1).values).float()
y = torch.tensor(data['DISEASE'].values).float()[:,None]  # make it 2D

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=len(y_test))

# -------------------------------
# 3. Define enhanced neural network
# -------------------------------
class HeartNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(13, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16,1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc_out(x)

# -------------------------------
# 4. Initialize model, loss, optimizer
# -------------------------------
model = HeartNetV2()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
loss_fn = nn.BCEWithLogitsLoss()
num_epochs = 200

# -------------------------------
# 5. Train the model
# -------------------------------
train_loss = []
test_loss = []
train_acc = []
test_acc = []

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    correct_train = 0
    total_train = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

        preds = (torch.sigmoid(y_pred) > 0.5).float()
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)

    train_loss.append(np.mean(batch_losses))
    train_acc.append(100 * correct_train / total_train)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader:
            y_test_pred = model(X_test_batch)
            loss = loss_fn(y_test_pred, y_test_batch)
            test_loss.append(loss.item())
            preds_test = (torch.sigmoid(y_test_pred) > 0.5).float()
            test_acc.append(100 * (preds_test == y_test_batch).sum().item() / y_test_batch.size(0))

# -------------------------------
# 6. Plot loss and accuracy
# -------------------------------
fig, ax = plt.subplots(1,2, figsize=(16,5))

ax[0].plot(train_loss, label='Train Loss')
ax[0].plot(test_loss, label='Test Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model Loss')
ax[0].legend()

ax[1].plot(train_acc, label='Train Accuracy')
ax[1].plot(test_acc, label='Test Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final Test Accuracy: {test_acc[-1]:.2f}%')
ax[1].legend()

plt.show()
