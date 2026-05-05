import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = "Data/feature/features.npy"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
GAMMA = 0.99
TARGET_UPDATE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = np.load(DATA_PATH, allow_pickle=True).item()
X = data["features"]
y = data["labels"]

num_classes = len(np.unique(y))
input_dim = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

class DQN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

online_net = DQN(input_dim, num_classes).to(device)
target_net = DQN(input_dim, num_classes).to(device)

target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=LR)
criterion = nn.MSELoss()

history = []
best_acc = 0.0

def get_batches(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

for epoch in range(EPOCHS):
    online_net.train()

    train_losses = []
    train_preds = []
    train_true = []

    # ---------------- TRAIN ----------------
    for xb, yb in get_batches(X_train, y_train, BATCH_SIZE):

        q_values = online_net(xb)

        q_selected = q_values.gather(1, yb.unsqueeze(1)).squeeze()

        next_states = xb.roll(-1, dims=0)

        with torch.no_grad():
            next_q = target_net(next_states)
            max_next_q = torch.max(next_q, dim=1)[0]

        rewards = torch.ones_like(q_selected)

        target = rewards + GAMMA * max_next_q

        loss = criterion(q_selected, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        preds = torch.argmax(q_values, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(yb.cpu().numpy())

    train_acc = accuracy_score(train_true, train_preds)
    train_loss = np.mean(train_losses)

    # ---------------- TEST ----------------
    online_net.eval()
    with torch.no_grad():
        q_values = online_net(X_test)
        preds = torch.argmax(q_values, dim=1)

        test_acc = accuracy_score(y_test.cpu().numpy(), preds.cpu().numpy())
        test_loss = 1 - test_acc 

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(online_net.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

    history.append({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    })

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    if epoch % TARGET_UPDATE == 0:
        target_net.load_state_dict(online_net.state_dict())

df = pd.DataFrame(history)
df.to_csv(os.path.join(MODEL_DIR, "training_log.csv"), index=False)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(df["train_loss"], label="Train Loss")
plt.plot(df["test_loss"], label="Test Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(df["train_acc"], label="Train Acc")
plt.plot(df["test_acc"], label="Test Acc")
plt.title("Accuracy Curve")
plt.legend()

plt.savefig(os.path.join(MODEL_DIR, "training_plot.png"))
plt.close()

print("✅ Final DQN training complete (paper-aligned)")
