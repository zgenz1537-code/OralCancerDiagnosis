import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


DATA_PATH = "Data/feature/features.npy"
MODEL_PATH = "models/best_model.pth"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

SPLITS = [0.3, 0.2]  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(DATA_PATH, allow_pickle=True).item()
X = data["features"]
y = data["labels"]

num_classes = len(np.unique(y))
input_dim = X.shape[1]

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

model = DQN(input_dim, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def evaluate_split(test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    def predict(X_data):
        X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        return preds, probs

    train_preds, train_probs = predict(X_train)
    test_preds, test_probs = predict(X_test)

    for name, y_true, y_pred in [
        ("train", y_train, train_preds),
        ("test", y_test, test_preds)
    ]:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix ({name}) {int((1-test_size)*100)}:{int(test_size*100)}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(SAVE_DIR, f"cm_{name}_{int(test_size*100)}.png"))
        plt.close()

    report = classification_report(y_test, test_preds, output_dict=True)
    df = pd.DataFrame(report).transpose()

    csv_name = f"metrics_{int(test_size*100)}.csv"
    df.to_csv(os.path.join(SAVE_DIR, csv_name))

    print(f"\n===== Split {int((1-test_size)*100)}:{int(test_size*100)} =====")
    print(classification_report(y_test, test_preds))

    y_bin = label_binarize(y_test, classes=np.arange(num_classes))

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], linestyle="--")
    plt.title(f"ROC Curve {int((1-test_size)*100)}:{int(test_size*100)}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"roc_{int(test_size*100)}.png"))
    plt.close()

    plt.figure()
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], test_probs[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.title(f"PR Curve {int((1-test_size)*100)}:{int(test_size*100)}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"pr_{int(test_size*100)}.png"))
    plt.close()

for split in SPLITS:
    evaluate_split(split)

print("\n✅ Paper-style evaluation complete.")