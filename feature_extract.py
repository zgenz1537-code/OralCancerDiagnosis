import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms

INPUT_DIR = "Data/preprocessed"
OUTPUT_PATH = "Data/feature/features.npy"
IMG_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = sorted(os.listdir(INPUT_DIR))
label_map = {cls: idx for idx, cls in enumerate(classes)}

print("Label mapping:", label_map)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

googlenet = models.googlenet(pretrained=True)
resnet = models.resnet50(pretrained=True)
vgg = models.vgg16(pretrained=True)

googlenet.fc = nn.Identity()
resnet.fc = nn.Identity()
vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])  # remove last layer

googlenet = googlenet.to(device).eval()
resnet = resnet.to(device).eval()
vgg = vgg.to(device).eval()

for model in [googlenet, resnet, vgg]:
    for param in model.parameters():
        param.requires_grad = False

image_paths = []
labels = []

for cls in classes:
    cls_path = os.path.join(INPUT_DIR, cls)
    for file in os.listdir(cls_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(cls_path, file))
            labels.append(label_map[cls])

print(f"Total images: {len(image_paths)}")

features_list = []
labels_list = []

def process_batch(batch_paths):
    images = []

    for path in batch_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)

    images = torch.stack(images).to(device)

    with torch.no_grad():
        f1 = googlenet(images)
        f2 = resnet(images)
        f3 = vgg(images)

        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)

        fused = torch.cat((f1, f2, f3), dim=1)

    return fused.cpu().numpy()

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i + BATCH_SIZE]
    batch_labels = labels[i:i + BATCH_SIZE]

    batch_features = process_batch(batch_paths)

    features_list.append(batch_features)
    labels_list.extend(batch_labels)

features_array = np.vstack(features_list)
labels_array = np.array(labels_list)

np.save(OUTPUT_PATH, {
    "features": features_array,
    "labels": labels_array
})

print("Feature extraction complete.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Feature shape: {features_array.shape}")