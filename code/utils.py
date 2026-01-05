"""Shared utilities for training and evaluation."""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import config
from model import get_model

def load_data():
    """Load and split dataset, excluding sample images."""
    df = pd.read_csv(config.CSV_FILENAME)
    sample_ids = config.SAMPLE_FLY_IDS + config.SAMPLE_NOFLY_IDS
    df = df[~df['id'].isin(sample_ids)]

    df['filepath'] = df.apply(
        lambda r: config.FLY_FOLDER / r['filename'] if r['folder'] == 'fly' else config.NOFLY_FOLDER / r['filename'],
        axis=1
    )
    df = df[df['filepath'].apply(lambda p: p.exists())]

    n_fly, n_nofly = (df['label'] == 1).sum(), (df['label'] == 0).sum()
    print(f"Dataset: {len(df)} samples (Fly: {n_fly}, NoFly: {n_nofly})")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_end = int(0.8 * len(df))
    val_end = train_end + int(0.1 * len(df))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:], n_nofly / n_fly


def get_sample_images():
    """Get sample images from test folder."""
    samples = []
    for img_path in config.TEST_FOLDER.glob('*.jpg'):
        img_id = int(img_path.stem)
        label = 1 if img_id in config.SAMPLE_FLY_IDS else 0
        samples.append({'id': img_id, 'filepath': img_path, 'label': label})
    return samples


def get_transforms(image_size, augmentation='none'):
    """Get transforms for training or validation."""
    if augmentation == 'heavy':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    elif augmentation == 'medium':
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # none - validation/test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def build_model(model_name, freeze_strategy='all', dropout=0.2):
    """Build model with specified freeze strategy and dropout."""
    model, unfreeze_layer = get_model(model_name, pretrained=True)

    # Replace classifier with custom dropout
    if model_name == 'efficientnet_b3':
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1), nn.Sigmoid())
    elif model_name == 'densenet121':
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1), nn.Sigmoid())
    elif model_name in ['resnet50', 'resnext50']:
        in_features = model.fc[1].in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1), nn.Sigmoid())

    # Freeze strategy
    if freeze_strategy == 'freeze':
        for param in model.parameters():
            param.requires_grad = False
        classifier = model.classifier if hasattr(model, 'classifier') else model.fc
        for param in classifier.parameters():
            param.requires_grad = True
    elif freeze_strategy == 'last_block':
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if unfreeze_layer in name or 'fc' in name or 'classifier' in name:
                param.requires_grad = True
    # else 'all' - all params trainable

    return model


def get_optimizer(model, opt_name, lr, weight_decay=0.01):
    """Get optimizer by name."""
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt_name == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'Adam':
        return optim.Adam(params, lr=lr)
    elif opt_name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:  # RMSprop
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)


def train_epoch(model, loader, criterion, optimizer, class_weight, device):
    """Train for one epoch."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        if class_weight:
            weights = torch.where(labels == 1, class_weight, 1.0)
            loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += ((outputs > 0.5).float() == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total