"""Training loop for bird flight classifier."""

import time
import torch
import torch.nn as nn
import torch.optim as optim

import config
from model import get_model, freeze_backbone
from metrics import evaluate



def train_epoch(model, loader, criterion, optimizer, class_weight):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)

        if class_weight:
            loss = criterion(outputs, labels)
            weights = torch.where(labels == 1, class_weight, 1.0)
            loss = (loss * weights).mean()
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct += ((outputs > 0.5).float() == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


def train_model(name, train_loader, val_loader, class_weight):
    print(f"\n{'='*50}\nTRAINING: {name.upper()}\n{'='*50}")

    model, unfreeze_layer = get_model(name, pretrained=True)
    freeze_backbone(model, unfreeze_layer)
    model = model.to(config.DEVICE)

    criterion = nn.BCELoss(reduction='none') if config.USE_WEIGHTED_LOSS else nn.BCELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_f1, best_epoch = 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                            class_weight if config.USE_WEIGHTED_LOSS else None)
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        print(f"Epoch {epoch+1:2d}/{config.NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_acc:.4f}/{val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1, best_epoch = val_metrics['f1'], epoch + 1
            config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_DIR / f"best_{name}.pth")

    training_time = time.time() - start
    print(f"Best F1: {best_f1:.4f} at epoch {best_epoch} | Time: {training_time:.1f}s")

    return history, training_time, best_epoch