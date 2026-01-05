"""Main entry point - runs selected configuration with crash recovery."""

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import BirdDataset
from metrics import evaluate
from utils import (load_data, get_sample_images, get_transforms,
                   build_model, get_optimizer, train_epoch)
from visualisation import (plot_roc_curves, plot_confusion_matrices,
                           predict_samples, create_sample_predictions)

# ============== SELECTED CONFIGURATION ==============
MODEL_NAME = 'efficientnet_b3'
OPTIMIZER_NAME = 'AdamW'
FREEZE_STRATEGY = 'all'
LEARNING_RATE = 0.0005
AUGMENTATION = 'heavy'
IMAGE_SIZE = 640
NUM_EPOCHS = 50
BATCH_SIZE = 16
DROPOUT = 0.2
WEIGHT_DECAY = 0.01
# ====================================================

CHECKPOINT_PATH = config.MODEL_SAVE_DIR / f"checkpoint_{MODEL_NAME}_selected.pth"
BEST_MODEL_PATH = config.MODEL_SAVE_DIR / f"best_{MODEL_NAME}_selected.pth"


def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_epoch):
    """Save training checkpoint."""
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_f1': best_f1,
        'best_epoch': best_epoch,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler):
    """Load checkpoint if exists. Returns (start_epoch, best_f1, best_epoch)."""
    if CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['best_f1'], checkpoint['best_epoch']
    return 0, 0.0, 0


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print(f"{'='*60}")
    print(f"TRAINING: {MODEL_NAME} | {OPTIMIZER_NAME} | {FREEZE_STRATEGY}")
    print(f"Image: {IMAGE_SIZE}px | Epochs: {NUM_EPOCHS} | Aug: {AUGMENTATION}")
    print(f"config.DEVICE: {config.DEVICE}")
    print(f"{'='*60}")

    clear_memory()

    train_df, val_df, test_df, class_weight = load_data()

    train_ds = BirdDataset(train_df, get_transforms(IMAGE_SIZE, AUGMENTATION))
    val_ds = BirdDataset(val_df, get_transforms(IMAGE_SIZE))
    test_ds = BirdDataset(test_df, get_transforms(IMAGE_SIZE))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(MODEL_NAME, FREEZE_STRATEGY, DROPOUT).to(config.DEVICE)
    optimizer = get_optimizer(model, OPTIMIZER_NAME, LEARNING_RATE, WEIGHT_DECAY)
    criterion = nn.BCELoss(reduction='none')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Load checkpoint if exists
    start_epoch, best_f1, best_epoch = load_checkpoint(model, optimizer, scheduler)
    if start_epoch > 0:
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f} at epoch {best_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, class_weight, config.DEVICE)
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_acc:.4f}/{val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1, best_epoch = val_metrics['f1'], epoch + 1
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_epoch)

        # Clear memory periodically
        if (epoch + 1) % 5 == 0:
            clear_memory()

    # Training complete - remove checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Training complete, checkpoint removed.")

    # Load best and evaluate
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    test_metrics = evaluate(model, test_loader, config.DEVICE)

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")

    # Save results and plots
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = [{
        'model': f"{MODEL_NAME}_{OPTIMIZER_NAME}_{FREEZE_STRATEGY}",
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_specificity': test_metrics['specificity'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
    }]
    pd.DataFrame(results).to_csv(config.RESULTS_DIR / 'selected_model_results.csv', index=False)

    roc_data = {MODEL_NAME: (test_metrics['labels'], test_metrics['probs'])}
    cm_data = {MODEL_NAME: np.array([[test_metrics['tn'], test_metrics['fp']], [test_metrics['fn'], test_metrics['tp']]])}

    plot_roc_curves(roc_data, config.RESULTS_DIR)
    plot_confusion_matrices(cm_data, config.RESULTS_DIR)

    # Sample predictions
    sample_images = get_sample_images()
    if sample_images:
        preds = predict_samples({MODEL_NAME: model}, sample_images, config.DEVICE, IMAGE_SIZE)
        create_sample_predictions(sample_images, preds, config.RESULTS_DIR)
        print(f"\nSample predictions saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()