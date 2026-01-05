"""Grid search for hyperparameter tuning."""

import time
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
from visualisation import (plot_metric_comparison, plot_roc_curves,
                           plot_confusion_matrices, predict_samples, create_sample_predictions)

# Top 10 configurations from ROUND 1 grid search
CONFIGS = [
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.0005, 'optimizer': 'AdamW', 'freeze': 'all'},
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.0005, 'optimizer': 'RMSprop', 'freeze': 'last_block'},
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.001, 'optimizer': 'RMSprop', 'freeze': 'last_block'},
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.0005, 'optimizer': 'AdamW', 'freeze': 'last_block'},
    {'model': 'resnet50', 'augmentation': 'medium', 'lr': 0.001, 'optimizer': 'RMSprop', 'freeze': 'last_block'},
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.001, 'optimizer': 'SGD', 'freeze': 'last_block'},
    {'model': 'resnet50', 'augmentation': 'medium', 'lr': 0.001, 'optimizer': 'Adam', 'freeze': 'last_block'},
    {'model': 'efficientnet_b3', 'augmentation': 'medium', 'lr': 0.0005, 'optimizer': 'RMSprop', 'freeze': 'all'},
    {'model': 'densenet121', 'augmentation': 'medium', 'lr': 0.0005, 'optimizer': 'Adam', 'freeze': 'all'},
    {'model': 'densenet121', 'augmentation': 'medium', 'lr': 0.001, 'optimizer': 'Adam', 'freeze': 'last_block'},
]


def train_config(params, train_df, val_df, class_weight, config_id):
    model = build_model(params['model'], params['freeze'], config.DROPOUT).to(config.DEVICE)
    optimizer = get_optimizer(model, params['optimizer'], params['lr'], config.WEIGHT_DECAY)
    criterion = nn.BCELoss(reduction='none')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    train_ds = BirdDataset(train_df, get_transforms(config.IMAGE_SIZE, params['augmentation']))
    val_ds = BirdDataset(val_df, get_transforms(config.IMAGE_SIZE))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    best_f1, best_epoch = 0.0, 0
    best_state = None

    for epoch in range(config.NUM_EPOCHS):
        train_epoch(model, train_loader, criterion, optimizer, class_weight, config.DEVICE)
        val_metrics = evaluate(model, val_loader, config.DEVICE)
        scheduler.step()

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model_dir = config.MODEL_SAVE_DIR / 'grid_search'
        model_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(config_id, int):
            model_name = f"{config_id:03d}_{params['model']}_{params['optimizer']}_{params['freeze']}.pth"
        else:
            model_name = f"{config_id}_{params['model']}_{params['optimizer']}_{params['freeze']}.pth"
        torch.save(best_state, model_dir / model_name)

    return model, best_f1, best_epoch


def main():
    print(f"{'='*60}\nGRID SEARCH (TOP 10) - {config.DEVICE}\n{'='*60}")

    total = len(CONFIGS)
    print(f"Total configurations: {total}\n")

    train_df, val_df, test_df, class_weight = load_data()

    test_ds = BirdDataset(test_df, get_transforms(config.IMAGE_SIZE))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    results = []
    start_time = time.time()

    for i, params in enumerate(CONFIGS):
        combo_str = f"{params['model'][:8]}|{params['augmentation'][:3]}|lr{params['lr']}|{params['optimizer'][:4]}|{params['freeze'][:4]}"

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"[{i+1}/{total}] {combo_str} | ETA: {eta/3600:.1f}h")

        try:
            model, best_val_f1, best_epoch = train_config(params, train_df, val_df, class_weight, i)
            model.to(config.DEVICE)
            test_metrics = evaluate(model, test_loader, config.DEVICE)

            results.append({
                **params,
                'best_epoch': best_epoch,
                'best_val_f1': best_val_f1,
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_specificity': test_metrics['specificity'],
                'test_f1': test_metrics['f1'],
                'test_auc': test_metrics['auc'],
                'tp': test_metrics['tp'],
                'tn': test_metrics['tn'],
                'fp': test_metrics['fp'],
                'fn': test_metrics['fn'],
            })
            print(f"    -> Val F1: {best_val_f1:.4f} | Test F1: {test_metrics['f1']:.4f}")

        except Exception as e:
            print(f"    -> ERROR: {e}")
            continue

    results_df = pd.DataFrame(results).sort_values('test_f1', ascending=False)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.RESULTS_DIR / 'grid_search_results.csv', index=False)

    print(f"\n{'='*60}\nTOP 10 RESULTS\n{'='*60}")
    print(f"{'Model':<12} {'Aug':<6} {'LR':<7} {'Opt':<7} {'Freeze':<6} {'F1':<8} {'AUC':<8}")
    for _, r in results_df.head(10).iterrows():
        print(f"{r['model'][:12]:<12} {r['augmentation'][:6]:<6} {r['lr']:<7} {r['optimizer'][:7]:<7} "
              f"{r['freeze'][:6]:<6} {r['test_f1']:<8.4f} {r['test_auc']:<8.4f}")

    # Generate plots for top 5
    top5 = results_df.head(5).to_dict('records')
    for r in top5:
        r['model'] = f"{r['model'][:8]}_{r['optimizer'][:4]}_{r['freeze'][:4]}"

    roc_data, cm_data, trained_models = {}, {}, {}

    print(f"\nGenerating plots for top 5...")
    for i, combo in enumerate(results_df.head(5).itertuples()):
        params = {'model': combo.model, 'augmentation': combo.augmentation,
                  'lr': combo.lr, 'optimizer': combo.optimizer, 'freeze': combo.freeze}

        model = build_model(params['model'], params['freeze'], config.DROPOUT).to(config.DEVICE)
        model_path = config.MODEL_SAVE_DIR / 'grid_search' / f"{i:03d}_{params['model']}_{params['optimizer']}_{params['freeze']}.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))

        metrics = evaluate(model, test_loader, config.DEVICE)
        name = f"{combo.model[:8]}_{combo.optimizer[:4]}_{combo.freeze[:4]}"
        roc_data[name] = (metrics['labels'], metrics['probs'])
        cm_data[name] = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
        trained_models[name] = model

    plot_metric_comparison(top5, config.RESULTS_DIR)
    plot_roc_curves(roc_data, config.RESULTS_DIR)
    plot_confusion_matrices(cm_data, config.RESULTS_DIR)

    sample_images = get_sample_images()
    if sample_images and trained_models:
        preds = predict_samples(trained_models, sample_images, config.DEVICE, config.IMAGE_SIZE)
        create_sample_predictions(sample_images, preds, config.RESULTS_DIR)
        print(f"Sample predictions saved to {config.RESULTS_DIR}")

    print(f"\nTotal time: {(time.time() - start_time)/3600:.2f} hours")


if __name__ == "__main__":
    main()