"""Evaluate saved grid search models and generate visualizations."""

import numpy as np
import pandas as pd
import torch

import config
from dataset import BirdDataset
from metrics import evaluate
from utils import load_data, get_sample_images, get_transforms, build_model
from visualisation import (plot_metric_comparison, plot_roc_curves,
                           plot_confusion_matrices, predict_samples, create_sample_predictions)

def main():
    print(f"{'='*60}\nEVALUATE GRID SEARCH MODELS - {config.DEVICE}\n{'='*60}")

    results_path = config.RESULTS_DIR / 'grid_search_results.csv'
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run grid_search.py first.")
        return

    results_df = pd.read_csv(results_path).sort_values('test_f1', ascending=False)
    print(f"Loaded {len(results_df)} configurations")

    _, _, test_df, _ = load_data()
    test_ds = BirdDataset(test_df, get_transforms(config.IMAGE_SIZE))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    top5 = results_df.head(5)
    roc_data, cm_data, trained_models = {}, {}, {}

    print(f"\nEvaluating top 5 configurations...")
    for i, row in enumerate(top5.itertuples()):
        model = build_model(row.model, row.freeze, config.DROPOUT).to(config.DEVICE)
        model_path = config.MODEL_SAVE_DIR / 'grid_search' / f"top{i}_{row.model}_{row.optimizer}_{row.freeze}.pth"

        if not model_path.exists():
            print(f"  Warning: {model_path.name} not found, skipping")
            continue

        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
        model.eval()

        metrics = evaluate(model, test_loader, config.DEVICE)
        name = f"{row.model[:8]}_{row.optimizer[:4]}_{row.freeze[:4]}"

        roc_data[name] = (metrics['labels'], metrics['probs'])
        cm_data[name] = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
        trained_models[name] = model

        print(f"  {name}: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    top5_list = top5.to_dict('records')
    for r in top5_list:
        r['model'] = f"{r['model'][:8]}_{r['optimizer'][:4]}_{r['freeze'][:4]}"

    plot_metric_comparison(top5_list, config.RESULTS_DIR)
    plot_roc_curves(roc_data, config.RESULTS_DIR)
    plot_confusion_matrices(cm_data, config.RESULTS_DIR)
    print(f"\nPlots saved to {config.RESULTS_DIR}")

    sample_images = get_sample_images()
    if sample_images and trained_models:
        preds = predict_samples(trained_models, sample_images, config.DEVICE, config.IMAGE_SIZE)
        create_sample_predictions(sample_images, preds, config.RESULTS_DIR)
        print(f"Sample predictions saved to {config.RESULTS_DIR}")
    else:
        print("Warning: No sample images or models found.")


if __name__ == "__main__":
    main()