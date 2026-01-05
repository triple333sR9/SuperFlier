"""Evaluation and visualization for trained models."""

import numpy as np
import pandas as pd
import torch

import config
from model import get_model
from metrics import evaluate
from utils import DEVICE, get_sample_images
from visualisation import generate_all_plots, predict_samples, create_sample_predictions


def load_trained_model(name):
    model, _ = get_model(name, pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_SAVE_DIR / f"best_{name}.pth",
                                      map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


def evaluate_all_models(test_loader, histories):
    results, roc_data, cm_data, trained_models = [], {}, {}, {}

    for name in config.MODELS:
        print(f"Evaluating {name}...")
        model = load_trained_model(name)
        metrics = evaluate(model, test_loader, DEVICE)

        results.append({
            'model': name,
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_specificity': metrics['specificity'],
            'test_f1': metrics['f1'],
            'test_auc': metrics['auc'],
            'tp': metrics['tp'], 'tn': metrics['tn'],
            'fp': metrics['fp'], 'fn': metrics['fn'],
        })

        roc_data[name] = (metrics['labels'], metrics['probs'])
        cm_data[name] = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
        trained_models[name] = model

    results.sort(key=lambda x: x['test_f1'], reverse=True)
    return results, roc_data, cm_data, trained_models


def run_evaluation(test_loader, histories):
    print(f"\n{'='*50}\nEVALUATION\n{'='*50}")

    results, roc_data, cm_data, trained_models = evaluate_all_models(test_loader, histories)

    print(f"\n{'='*80}\nFINAL RESULTS\n{'='*80}")
    print(f"{'Model':<18} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Spec':<8} {'F1':<8} {'AUC':<8}")
    for r in results:
        print(f"{r['model']:<18} {r['test_accuracy']:<8.4f} {r['test_precision']:<8.4f} "
              f"{r['test_recall']:<8.4f} {r['test_specificity']:<8.4f} {r['test_f1']:<8.4f} {r['test_auc']:<8.4f}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(config.RESULTS_DIR / 'results.csv', index=False)
    generate_all_plots(results, histories, roc_data, cm_data, config.RESULTS_DIR)

    sample_images = get_sample_images()
    if sample_images and all(s['filepath'].exists() for s in sample_images):
        preds = predict_samples(trained_models, sample_images, DEVICE, config.IMAGE_SIZE)
        create_sample_predictions(sample_images, preds, config.RESULTS_DIR)
        print(f"\nSample predictions saved to {config.RESULTS_DIR}")
    else:
        print("\nWarning: Sample images not found. Run prepare_samples.py first.")

    print(f"\nBest model: {results[0]['model']} (F1: {results[0]['test_f1']:.4f})")
    return results