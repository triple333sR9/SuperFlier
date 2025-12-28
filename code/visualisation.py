"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_curve
import torch
from torchvision import transforms


def plot_metric_comparison(results_list, save_dir):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
    models = [r['model'] for r in results_list]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        values = [r[f'test_{metric}'] for r in results_list]
        bars = ax.bar(models, values, color='steelblue')
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.set_xticklabels(models, rotation=45, ha='right')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f'{val:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / 'metric_comparison.png', dpi=150)
    plt.close()


def plot_training_curves(history_dict, save_dir):
    fig, axes = plt.subplots(len(history_dict), 2, figsize=(12, 4 * len(history_dict)))
    if len(history_dict) == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, history) in enumerate(history_dict.items()):
        axes[idx, 0].plot(history['train_loss'], label='Train')
        axes[idx, 0].plot(history['val_loss'], label='Val')
        axes[idx, 0].set_title(f'{model_name} - Loss')
        axes[idx, 0].legend()

        axes[idx, 1].plot(history['train_acc'], label='Train')
        axes[idx, 1].plot(history['val_acc'], label='Val')
        axes[idx, 1].set_title(f'{model_name} - Accuracy')
        axes[idx, 1].legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_roc_curves(roc_data, save_dir):
    plt.figure(figsize=(8, 8))
    for model_name, (labels, probs) in roc_data.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=model_name)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves.png', dpi=150)
    plt.close()


def plot_confusion_matrices(cm_data, save_dir):
    n_models = len(cm_data)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['NoFly', 'Fly'], yticklabels=['NoFly', 'Fly'])
        ax.set_title(model_name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=150)
    plt.close()


def create_sample_predictions(sample_images, model_predictions, save_dir):
    """Create images with all model predictions overlaid."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for img_info in sample_images:
        img = Image.open(img_info['filepath']).convert('RGB')
        img = img.resize((600, 600))
        draw = ImageDraw.Draw(img)

        true_label = "Flying" if img_info['label'] == 1 else "Not Flying"
        text_lines = [f"True: {true_label}", "-" * 25]

        for model_name, preds in model_predictions.items():
            pred_info = preds[img_info['id']]
            pred_label = "Flying" if pred_info['pred'] == 1 else "Not Flying"
            conf = pred_info['prob']
            text_lines.append(f"{model_name}: {pred_label} ({conf:.2f})")

        text = "\n".join(text_lines)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        padding = 10
        draw.rectangle([5, 5, text_w + padding * 2 + 5, text_h + padding * 2 + 5], fill=(0, 0, 0, 200))
        draw.text((padding + 5, padding + 5), text, fill='white', font=font)

        img.save(save_dir / f"sample_{img_info['id']:05d}.jpg")


def predict_samples(models_dict, sample_images, device, image_size):
    """Get predictions from all models for sample images."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    predictions = {}
    for model_name, model in models_dict.items():
        model.eval()
        predictions[model_name] = {}

        for img_info in sample_images:
            img = Image.open(img_info['filepath']).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                prob = model(img_tensor).item()

            predictions[model_name][img_info['id']] = {
                'prob': prob,
                'pred': 1 if prob > 0.5 else 0
            }

    return predictions


def generate_all_plots(results_list, history_dict, roc_data, cm_data, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(results_list, save_dir)
    plot_training_curves(history_dict, save_dir)
    plot_roc_curves(roc_data, save_dir)
    plot_confusion_matrices(cm_data, save_dir)