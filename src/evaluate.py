"""
Evaluation pipeline for Industrial Visual Defect Detection.
Computes multi-class precision, recall, F1-score, overall accuracy,
generates normalized confusion matrix, and visualizes CLAHE contrast comparisons.
"""

import os
import json
import argparse
from typing import Dict, Any
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

try:
    from .dataset import get_industrial_dataloaders, CLASSES
    from .model import IndustrialDefectCNN
    from .preprocessor import IndustrialPreprocessor
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from dataset import get_industrial_dataloaders, CLASSES
    from model import IndustrialDefectCNN
    from preprocessor import IndustrialPreprocessor


def plot_defect_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 7))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, square=True
    )
    plt.title('Normalized Industrial Defect Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Defect Type', fontsize=11, labelpad=8)
    plt.ylabel('True Defect Type', fontsize=11, labelpad=8)
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_clahe_comparison(test_loader, preprocessor: IndustrialPreprocessor, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images, labels = test_loader.dataset.images, test_loader.dataset.labels
    classes = CLASSES

    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 2.5 * len(classes)))

    for i, cls_name in enumerate(classes):
        match_indices = np.where(labels == i)[0]
        if len(match_indices) == 0:
            continue
        sample_img = images[match_indices[0]]

        # Raw image
        axes[i, 0].imshow(sample_img, cmap='gray')
        axes[i, 0].set_title(f"Raw Scan: {cls_name}", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Enhanced with CLAHE
        enhanced_img = preprocessor.enhance(sample_img)
        axes[i, 1].imshow(enhanced_img, cmap='gray')
        axes[i, 1].set_title(f"CLAHE Enhanced", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # Difference / Contrast Map
        diff = cv2.absdiff(enhanced_img, sample_img)
        axes[i, 2].imshow(diff, cmap='inferno')
        axes[i, 2].set_title(f"Amplified Flaw Gradient", fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CLAHE contrast comparison saved to: {save_path}")


def evaluate_defect_model(
    checkpoint_path: str,
    batch_size: int = 32,
    fig_dir: str = './github/examples',
    log_dir: str = './development/logs'
) -> Dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Industrial Evaluation on device: {device}")

    _, _, test_loader, classes = get_industrial_dataloaders(batch_size=batch_size, apply_clahe=True)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IndustrialDefectCNN(in_channels=1, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.long()
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_targets)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    clf_report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    os.makedirs(fig_dir, exist_ok=True)
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plot_defect_confusion_matrix(cm, classes, cm_path)

    preprocessor = IndustrialPreprocessor(clip_limit=3.0)
    clahe_path = os.path.join(fig_dir, "clahe_contrast_comparison.png")
    plot_clahe_comparison(test_loader, preprocessor, clahe_path)

    metrics = {
        'total_test_samples': int(len(all_targets)),
        'overall_accuracy': float(round(accuracy, 4)),
        'macro_precision': float(round(p_macro, 4)),
        'macro_recall': float(round(r_macro, 4)),
        'macro_f1_score': float(round(f1_macro, 4)),
        'weighted_f1_score': float(round(f1_weighted, 4)),
        'per_class_metrics': {
            cls: {
                'precision': float(round(clf_report[cls]['precision'], 4)),
                'recall': float(round(clf_report[cls]['recall'], 4)),
                'f1_score': float(round(clf_report[cls]['f1-score'], 4)),
                'support': int(clf_report[cls]['support'])
            } for cls in classes
        }
    }

    print("\n" + "="*55)
    print("INDUSTRIAL DEFECT TEST EVALUATION RESULTS")
    print("="*55)
    print(f"Overall Accuracy:   {metrics['overall_accuracy']*100:.2f}%")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:     {metrics['macro_f1_score']:.4f}")
    print(f"Weighted F1-Score:  {metrics['weighted_f1_score']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls, vals in metrics['per_class_metrics'].items():
        print(f"  - {cls:<16}: F1 = {vals['f1_score']:.4f} (P={vals['precision']:.4f}, R={vals['recall']:.4f})")
    print("="*55)

    os.makedirs(log_dir, exist_ok=True)
    out_json = os.path.join(log_dir, "industrial_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Industrial metrics saved to: {out_json}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate industrial defect detection model")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to trained checkpoint (.pth). If omitted, searches default local checkpoint paths.")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if ckpt is None:
        default_candidates = [
            '../../development/model_checkpoints/industrial_defect_best.pth',
            '../development/model_checkpoints/industrial_defect_best.pth',
            './development/model_checkpoints/industrial_defect_best.pth',
            'model_checkpoints/industrial_defect_best.pth'
        ]
        for c in default_candidates:
            if os.path.exists(c):
                ckpt = c
                break

    if ckpt is None or not os.path.exists(ckpt):
        print("Note: No local checkpoint specified. Provide --checkpoint <path/to/checkpoint.pth>")
        print("Example: python src/evaluate.py --checkpoint ../development/model_checkpoints/industrial_defect_best.pth")
    else:
        evaluate_defect_model(checkpoint_path=ckpt)
