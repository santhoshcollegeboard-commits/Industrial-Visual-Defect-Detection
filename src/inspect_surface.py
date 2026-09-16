"""
Automated Visual Surface Inspection Tool.
Simulates manufacturing line quality control:
Takes raw steel strip surface images, applies CLAHE enhancement, executes neural inference,
determines defect severity and returns an automated PASS / FAIL inspection decision.
"""

import os
import argparse
from typing import Dict, Any
import numpy as np
import cv2
import torch
import torch.nn.functional as F

try:
    from .model import IndustrialDefectCNN
    from .preprocessor import IndustrialPreprocessor
    from .dataset import CLASSES
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from model import IndustrialDefectCNN
    from preprocessor import IndustrialPreprocessor
    from dataset import CLASSES


class SurfaceInspector:
    def __init__(self, checkpoint_path: str = None, confidence_threshold: float = 0.65, device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.confidence_threshold = confidence_threshold
        self.preprocessor = IndustrialPreprocessor(clip_limit=3.0)
        self.classes = CLASSES

        if checkpoint_path is None:
            default_candidates = [
                '../../development/model_checkpoints/industrial_defect_best.pth',
                '../development/model_checkpoints/industrial_defect_best.pth',
                './development/model_checkpoints/industrial_defect_best.pth',
                'model_checkpoints/industrial_defect_best.pth'
            ]
            for c in default_candidates:
                if os.path.exists(c):
                    checkpoint_path = c
                    break

        self.model = IndustrialDefectCNN(in_channels=1, num_classes=len(self.classes)).to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"Loaded inspection weights from: {checkpoint_path}")
        else:
            print("Running with initialized model weights (no checkpoint specified).")
        self.model.eval()

    def inspect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Executes automated inspection on a single surface frame.
        """
        enhanced = self.preprocessor.enhance(image)
        norm = (enhanced.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        defect_type = self.classes[pred_idx]

        inspection_status = "FAIL (DEFECTIVE)" if confidence >= self.confidence_threshold else "REVIEW (LOW CONFIDENCE)"

        return {
            'defect_type': defect_type,
            'confidence': round(confidence, 4),
            'status': inspection_status,
            'class_probabilities': {cls: round(float(probs[i]), 4) for i, cls in enumerate(self.classes)}
        }


def run_sample_inspection(image_path: str = None, checkpoint_path: str = None, output_image_path: str = 'examples/inspection_demo.png'):
    inspector = SurfaceInspector(checkpoint_path=checkpoint_path)

    if image_path and os.path.exists(image_path):
        sample_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if sample_img is None:
            raise ValueError(f"Could not read image from: {image_path}")
        sample_img = cv2.resize(sample_img, (128, 128))
    else:
        # Generate a realistic test scratch sample for demonstration
        sample_img = np.random.normal(160, 15, (128, 128)).astype(np.float32)
        cv2.line(sample_img, (20, 15), (105, 115), color=240, thickness=1)
        cv2.line(sample_img, (21, 15), (106, 115), color=40, thickness=1)
        sample_img = np.uint8(np.clip(sample_img, 0, 255))

    result = inspector.inspect(sample_img)

    print("=" * 45)
    print("SURFACE INSPECTION RESULT")
    print("=" * 45)
    print(f"Decision Status:    {result['status']}")
    print(f"Detected Defect:    {result['defect_type']}")
    print(f"Model Confidence:   {result['confidence']*100:.2f}%")
    print("=" * 45)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    enhanced = inspector.preprocessor.enhance(sample_img)
    vis = np.hstack([sample_img, enhanced])
    color_vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    color = (0, 0, 255) if 'FAIL' in result['status'] else (0, 255, 255)
    cv2.putText(color_vis, f"{result['status']} | {result['defect_type']} ({result['confidence']*100:.1f}%)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_image_path, color_vis)
    print(f"Inspection demo visual saved to: {output_image_path}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run automated industrial surface defect inspection")
    parser.add_argument('--image', type=str, default=None, help="Path to surface image (optional; generates sample if omitted)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument('--output', type=str, default='examples/inspection_demo.png', help="Output path for visual verification frame")
    args = parser.parse_args()

    run_sample_inspection(image_path=args.image, checkpoint_path=args.checkpoint, output_image_path=args.output)
