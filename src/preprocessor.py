"""
Industrial surface image enhancement and preprocessing pipeline.
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) and bilateral filtering
to amplify low-contrast surface texture flaws (cracks, inclusions, pits, scale, scratches).
"""

from typing import Tuple
import cv2
import numpy as np
import torch


class IndustrialPreprocessor:
    """
    Enhancement engine for steel strip and manufacturing surface defect inspection.
    """
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral smoothing followed by CLAHE contrast amplification.
        Expects grayscale image [0, 255] or float [0, 1].
        """
        if image.dtype in [np.float32, np.float64]:
            img_uint8 = np.uint8(np.clip(image * 255.0, 0, 255))
        else:
            img_uint8 = image

        if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_uint8

        # Edge-preserving denoising
        filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

        # Contrast Limited Adaptive Histogram Equalization
        enhanced = self.clahe.apply(filtered)

        return enhanced

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.enhance(image)
