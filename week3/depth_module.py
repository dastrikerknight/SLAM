from typing import Optional, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class MonocularDepthEstimator:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: Optional[Union[str, torch.device]] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected input image as a numpy ndarray")

        orig_h, orig_w = image.shape[:2]

        # Ensure 3-channel RGB representation
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = outputs.predicted_depth

        # Bilinear interpolation back to input image resolution
        interpolated = F.interpolate(
            pred.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )

        depth = interpolated.squeeze().cpu().numpy().astype(np.float32)
        return depth

if __name__ == "__main__":
    # Self-check verification
    estimator = MonocularDepthEstimator()
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    depth_map = estimator.predict_depth(dummy_bgr)

    assert depth_map.shape == (480, 640), f"Shape mismatch: {depth_map.shape}"
    assert depth_map.dtype == np.float32, f"Dtype mismatch: {depth_map.dtype}"
    assert not np.isnan(depth_map).any(), "Depth map contains NaN values"
    print(f"MonocularDepthEstimator verified: input {dummy_bgr.shape} -> depth {depth_map.shape} ({depth_map.dtype})")
