from typing import Dict, Tuple, Optional
import numpy as np
import cv2
import torch
import torchvision.models.segmentation as seg
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

class SemanticSegmenter:
    """Pretrained lightweight semantic segmentation for 3D digital twin labeling."""
    
    PALETTE = np.array([
        [40, 40, 40],       # 0: background
        [128, 0, 0],        # 1: aeroplane
        [0, 128, 0],        # 2: bicycle
        [128, 128, 0],      # 3: bird
        [0, 0, 128],        # 4: boat
        [128, 0, 128],      # 5: bottle / cup
        [0, 128, 128],      # 6: bus
        [128, 128, 128],    # 7: car
        [64, 0, 0],         # 8: cat
        [192, 0, 0],        # 9: chair
        [64, 128, 0],       # 10: cow
        [192, 128, 0],      # 11: diningtable / desk
        [64, 0, 128],       # 12: dog
        [192, 0, 128],      # 13: horse
        [64, 128, 128],     # 14: motorbike
        [192, 128, 128],    # 15: person
        [0, 64, 0],         # 16: pottedplant
        [128, 64, 0],       # 17: sheep
        [0, 192, 0],        # 18: sofa
        [128, 192, 0],      # 19: train
        [0, 64, 128],       # 20: tvmonitor / computer screen
    ], dtype=np.uint8)

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = seg.lraspp_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.transforms = weights.transforms()
        self.categories = weights.meta["categories"]

    def predict_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Takes BGR uint8 image, returns (H, W) uint8 class IDs and detected labels dict."""
        orig_h, orig_w = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        inp = self.transforms(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)["out"]
            # Interpolate logits back to original image resolution
            out = torch.nn.functional.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            class_map = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        unique_ids = np.unique(class_map)
        detected = {int(cid): self.categories[cid] for cid in unique_ids if cid < len(self.categories)}
        return class_map, detected

    def colorize_mask(self, class_map: np.ndarray) -> np.ndarray:
        """Returns (H, W, 3) RGB visualization of the semantic mask."""
        return self.PALETTE[np.clip(class_map, 0, len(self.PALETTE) - 1)]

if __name__ == "__main__":
    segmenter = SemanticSegmenter()
    dummy = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mask, labels = segmenter.predict_mask(dummy)
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    color = segmenter.colorize_mask(mask)
    assert color.shape == (480, 640, 3)
    print(f"SemanticSegmenter verified: detected {list(labels.values())}")
