import torch
import torch.nn as nn
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torchvision.models import efficientnet_b0

DEVICE = torch.device('cpu')
IDX_TO_CLASS = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

BASE_DIR = Path(__file__).parent

# ── Classifier Architecture ──
class ActualClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )
        self.cnn = base

    def forward(self, x):
        return self.cnn(x)

# ── Load Models ──
print('Loading classifier...')
cls_model = ActualClassifier(num_classes=4)
cls_model.load_state_dict(
    torch.load(BASE_DIR / 'models' / 'classifier_best.pth', map_location=DEVICE)
)
cls_model = cls_model.to(DEVICE)
cls_model.eval()
print('Classifier ready.')

print('Loading segmentation model...')
seg_model = smp.UnetPlusPlus(
    encoder_name    = 'efficientnet-b0',
    encoder_weights = None,
    in_channels     = 3,
    classes         = 1,
    activation      = None
)
seg_model.load_state_dict(
    torch.load(BASE_DIR / 'models' / 'segmentor_best.pth', map_location=DEVICE)
)
seg_model = seg_model.to(DEVICE)
seg_model.eval()
print('Segmentation model ready.')

# ── Transforms ──
cls_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

seg_transform = A.Compose([
    A.Resize(512, 512),
    A.CLAHE(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── predict function ──
# -- Updated predict function inside model.py --

def predict(image_path: str, seg_threshold: float = 0.60) -> dict:
    img_path_obj = Path(image_path) # Convert to Path object for safer handling
    img_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img_raw is None:
        raise FileNotFoundError(f'Cannot load image: {image_path}')

    if img_raw.dtype == np.uint16:
        img_raw = ((img_raw - img_raw.min()) /
                   (img_raw.max() - img_raw.min() + 1e-6) * 255).astype(np.uint8)
    
    if len(img_raw.shape) == 2:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # 1. Classification
    cls_in = cls_transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(cls_in)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
        conf   = probs[idx].item() * 100
        all_p  = {IDX_TO_CLASS[i]: round(probs[i].item() * 100, 2)
                  for i in range(4)}
    diagnosis = IDX_TO_CLASS[idx]

    # 2. Segmentation
    seg_in = seg_transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        seg_out  = seg_model(seg_in)
        prob_map = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
        mask_bin = (prob_map > seg_threshold).astype(np.uint8)

    coverage = round(
        float(mask_bin.sum()) / (mask_bin.shape[0] * mask_bin.shape[1]) * 100, 4
    )

    if diagnosis == 'Stone':
        severity = 'Low' if coverage < 5 else ('Medium' if coverage < 20 else 'Critical')
    else:
        severity = 'N/A'

    # --- 3. UPDATED: VISUAL RESULT GENERATION ---
    # Resize original to match segmentation size
    overlay = cv2.resize(img_rgb, (512, 512))
    
    # Create red mask overlay
    red_mask = np.zeros_like(overlay)
    red_mask[mask_bin == 1] = [255, 0, 0] 
    
    # Increase mask weight to 0.5 for better visibility
    visualization = cv2.addWeighted(overlay, 0.6, red_mask, 0.4, 0)
    
    # Create a unique name: kiney_test1.jpg -> kiney_test1_mask_result.jpg
    # This avoids string replace errors and cleanup deletion
    save_name = f"{img_path_obj.stem}_mask_result.jpg"
    save_path = img_path_obj.with_name(save_name)
    
    cv2.imwrite(str(save_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    # --------------------------------------------

    return {
        'diagnosis':          diagnosis,
        'confidence_pct':     round(conf, 2),
        'all_probabilities':  all_p,
        'stone_coverage_pct': coverage,
        'severity':           severity,
        'mask':               mask_bin,
        'prob_map':           prob_map,
        'original_image':     img_rgb,
        'result_image_path':  str(save_path) # Return the full path as a string
    }