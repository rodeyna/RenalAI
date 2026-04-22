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

# ── Classifier — torchvision EfficientNet-B0 ──
# Your .pth has keys like cnn.features.0.0.weight (torchvision style)
# so we wrap torchvision's EfficientNet exactly like you did in training

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=None)
        # Replace the classifier head to match what you trained
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        self.cnn = base

    def forward(self, x):
        return self.cnn(x)

# ── BUT wait — your keys show cnn.classifier.1 and cnn.classifier.4
# which means your head had TWO linear layers, not one.
# Let's check: classifier.1.weight and classifier.4.weight
# index 1 and 4 means: [0]=Dropout, [1]=Linear, [2]=ReLU/GELU,
# [3]=Dropout, [4]=Linear
# So your actual head was: Dropout → Linear → activation → Dropout → Linear

class ActualClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=None)
        in_features = base.classifier[1].in_features  # 1280 for B0
        # Match exactly your training head structure
        # keys: cnn.classifier.1 and cnn.classifier.4
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),   # index 0
            nn.Linear(in_features, 256),        # index 1  ← cnn.classifier.1
            nn.GELU(),                           # index 2
            nn.Dropout(p=0.4),                  # index 3
            nn.Linear(256, num_classes)          # index 4  ← cnn.classifier.4
        )
        self.cnn = base

    def forward(self, x):
        return self.cnn(x)


print('Loading classifier...')
cls_model = ActualClassifier(num_classes=4)
cls_model.load_state_dict(
    torch.load(BASE_DIR / 'models' / 'classifier_best.pth',
               map_location=DEVICE)
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
    torch.load(BASE_DIR / 'models' / 'segmentor_best.pth',
               map_location=DEVICE)
)
seg_model = seg_model.to(DEVICE)
seg_model.eval()
print('Segmentation model ready.')

# ── Transforms ──
cls_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

seg_transform = A.Compose([
    A.Resize(512, 512),
    A.CLAHE(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── predict function ──
def predict(image_path: str, seg_threshold: float = 0.60) -> dict:
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

    # Classification
    cls_in = cls_transform(image=img_rgb)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cls_model(cls_in)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
        conf   = probs[idx].item() * 100
        all_p  = {IDX_TO_CLASS[i]: round(probs[i].item() * 100, 2)
                  for i in range(4)}
    diagnosis = IDX_TO_CLASS[idx]

    # Segmentation
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

    return {
        'diagnosis':          diagnosis,
        'confidence_pct':     round(conf, 2),
        'all_probabilities':  all_p,
        'stone_coverage_pct': coverage,
        'severity':           severity,
        'mask':               mask_bin,
        'prob_map':           prob_map,
        'original_image':     img_rgb
    }