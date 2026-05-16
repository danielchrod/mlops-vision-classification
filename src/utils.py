import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def get_transforms(img_size=512):
    """Retorna las transformaciones base para validación/inferencia."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_checkpoint(model, path, device):
    """Carga los pesos del modelo de forma segura."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
