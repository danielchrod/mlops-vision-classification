import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset

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



def get_transforms(img_size=512, is_train=False):
    """Retorna las transformaciones base para entrenamiento o validación."""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5), # Aumento de datos básico para entrenar
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
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

class FoodDataset(Dataset):
    """Dataset personalizado para leer las imágenes y máscaras reales de las carpetas."""
    def __init__(self, data_dir, transform=None):
        all_imgs = sorted(glob.glob(os.path.join(data_dir, "images", "*")))
        all_masks = sorted(glob.glob(os.path.join(data_dir, "masks", "*")))
        
        self.img_paths = [p for p in all_imgs if os.path.isfile(p) and not p.endswith('.gitkeep')]
        self.mask_paths = [p for p in all_masks if os.path.isfile(p) and not p.endswith('.gitkeep')]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
  
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
  
        mask = cv2.imread(self.mask_paths[idx], 0) 
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"].long() 
            
        return img, mask