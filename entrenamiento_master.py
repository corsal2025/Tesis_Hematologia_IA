import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BATCH_SIZE = 8
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

class HematologiaDataset(Dataset):
    def __init__(self, imagenes_validas):
        self.imagenes = imagenes_validas

    def __len__(self): return len(self.imagenes)

    def __getitem__(self, idx):
        nombre = self.imagenes[idx]
        img_path = os.path.join(IMG_DIR, nombre + ".jpg")
        mask_path = os.path.join(MASK_DIR, nombre + ".png")

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask

def dice_coeff(preds, targets):
    preds = torch.sigmoid(preds) > 0.5
    inter = (preds * targets).sum()
    return (2. * inter + 1e-7) / (preds.sum() + targets.sum() + 1e-7)

def main():
    imgs = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith('.jpg')}
    msks = {os.path.splitext(f)[0] for f in os.listdir(MASK_DIR) if f.endswith('.png')}
    validas = list(imgs & msks)
    
    if not validas:
        print("Esperando muestras... Pega tus imágenes en data/images y máscaras en data/masks")
        return

    print(f"Iniciando procesamiento morfológico de {len(validas)} muestras válidas.")
    dataset = HematologiaDataset(validas)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = smp.Unet("mobilenet_v2", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, 6): # 5 Épocas iniciales
        loop = tqdm(loader, desc=f"Época {epoch}")
        for data, targets in loop:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            preds = model(data)
            loss = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(Dice=dice_coeff(preds, targets).item())

    torch.save(model.state_dict(), "modelos_guardados/unet_hematologia.pth")
    print("Entrenamiento exitoso. Modelo guardado.")

if __name__ == "__main__":
    main()
