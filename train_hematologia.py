import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN DEL LABORATORIO (Hardware)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 512  # Resolución óptima para tu RTX 3050
IMG_WIDTH = 512
BATCH_SIZE = 8    # Balance para no agotar los 4GB de VRAM
LEARNING_RATE = 1e-4
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# ==========================================
# 2. AUDITORÍA DE MUESTRAS (Integridad)
# ==========================================
def realizar_auditoria():
    print(f"--- Iniciando Auditoría de Muestras ---")
    if not os.path.isdir(IMG_DIR):
        print(f"Directorio de imágenes no encontrado: {IMG_DIR}")
        return []
    if not os.path.isdir(MASK_DIR):
        print(f"Directorio de máscaras no encontrado: {MASK_DIR}")
        return []

    imagenes = {os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    mascaras = {os.path.splitext(f)[0] for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    validas = imagenes & mascaras
    print(f"Muestras válidas para entrenamiento: {len(validas)}")
    print(f"Alertas (Fotos sin máscara): {len(imagenes - mascaras)}")
    return list(validas)

# ==========================================
# 3. GESTOR DE DATOS (HematologiaDataset)
# ==========================================
class HematologiaDataset(Dataset):
    def __init__(self, lista_validas):
        self.lista_validas = lista_validas

    def __len__(self):
        return len(self.lista_validas)

    def __getitem__(self, idx):
        nombre = self.lista_validas[idx]
        img_path = os.path.join(IMG_DIR, nombre + ".jpg")
        mask_path = os.path.join(MASK_DIR, nombre + ".png") # Ajustar según tu extensión

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = (mask > 127).astype(np.float32) # Normalización binaria

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask

# ==========================================
# 4. MÉTRICA CIENTÍFICA (Dice Coefficient)
# ==========================================
def dice_coeff(preds, targets):
    preds = torch.sigmoid(preds) > 0.5
    intersection = (preds * targets).sum()
    return (2. * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)

# ==========================================
# 5. EJECUCIÓN PRINCIPAL (Entrenamiento)
# ==========================================
def main():
    muestras_validas = realizar_auditoria()
    if not muestras_validas:
        print("Error: No se encontraron muestras válidas. Revisa las carpetas.")
        return

    dataset = HematologiaDataset(muestras_validas)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = smp.Unet(
        encoder_name="mobilenet_v2", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\nEntrenamiento iniciado en {torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE}...")
    
    model.train()
    for epoch in range(1, 11): # 10 ciclos (Epochs)
        loop = tqdm(loader)
        total_dice = 0
        
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dice = dice_coeff(predictions, targets)
            total_dice += dice.item()
            
            loop.set_description(f"Epoch [{epoch}/10]")
            loop.set_postfix(loss=loss.item(), dice=dice.item())

    print("\n--- Entrenamiento Finalizado con Éxito ---")
    torch.save(model.state_dict(), "modelo_hematologia_final.pth")
    print("Modelo guardado como 'modelo_hematologia_final.pth'")

if __name__ == "__main__":
    main()
