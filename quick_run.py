import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Create small synthetic dataset
IMG_DIR = 'data/images'
MASK_DIR = 'data/masks'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def make_synthetic(name, h=512, w=512):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    # draw a filled circle as cell
    center = (w//2 + np.random.randint(-40,40), h//2 + np.random.randint(-40,40))
    radius = np.random.randint(30, 80)
    color = (int(np.random.randint(100,255)), int(np.random.randint(100,255)), int(np.random.randint(100,255)))
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(mask, center, radius, 255, -1)
    cv2.imwrite(os.path.join(IMG_DIR, name + '.jpg'), img)
    cv2.imwrite(os.path.join(MASK_DIR, name + '.png'), mask)

N = 8
names = [f'synth_{i}' for i in range(N)]
for n in names:
    make_synthetic(n)

# Dataset (simple, similar to train_hematologia)
class SimpleDataset(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(IMG_DIR, name + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        mask = cv2.imread(os.path.join(MASK_DIR, name + '.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512,512))
        mask = (mask > 127).astype(np.float32)
        img = torch.from_numpy(img).permute(2,0,1).float()/255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = SimpleDataset(names)
loader = DataLoader(ds, batch_size=2, shuffle=True)

# Model
model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
opt = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

print('Starting short training run on', device)
model.train()
for epoch in range(1):
    loop = tqdm(loader, desc=f'Epoch {epoch+1}')
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        # compute dice approx
        preds = (torch.sigmoid(out) > 0.5).float()
        inter = (preds * yb).sum()
        dice = (2*inter + 1e-7) / (preds.sum() + yb.sum() + 1e-7)
        loop.set_postfix(loss=loss.item(), dice=dice.item())

print('Short training finished. Saving small checkpoint...')
torch.save(model.state_dict(), 'quick_run_checkpoint.pth')
print('Saved quick_run_checkpoint.pth')
