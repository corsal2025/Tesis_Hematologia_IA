import albumentations as A
from albumentations.pytorch import ToTensorV2

# Transformaciones profesionales para simular variaciones de laboratorio
transformaciones_clinicas = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.7), # Simula variaciones de tinción
    A.HorizontalFlip(p=0.5), # La rotación de la célula no afecta su diagnóstico
    A.VerticalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # Simula ruido del sensor del microscopio
    ToTensorV2()
])

__all__ = ["transformaciones_clinicas"]
