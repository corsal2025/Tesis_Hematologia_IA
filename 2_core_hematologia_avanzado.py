import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

# Diccionario de Colores Profesionales (RGB) para Informes Visuales
CLASES_COLOR = {
    "Fondo": (0, 0, 0),
    "Eritrocito": (0, 0, 255),       # Rojo
    "Leucocito": (255, 0, 0),        # Azul
    "Plaqueta": (0, 255, 0),         # Verde
    "Borde_Separacion": (255, 255, 0)# Amarillo (Watershed lines)
}

class ProcesadorMorfologico:
    def __init__(self):
        # La arquitectura base sigue siendo U-Net + MobileNetV2 por eficiencia térmica y de VRAM
        pass

    def separar_celulas_superpuestas(self, mascara_probabilidad):
        """
        Aplica Watershed Algorithm para separar eritrocitos aglomerados.
        Transforma una máscara binaria en una segmentación de instancias.
        """
        # Binarización de alta precisión
        _, binaria = cv2.threshold(mascara_probabilidad, 0.5, 255, cv2.THRESH_BINARY)
        binaria = np.uint8(binaria)

        # Eliminación de ruido microscópico (Apertura morfológica)
        kernel = np.ones((3,3), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)

        # Identificación del fondo seguro (Dilatación)
        fondo_seguro = cv2.dilate(apertura, kernel, iterations=3)

        # Transformada de distancia para encontrar los centros exactos de cada célula
        dist_transform = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
        _, centros_seguros = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        centros_seguros = np.uint8(centros_seguros)

        # Zona desconocida (bordes en disputa)
        zona_desconocida = cv2.subtract(fondo_seguro, centros_seguros)

        # Etiquetado de marcadores
        _, marcadores = cv2.connectedComponents(centros_seguros)
        marcadores = marcadores + 1
        marcadores[zona_desconocida == 255] = 0

        # Aplicación de Watershed para generar contornos finos y elegantes
        imagen_color = cv2.cvtColor(mascara_probabilidad, cv2.COLOR_GRAY2BGR)
        marcadores_finales = cv2.watershed(imagen_color, marcadores)
        
        return marcadores_finales

    def clasificar_anemia_por_volumen(self, marcadores_instancias, calibracion_pixel_um):
        """
        Extrae el área de cada célula separada para determinar la distribución 
        del Volumen Corpuscular Medio (VCM) digital.
        """
        areas = []
        for etiqueta in np.unique(marcadores_instancias):
            if etiqueta <= 1: # Ignorar fondo y bordes
                continue
            mascara_celula = np.zeros_like(marcadores_instancias, dtype=np.uint8)
            mascara_celula[marcadores_instancias == etiqueta] = 255
            area_pixeles = cv2.countNonZero(mascara_celula)
            areas.append(area_pixeles * calibracion_pixel_um)
        
        vcm_digital_promedio = np.mean(areas) if areas else 0
        
        # Clasificación diagnóstica presuntiva
        if vcm_digital_promedio < 80.0:
            return "Microcítica (Sugerente: Ferropénica, Talasemia, Enf. Crónica)"
        elif vcm_digital_promedio > 100.0:
            return "Macrocítica (Sugerente: Megaloblástica, Déficit B12/Folato)"
        else:
            return "Normocítica"

# Métrica de Rigor: Dice Coefficient Multi-Clase
def dice_coeff_multiclase(preds, targets, num_clases=3):
    dice_scores = []
    preds = torch.argmax(preds, dim=1)
    for clase in range(num_clases):
        pred_clase = (preds == clase).float()
        target_clase = (targets == clase).float()
        interseccion = (pred_clase * target_clase).sum()
        dice = (2. * interseccion + 1e-7) / (pred_clase.sum() + target_clase.sum() + 1e-7)
        dice_scores.append(dice)
    return torch.mean(torch.stack(dice_scores))
