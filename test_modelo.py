import torch
import segmentation_models_pytorch as smp
import time

def main():
    # 1. Configuración del dispositivo (Tu RTX 3050)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Calibración de Hardware ---")
    print(f"Dispositivo detectado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 2. Definición del Modelo Híbrido
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    # 3. Prueba de Estrés (Dry Run)
    batch_size = 4
    H, W = 512, 512
    input_tensor = torch.randn(batch_size, 3, H, W).to(device)

    print(f"\nIniciando prueba de inferencia con lote de {batch_size} imágenes...")
    start_time = time.time()

    try:
        with torch.no_grad():
            output = model(input_tensor)
        success = True
    except RuntimeError as e:
        print('RuntimeError durante inferencia (posible OOM):', e)
        success = False
    except Exception as e:
        print('Error inesperado durante inferencia:', e)
        success = False

    end_time = time.time()

    # 4. Reporte de Consumo de Memoria
    if torch.cuda.is_available():
        mem_reserved = torch.cuda.memory_reserved(0) / 1e6
    else:
        mem_reserved = 0.0

    print(f"--- Resultados del Test ---")
    print(f"Tiempo de procesamiento: {end_time - start_time:.4f} segundos")
    print(f"Memoria de Video (VRAM) reservada: {mem_reserved:.2f} MB")
    print(f"Estado: {'EXITOSO' if success and mem_reserved < 4000 else 'ALERTA: Memoria Insuficiente o fallo'}")

if __name__ == '__main__':
    main()
