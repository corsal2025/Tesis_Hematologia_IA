# Metodología

## Configuración de hardware
- Máquina: WSL2 / VM local (según entorno de trabajo)
- RAM asignada: 32 GB
- CPU: ajustar número de núcleos según disponibilidad
- Almacenamiento: SSD recomendado para I/O de datos

## Entorno de software
- Sistema: Ubuntu (WSL2 o nativo)
- Entorno virtual: `python3 -m venv .venv`
- Dependencias: ver `requirements.txt` (numpy, pandas, matplotlib, scipy, jupyter)

## Notas de configuración inicial
1. Asignar memoria y CPUs en la configuración de WSL2 si aplica.
2. Actualizar el sistema y paquetes:
```bash
sudo apt update && sudo apt upgrade -y
```
3. Crear y activar el entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Buenas prácticas
- No subir datos sensibles al repositorio (`data/` está en `.gitignore`).
- Mantener los notebooks en `notebooks/` y el código reutilizable en `src/`.

