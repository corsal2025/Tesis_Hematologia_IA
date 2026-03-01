import os
import shutil
from pathlib import Path

def estructurar_laboratorio():
    print("="*60)
    print("INICIANDO PURGA DE RESIDUOS Y REESTRUCTURACIÓN CLÍNICA")
    print("="*60)

    directorio_base = Path.cwd()
    
    # 1. Purga de basura digital
    basura_extensiones = ['.pyc', '.pyo', '.pyd', '.DS_Store', 'Thumbs.db', 'desktop.ini']
    directorios_basura = ['__pycache__', '.ipynb_checkpoints', '.pytest_cache']

    archivos_eliminados = 0
    for root, dirs, files in os.walk(directorio_base, topdown=False):
        for name in files:
            if any(name.endswith(ext) for ext in basura_extensiones):
                try:
                    os.remove(os.path.join(root, name))
                    archivos_eliminados += 1
                except Exception:
                    pass
        for name in dirs:
            if name in directorios_basura:
                try:
                    shutil.rmtree(os.path.join(root, name))
                    archivos_eliminados += 1
                except Exception:
                    pass

    print(f"[OK] Purga completada. {archivos_eliminados} elementos residuales eliminados.")

    # 2. Creación de la Jerarquía Profesional
    estructura = [
        "data/raw",               # Los 50 GB originales intactos
        "data/procesado/images",  # Imágenes redimensionadas y normalizadas
        "data/procesado/masks",   # Máscaras generadas
        "data/clasificacion/anemia_microcitica", # Ferropénicas, Talasemias
        "data/clasificacion/anemia_normocitica", # Normales, Crónicas tempranas
        "data/clasificacion/anemia_macrocitica", # Megaloblásticas, Déficit B12
        "modelos/pesos",          # Archivos .pth
        "reportes/pdf",           # Informes finales generados
        "reportes/graficos",      # Scatter plots y estadísticas
        "src/utils",              # Scripts auxiliares
        "mcp_server"              # Integración Antigravity
    ]

    for carpeta in estructura:
        try:
            (directorio_base / carpeta).mkdir(parents=True, exist_ok=True)
            print(f"[OK] Estructura verificada: {carpeta}/")
        except Exception as e:
            print(f"[ERROR] No se pudo crear {carpeta}: {e}")

if __name__ == "__main__":
    estructurar_laboratorio()
