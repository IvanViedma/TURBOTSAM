from scripts.NapariSAM import NapariSAM 

"""
TURBOT SAM es una aplicación de segmentación de imágenes diseñada para facilitar el análisis y la segmentación de imágenes de tanques de rodaballos para GAMMALAB. Sus principales objetivos son:

Carga de Imágenes: Permite cargar imágenes de rodaballos en varios formatos comunes como TIFF, PNG y JPG.

Segmentación: Ofrece herramientas para segmentar automáticamente regiones de interés en las imágenes cargadas.

Postprocesamiento: Proporciona opciones de postprocesamiento para refinar y mejorar las segmentaciones automáticas.

Visualización Interactiva: Permite visualizar las imágenes cargadas, así como las segmentaciones resultantes de manera interactiva.

Exportación de Resultados: Facilita la exportación de los resultados de segmentación para su posterior análisis.
"""

def main():
    
    # Ejecutar la aplicación Napari
    napari_app = NapariSAM()
    napari_app.run()
    
if __name__ == "__main__":
    main()
