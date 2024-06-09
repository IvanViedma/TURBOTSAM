from skimage import morphology
from scipy.ndimage import center_of_mass
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import cv2

class ProcesarMascaras:
    """
    Clase que proporciona metodos para procesar y visualizar mascaras (segmentaciones) en imagenes
    """
    
    @staticmethod
    def mostrarMascaras(mascaras: List[Dict[str, any]]) -> np.ndarray:
        """
        Visualiza las mascaras (segmentaciones) sobre una imagen. Toma una lista de mascaras en forma de diccionarios, donde cada anotación contiene la información sobre la segmentacion de una region de interes en la imagen, como su area y la mascara de segmentacion.

        Primero, ordena las mascaras por area de manera descendente.
        Luego, crea una imagen transparente donde se visualizarán las mascaras.
        Colorea las regiones de las mascaras en la imagen transparente con colores aleatorios.
        Finalmente, muestra la imagen con las mascaras resaltadas y devuelve esta imagen.

        Args:
            mascaras (list): Lista de mascaras.

        Returns:
            np.ndarray: La imagen con las mascaras resaltadas.
        """
        try:
            if len(mascaras) == 0:
                return
            
            ordenarMascaras = sorted(mascaras, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.ones((ordenarMascaras[0]['segmentation'].shape[0], ordenarMascaras[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in ordenarMascaras:
                m = ann['segmentation']
                colorearMascara = np.concatenate([np.random.random(3), [0.35]])
                img[m] = colorearMascara
            ax.imshow(img)
            return img
        except Exception:
            raise
    
    @staticmethod
    def mostrarLabels(mascaras: List[Dict[str, any]]) -> np.ndarray:
        """
        Toma una lista de mascaras y genera una matriz de etiquetas donde cada region segmentada tiene un valor unico de etiqueta.

        Recorre todas las mascaras y convierte las coordenadas de los segmentos en mascaras binarias.
        Luego, combina todas estas mascaras en una sola matriz de etiquetas asignando valores de etiqueta unicos a cada region segmentada.
        Finalmente, devuelve la matriz de etiquetas resultante que representa todas las regiones segmentadas de las mascaras.

        Args:
            mascaras (list): Lista de mascaras.

        Returns:
            np.ndarray: La imagen con las mascaras resaltadas.
        """
        try:
            if len(mascaras) == 0:
                return None

            # Crear una lista para almacenar las máscaras
            listaMascaras = []

            # Recorrer todas las mascaras
            for ann in mascaras:
                # Obtener las coordenadas de los segmentos
                segmentation = ann['segmentation']

                # Crear una matriz de ceros para la máscara
                mascara = np.zeros((segmentation.shape[0], segmentation.shape[1]), dtype=np.uint8)

                # Establecer los píxeles correspondientes a los segmentos en 1
                mascara[segmentation] = 1

                # Agregar la máscara a la lista
                listaMascaras.append(mascara)

            # Combinar todas las máscaras en una sola matriz de etiquetas
            labels = np.zeros_like(listaMascaras[0], dtype=np.uint32)
            for i, mascara in enumerate(listaMascaras, start=1):
                labels[mascara == 1] = i

            return labels
        except Exception:
            raise
    
    @staticmethod
    def procesarMascaras(imagenEtiquetada: np.ndarray, mascarasInfo: List[Dict[str, any]], imagenOriginal: np.ndarray, min_size: int, max_size: int, min_intensity: int) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        '''
        Crea una imagen de etiquetas agregando mascaras una a una en una imagen vacia, dada la informacion de las mascaras.

        Inputs:
        - imagenEtiquetada: Una imagen de etiquetas inicializada con ceros, con el mismo tamaño y forma que la imagen original.
        - mascarasInfo: Una lista de diccionarios, cada uno conteniendo informacion sobre una mascara segmentada.
        - imagenOriginal: La imagen original en la que se aplicaran las mascaras.
        - min_size: El umbral de tamaño minimo para considerar una mascara.
        - max_size: El umbral de tamaño maximo para considerar una mascara.
        - min_intensity: La intensidad minima requerida para que un pixel sea considerado parte de una mascara.

        Outputs:
        - Una tupla que contiene:
            - Una imagen de etiquetas que contiene todas las mascaras aplicadas.
            - Una lista de diccionarios que contiene informacion sobre las mascaras filtradas.

        '''
        try:
            mascarasFiltradas = []
            ordenarMascaras = sorted(mascarasInfo, key=lambda x: x['area'], reverse=True)

            for enum, mascaraInfo in enumerate(ordenarMascaras):
                segmentation = mascaraInfo['segmentation']
                area = mascaraInfo['area']
                mnarray = np.array(segmentation, dtype=bool)
                mnarray[imagenOriginal < min_intensity] = False
                pixels = imagenOriginal[mnarray]

                if pixels.size == 0:
                    continue

                mnarray = morphology.opening(mnarray, morphology.disk(3))

                if area > min_size and area < max_size:
                    imagenEtiquetada[mnarray] = enum + 1
                    mascarasFiltradas.append(mascaraInfo)

            return imagenEtiquetada, mascarasFiltradas
        except Exception:
            raise
    
    @staticmethod
    def pintarCentroidesMascaras(mascaras: List[Dict[str, any]]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Pinta los centroides de las máscaras en una imagen.

        Args:
            mascaras (list[dict]): Lista de diccionarios que contienen información sobre las máscaras.

        Returns:
            Tuple[np.ndarray, list]: Una tupla que contiene la imagen con los centroides pintados y una lista de coordenadas de los centroides.
        """
        try:
            # Crear una imagen vacía del mismo tamaño que las máscaras
            altura, anchura = mascaras[0]['segmentation'].shape
            img = np.zeros((altura, anchura), dtype=np.uint8)

            # Crear una lista para almacenar las posiciones de los centroides
            centroides = []
        
            # Iterar sobre cada máscara y calcular el centroidee
            for mascara in mascaras:
                # Calcular el centroidee de la máscara
                centroide = center_of_mass(mascara['segmentation'])

                # Dibujar un punto en el centroidee de la máscara
                centroideY, centroideX = map(int, centroide)
                cv2.circle(img, (centroideX, centroideY), 3, (255, 255, 255), -1)
                
                # Agregar las coordenadas del centroidee a la lista de centroides
                centroides.append((centroideX, centroideY))

            return img, centroides
        except Exception:
            raise

    @staticmethod
    def superponerMascaras(mascarasPorCuadrante: List[List[Dict]], dimensiones: Tuple[int, int]) -> List[dict]:
        """
        Superpone las mascaras generadas por cuadrante en una sola lista de mascaras en la imagen original.

        Args:
            mascarasPorCuadrante (list): Lista de listas de mascaras por cuadrante.
            dimensiones (tuple): Dimensiones totales de la imagen original (altura, anchura).

        Returns:
            list: Lista de máscaras superpuestas en la imagen original.
        """
        try:
            alturaTotal, anchuraTotal = dimensiones
            raiz = int(np.sqrt(len(mascarasPorCuadrante)))
            alturaCuadrante = alturaTotal // raiz
            anchuraCuadrante = anchuraTotal // raiz

            mascarasSuperpuestas = []

            idx = 0
            for i in range(raiz):
                for j in range(raiz):
                    mascaras = mascarasPorCuadrante[idx]
                    
                    for mascara in mascaras:
                        segmentacion = mascara['segmentation']
                        nuevaSegmentacion = np.zeros((alturaTotal, anchuraTotal), dtype=bool)
                        y, x = np.where(segmentacion)
                        
                        y += i * alturaCuadrante
                        x += j * anchuraCuadrante
                        
                        nuevaSegmentacion[y, x] = True
                        
                        mascaraSuperpuesta = {
                            'segmentation': nuevaSegmentacion,
                            'area': mascara['area'],
                            'bbox': mascara['bbox'],
                            'predicted_iou': mascara['predicted_iou'],
                            'point_coords': mascara['point_coords'],
                            'stability_score': mascara['stability_score'],
                            'crop_box': mascara['crop_box']
                        }
                        mascarasSuperpuestas.append(mascaraSuperpuesta)
                        
                    idx += 1

            return mascarasSuperpuestas
        except Exception:
            raise
    