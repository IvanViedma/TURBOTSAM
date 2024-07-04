from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
import torch
import numpy as np
from typing import Iterator, Tuple, List, Dict

class TurbotSAM: 
    """
    Clase que emplea un modelo Segment Anything Model (SAM) y genera mascaras para toda la imagen.
    Genera una cuadrícula de puntos sobre la imagen, luego filtra mascaras de baja calidad y duplicadas. 

    Argumentos:
    model (Sam): El modelo SAM a utilizar para la prediccion de mascaras.
    points_per_side (int o None): El numero de puntos a muestrear a lo largo de un lado de la imagen. El numero total de puntos es points_per_side**2. Si es None, 'point_grids' debe proporcionar un muestreo de puntos explícito.
    points_per_batch (int): Establece el número de puntos ejecutados simultáneamente por el modelo. Números más altos pueden ser más rápidos pero usan más memoria GPU.
    pred_iou_thresh (float): Un umbral de filtrado en [0,1], utilizando la calidad de la máscara predicha por el modelo.
    stability_score_thresh (float): Un umbral de filtrado en [0,1], utilizando la estabilidad de la máscara bajo cambios en el umbral utilizado para binarizar las predicciones de máscara del modelo.
    stability_score_offset (float): La cantidad para desplazar el umbral al calcular el puntaje de estabilidad.
    box_nms_thresh (float): El umbral de IoU de caja utilizado por la supresión no máxima para filtrar máscaras duplicadas.
    crop_n_layers (int): Si >0, la predicción de máscara se ejecutará nuevamente en recortes de la imagen. Establece el número de capas a ejecutar, donde cada capa tiene 2**i_layer número de recortes de imagen.
    crop_nms_thresh (float): El umbral de IoU de caja utilizado por la supresion no máxima para filtrar máscaras duplicadas entre diferentes recortes.
    crop_overlap_ratio (float): Establece el grado en que se superponen los recortes. En la primera capa de recorte, los recortes se superpondrán en esta fracción de la longitud de la imagen. Capas posteriores con más recortes reducen esta superposición.
    crop_n_points_downscale_factor (int): El número de puntos por lado muestreados en la capa n se reduce en crop_n_points_downscale_factor**n.
    point_grids (list(np.ndarray) o None): Una lista de cuadrículas de puntos explicitas utilizadas para muestreo, normalizadas a [0,1]. La enésima cuadrícula en la lista se usa en la enésima capa de recorte. Exclusivo con points_per_side.
    min_mask_region_area (int): Si >0, se aplicará un postprocesamiento para eliminar regiones desconectadas y agujeros en máscaras con área menor que min_mask_region_area. Requiere opencv.
    output_mode (str): La forma en que se devuelven las máscaras. Puede ser 'binary_mask', 'uncompressed_rle' o 'coco_rle'. 'coco_rle' requiere pycocotools. Para resoluciones grandes, 'binary_mask' puede consumir grandes cantidades de memoria.
    """
        
    def __init__(self,points_per_side,points_per_batch,pred_iou_thresh,
                 stability_score_thresh,stability_score_offset,
                 box_nms_thresh,crop_n_layers,crop_nms_thresh,
                 crop_overlap_ratio,crop_n_points_downscale_factor,
                 min_mask_region_area):
        
        try:
            self.checkpoint = "models/mobile_sam.pt"
            self.modelType = "vit_t"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sam = sam_model_registry[self.modelType](checkpoint=self.checkpoint)
            self.sam.to(device=self.device)
            self.sam.eval()
            
            self.generadorMascaras = SamAutomaticMaskGenerator(
                model = self.sam,
                points_per_side = points_per_side,
                points_per_batch = points_per_batch,
                pred_iou_thresh = pred_iou_thresh,
                stability_score_thresh = stability_score_thresh,
                stability_score_offset = stability_score_offset,
                box_nms_thresh = box_nms_thresh,
                crop_n_layers = crop_n_layers,
                crop_nms_thresh = crop_nms_thresh,
                crop_overlap_ratio = crop_overlap_ratio,
                crop_n_points_downscale_factor = crop_n_points_downscale_factor,
                min_mask_region_area = min_mask_region_area,
            )
        except Exception:
            raise

    def generarMascaras(self, imagen: np.ndarray) -> List[Dict[str, any]]:
        """
        Genera mascaras a partir de una imagen utilizando el generador de mascaras asociado a esta instancia.

        Args:
            imagen: La imagen de entrada.

        Returns:
            Las máscaras generadas.
        """
        try:
            return self.generadorMascaras.generate(imagen)
        except Exception:
            raise

    def generarMascarasPorCuadrante(self, cuadrantes: List[np.ndarray], postprocesamiento: bool) -> Iterator[Tuple[float, List[any], int]]:
        """
        Genera mascaras por cuadrante a partir de una lista de cuadrantes.

        Args:
            cuadrantes: Lista de cuadrantes de la imagen.
            postprocesamiento: Indica si se realiza postprocesamiento.

        Yields:
            Tuple[float, list[Any], int]: Una tupla que contiene el progreso, las mascaras por cuadrante y el contador.
        """
        try:
            mascarasPorCuadrante = []
            numCuadrantes = len(cuadrantes)
            aux = 90 if not postprocesamiento else 50
            cont = 0
            
            for cuadrante in cuadrantes:
                cont += 1
                masks = self.generarMascaras(cuadrante)
                mascarasPorCuadrante.append(masks)
                porcentaje = (cont / numCuadrantes) * aux          
                yield porcentaje, mascarasPorCuadrante, cont     
        except Exception:
            raise        

   
