from typing import Union, List, Tuple
import cv2
import numpy as np
import math

class Utils:
    """
    Clase que contiene metodos utiles para el procesamiento de imagenes y otras utilidades.
    """

    @staticmethod
    def generarCuadrantes(dimensiones: Tuple[int, int], numCuadrantes: int, sectores: bool = False) -> np.ndarray:
        """
        Genera una matriz de etiquetas que representa los cuadrantes de una imagen.

        Args:
            dimensiones (tuple): Dimensiones de la imagen (altura, anchura).
            numCuadrantes (int): NUmero total de cuadrantes deseados.
            sectores (bool): Indica si se deben asignar fondo a los cuadrantes.

        Returns:
            np.ndarray: Matriz de etiquetas que representa los cuadrantes.
        """
        try:
            altura, anchura = dimensiones
            raiz = int(math.sqrt(numCuadrantes))
            alturaCuadrante = altura // raiz
            anchuraCuadrante = anchura // raiz

            # Inicializar una matriz de etiquetas
            labelsCuadrantes = np.zeros((altura, anchura), dtype=np.uint8)

            # Asignar un valor de etiqueta a cada cuadrante si sectores es True
            if sectores:
                for i in range(raiz):
                    for j in range(raiz):
                        filaInicio = i * alturaCuadrante
                        filaFin = (i + 1) * alturaCuadrante
                        colInicio = j * anchuraCuadrante
                        colFin = (j + 1) * anchuraCuadrante
                        labelsCuadrantes[filaInicio:filaFin, colInicio:colFin] = i * raiz + j + 1

            # Dibujar las líneas que separan los cuadrantes
            for i in range(raiz):
                for j in range(raiz):
                    filaInicio = i * alturaCuadrante
                    colInicio = j * anchuraCuadrante
                    cv2.line(labelsCuadrantes, (colInicio, filaInicio), (colInicio, filaInicio + alturaCuadrante), (255, 255, 255), 2)
                    cv2.line(labelsCuadrantes, (colInicio, filaInicio), (colInicio + anchuraCuadrante, filaInicio), (255, 255, 255), 2)

            return labelsCuadrantes
        except Exception:
            raise

    @staticmethod
    def recortarCuadrantes(imagen: Union[np.ndarray, None], numCuadrantes: int) -> List[np.ndarray]:
        """
        Recorta una imagen en cuadrantes.

        Args:
            imagen (np.ndarray): La imagen a recortar.
            numCuadrantes (int): Numero total de cuadrantes deseados.

        Returns:
            list[np.ndarray]: Lista de cuadrantes de la imagen.
        """
        try:
            altura, anchura, _ = imagen.shape
            raiz = int(np.sqrt(numCuadrantes))
            alturaCuadrante = altura // raiz
            anchuraCuadrante = anchura // raiz
            cuadrantes = []
            for i in range(raiz):
                for j in range(raiz):
                    inicioY = i * alturaCuadrante
                    inicioX = j * anchuraCuadrante
                    cuadrante = imagen[inicioY:inicioY + alturaCuadrante, inicioX:inicioX + anchuraCuadrante]
                    cuadrantes.append(cuadrante)
            
            return cuadrantes
        except Exception:
            raise
    
    @staticmethod
    def obtenerDimensionesImagen(imagen: Union[np.ndarray, None]) -> Tuple[int, int]:
        """
        Obtiene las dimensiones de una imagen.

        Args:
            imagen (np.ndarray): La imagen de la cual se obtendrán las dimensiones.

        Returns:
            tuple[int, int]: Altura y anchura de la imagen.
        """
        try:
            altura, anchura, _ = imagen.shape
            return (altura, anchura)
        except Exception:
            raise
    
    @staticmethod
    def convertRGB(imagen: Union[np.ndarray, None]) -> np.ndarray:
        """
        Convierte una imagen RGB en una imagen RGB donde cada canal tiene el mismo valor que el canal
        de escala de grises.

        Args:
        imagen: La imagen a convertir.

        Returns:
            np.ndarray: La imagen en formato RGB.
        """
        try:
            #Convertir a escala de grises
            imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Obtener las dimensiones de la imagen en escala de grises
            altura, anchura = imagenGris.shape

            # Crear una imagen RGB con las mismas dimensiones que la imagen en escala de grises
            imagenRGB = np.zeros((altura, anchura, 3), dtype=np.uint8)

            # Asignar el valor del canal de escala de grises a cada canal de la imagen RGB
            imagenRGB[:, :, 0] = imagenGris
            imagenRGB[:, :, 1] = imagenGris
            imagenRGB[:, :, 2] = imagenGris

            return imagenRGB
        except Exception:
            raise
    
    @staticmethod
    def calcularErrores(numeroRodOrginial: int, numeroRodCalculado: int) -> Tuple[int, float]:
        """
        Calcula el error absoluto y relativo entre el numero original y el numero calculado.

        Args:
            numeroRodOrginial (int): El numero original de rodaballos.
            numeroRodCalculado (int): El numero de rodaballos calculado.

        Returns:
            Tuple[int, float]: Una tupla que contiene el error absoluto y el error relativo.
        """
        try:
            errorAbsoluto = abs(numeroRodOrginial - numeroRodCalculado)
            errorRelativo = (abs(numeroRodOrginial - numeroRodCalculado) / numeroRodOrginial) * 100
            errorRelativo = round(errorRelativo, 2)
            
            return errorAbsoluto, errorRelativo
        except Exception:
            raise