from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QCheckBox, QComboBox, QLabel, QProgressBar, QTextEdit, QDoubleSpinBox, QSpinBox, QGroupBox
from PyQt5.QtCore import QTimer
from typing import Union, List, Tuple
from scripts.Utils import Utils
from scripts.TurbotSAM import TurbotSAM
from scripts.ProcesarMascaras import ProcesarMascaras
from skimage.color import label2rgb
import napari
import torch
import cv2
import csv
import threading
import numpy as np
import matplotlib.pyplot as plt

class NapariSAM:
    """
    Clase principal encargada de inicializar la interfaz Napari y gestionar los procesos iniciados por el usuario.
    """
    
    def __init__(self):
        
        try:
            # Aplicacion Napari
            self.viewer = napari.Viewer()
        
            # Generar widget
            self.widget = QWidget()
            self.layout = QVBoxLayout()
            self.widget.setLayout(self.layout)
            
            # Inicializar campos
            self.__inicializarCarga()
            self.__inicializarCuadrantes()
            self.__inicializarParametros()
            self.__inicializarProcesamiento()
            self.__inicializarSegmentacion()
            self.__inicializarCuadro()
            self.__inicializarExportaciones()
            self.__incicializarVariables()
            
        
            # Agregar campos
            self.layout.addWidget(self.cargarLabel)
            self.layout.addWidget(self.cargarGroupbox)
            self.layout.addWidget(self.cuadrantesLabel)
            self.layout.addWidget(self.cuadrantesGroupbox)
            self.layout.addWidget(self.parametrosLabel)
            self.layout.addWidget(self.parametrosGroupbox)
            self.layout.addWidget(self.procesamientoLabel)
            self.layout.addWidget(self.procesamientoGroupbox)
            self.layout.addWidget(self.segmentacionLabel)
            self.layout.addWidget(self.segmentacionGroupbox)
            self.layout.addWidget(self.cuadroLabel)
            self.layout.addWidget(self.cuadroGroupbox)
            self.layout.addWidget(self.exportacionesLabel)
            self.layout.addWidget(self.exportacionesGroupbox)
            
            # Mostrar widget
            self.viewer.window.add_dock_widget(self.widget, name="TURBOT SAM")
            
            # Comprobamos si CUDA esta habilitado para poder usar la GPU
            if torch.cuda.is_available():
                self.log.append("<span style='color: green;'>[INFO]</span> CUDA habilitado. Se usará la memoria GPU para el procesamiento")
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> CUDA no está habilitado y por lo tanto no se usará la memoria GPU para el procesamiento. Se utilizará la CPU lo que puede ocasionar tiempos largos de procesamiento")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion: {str(e)}")
            
    # Funciones de inicializacion de Napari
      
    def __inicializarCarga(self) -> None:
        """
        Inicializa la interfaz de usuario para la carga de imagenes.

        Este metodo configura los elementos de la interfaz de usuario, incluyendo un boton
        para cargar imagenes y opciones de zoom para las imagenes.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.cargarLabel = QLabel("Selección de Imagen")
            self.cargarLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.cargarGroupbox = QGroupBox("")
            self.cargarLayout = QVBoxLayout()
            self.cargarGroupbox.setLayout(self.cargarLayout)
            
            # Seccion para cargar la imagen
            self.btnCargar = QPushButton("CARGAR IMAGEN")
            self.cargarLayout.addWidget(self.btnCargar)
            self.btnCargar.clicked.connect(self.__cargarImagenWidget)
            
            # Seleccionar si es una imagen con zoom o no
            self.chkZoom = QCheckBox("Imagen Con Zoom")
            self.cargarLayout.addWidget(self.chkZoom)
            self.chkZoom.setChecked(True)
            self.chkZoom.stateChanged.connect(self.__toggleZoom)

            self.chkNoZoom = QCheckBox("Imagen Sin Zoom")
            self.cargarLayout.addWidget(self.chkNoZoom)
            self.chkNoZoom.stateChanged.connect(self.__toggleNoZoom)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de carga de imagen: {str(e)}")
            
    def __inicializarCuadrantes(self) -> None:
        """
        Inicializa la interfaz de usuario para la seleccion de cuadrantes.

        Este metodo configura los elementos de la interfaz de usuario, incluyendo un menu
        desplegable para seleccionar el numero de cuadrantes y una opcion para mostrar sectores.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.cuadrantesLabel = QLabel("Selección de cuadrantes")
            self.cuadrantesLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.cuadrantesGroupbox = QGroupBox("")
            self.cuadrantesLayout = QVBoxLayout()
            self.cuadrantesGroupbox.setLayout(self.cuadrantesLayout)
            
            # Crear el menu desplegable para seleccionar el numero de cuadrantes
            self.cuadrantesCombo = QComboBox()
            self.cuadrantesCombo.setToolTip("Selecciona el número de cuadrantes a establecer. Ten en cuenta que cuantos más cuadrantes establezcas tendrás un mejor resultado de segmentación para objetos muy pequeños, no obstante deberás establecer un tamaño minimo de área correcto para evitar segmentar todo tipo de unidades.")
            self.cuadrantesCombo.addItem("Sin cuadrantes")
            self.cuadrantesCombo.addItem("4")
            self.cuadrantesCombo.addItem("9")
            self.cuadrantesCombo.addItem("16")
            self.cuadrantesCombo.addItem("25")
            self.cuadrantesCombo.addItem("36")
            self.cuadrantesLayout.addWidget(self.cuadrantesCombo)
            self.cuadrantesCombo.currentIndexChanged.connect(self.__actualizarCuadrantes)
            
            # Opcion de mostrar sectores en los cuadrantes
            self.chkSectores = QCheckBox("Mostrar sectores")
            self.chkSectores.setToolTip("Selecciona esta opción para pintar fondo sobre cada cuadrante")
            self.cuadrantesLayout.addWidget(self.chkSectores)
            self.chkSectores.stateChanged.connect(self.__actualizarSectores)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de cuadrantes: {str(e)}")
        
    def __inicializarParametros(self) -> None:
        """
        Inicializa la interfaz de usuario para los parametros de segmentacion.

        Este metodo configura los elementos de la interfaz de usuario para mostrar los parametros
        de segmentacion y sus valores por defecto.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.parametrosLabel = QLabel("Parámetros de Segmentación")
            parametrosText = """
            Selección de los valores de los parámetros de segmentación de SAM. Por defecto, se actualizan a un ajuste óptimo dependiendo de la selección de imagen con/sin zoom. Para un ajuste más preciso para cada imagen se pueden ajustar estos valores.
            
            1) points_per_side: El numero de puntos a muestrear a lo largo de un lado de la imagen. El numero total de puntos es points_per_side**2. Si es None, 'point_grids' debe proporcionar un muestreo de puntos explícito.
            2) points_per_batch: Establece el número de puntos ejecutados simultáneamente por el modelo. Números más altos pueden ser más rápidos pero usan más memoria GPU.
            3) pred_iou_thresh: Un umbral de filtrado en [0,1], utilizando la calidad de la máscara predicha por el modelo.
            4) stability_score_thresh: Un umbral de filtrado en [0,1], utilizando la estabilidad de la máscara bajo cambios en el umbral utilizado para binarizar las predicciones de máscara del modelo.
            5) stability_score_offset: La cantidad para desplazar el umbral al calcular el puntaje de estabilidad.
            6) box_nms_thresh: El umbral de IoU de caja utilizado por la supresión no máxima para filtrar máscaras duplicadas.
            7) crop_n_layers: Si >0, la predicción de máscara se ejecutará nuevamente en recortes de la imagen. Establece el número de capas a ejecutar, donde cada capa tiene 2**i_layer número de recortes de imagen.
            8) crop_nms_thresh: El umbral de IoU de caja utilizado por la supresion no máxima para filtrar máscaras duplicadas entre diferentes recortes.
            9) crop_overlap_ratio: Establece el grado en que se superponen los recortes. En la primera capa de recorte, los recortes se superpondrán en esta fracción de la longitud de la imagen. Capas posteriores con más recortes reducen esta superposición.
            10) crop_n_points_downscale_factor: El número de puntos por lado muestreados en la capa n se reduce en crop_n_points_downscale_factor**n.
            11) point_grids: Una lista de cuadrículas de puntos explicitas utilizadas para muestreo, normalizadas a [0,1]. La enésima cuadrícula en la lista se usa en la enésima capa de recorte. Exclusivo con points_per_side.
            12) min_mask_region_area: Si >0, se aplicará un postprocesamiento para eliminar regiones desconectadas y agujeros en máscaras con área menor que min_mask_region_area.
            """
                            
            self.parametrosLabel.setToolTip(parametrosText)
            self.parametrosLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.parametrosGroupbox = QGroupBox("")
            self.parametrosLayout = QVBoxLayout()
            self.parametrosGroupbox.setLayout(self.parametrosLayout)
            
            # Agregar entradas para los parametros por defecto, los parametros son los correspondientes a imagenes con zoom
            
            self.paramsInputs = {}
            
            self.params = [
                ("points_per_side", 64),
                ("points_per_batch", 64),
                ("pred_iou_thresh", 0.88),
                ("stability_score_thresh", 0.95),
                ("stability_score_offset", 1),
                ("box_nms_thresh", 0.4),
                ("crop_n_layers", 0),
                ("crop_nms_thresh", 0.3),
                ("crop_overlap_ratio", 0.5),
                ("crop_n_points_downscale_factor", 2),
                ("min_mask_region_area", 0),
            ]
            
            for paramName, defaultValue in self.params:
                paramLayout = QHBoxLayout()
                paramLabel = QLabel(paramName)
                if isinstance(defaultValue, int):
                    inputWidget = QSpinBox()
                    inputWidget.setMaximum(1000)
                    inputWidget.setValue(defaultValue)
                else:
                    inputWidget = QDoubleSpinBox()
                    inputWidget.setMaximum(1000)
                    inputWidget.setValue(defaultValue)
                self.paramsInputs[paramName] = inputWidget
                paramLayout.addWidget(paramLabel)
                paramLayout.addWidget(inputWidget)
                self.parametrosLayout.addLayout(paramLayout)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de parametros: {str(e)}")
        
    def __inicializarProcesamiento(self) -> None:  
        """
        Inicializa la interfaz de usuario para los parametros de procesamiento.

        Este metodo configura los elementos de la interfaz de usuario para mostrar los parametros
        de procesamiento y sus valores por defecto.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.procesamientoLabel = QLabel("Parámetros de Postprocesamiento")
            self.procesamientoLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.procesamientoGroupbox = QGroupBox("")
            self.procesamientoLayout = QVBoxLayout()
            self.procesamientoGroupbox.setLayout(self.procesamientoLayout)
            
            # Seleccionar si procesamiento o no
            self.chkProcesamiento = QCheckBox("Activar postprocesamiento")
            procesamientoText = """
            Selecciona esta opción para establecer el postprocesamiento. Una vez ejecutada la segmentación automática de SAM, dará comienzo un proceso de filtrado de mascaras por los siguientes parámetros:
            
            1) min_size: Tamaño mínimo de máscara medido como "pixeles contenidos dentro de la máscara"
            2) max_size: Tamaño máximo de máscara medido como constante en la expresión:  max_size * numeroPixelesImagen[pixeles]. De esta manera es mas facil estimar el tamaño maximo de mascara que querramos. 
            Por ejemplo: Si tenemos una imagen de 800 x 600 que da un total de 480 mil pixeles y quiero que el tamaño máximo de las máscaras sean de un cuarto de la imagen tendria que establecer max_size = 0.25 lo que daría lugar a un tamaño máximo de máscara de 120 mil pixeles.
            3) min_intensity: La intensidad minima requerida para que un pixel sea considerado parte de una mascara. Ayuda a eliminar regiones que no cumplen con el umbral de intensidad mínimo
            """
            
            self.chkProcesamiento.setToolTip(procesamientoText)
            self.procesamientoLayout.addWidget(self.chkProcesamiento)
            self.chkProcesamiento.stateChanged.connect(self.__toggleProcesamiento)
            
            # Agregar parametros de post procesamiento, por defecto esta con zoom
            self.processInputs = {}
            
            self.processParams = [
                ("min_size", 100),
                ("max_size", 0.0015),
                ("min_intensity", 10),
            ]
            
            for processName, defaultValue in self.processParams:
                processLayout = QHBoxLayout()
                processLabel = QLabel(processName)
                if isinstance(defaultValue, int):
                    inputWidget = QSpinBox()
                    inputWidget.setMaximum(1000)
                    inputWidget.setValue(defaultValue)
                else:
                    inputWidget = QDoubleSpinBox()
                    inputWidget.setMaximum(1000)
                    inputWidget.setDecimals(5)
                    inputWidget.setValue(defaultValue)
                self.processInputs[processName] = inputWidget
                processLayout.addWidget(processLabel)
                processLayout.addWidget(inputWidget)
                self.procesamientoLayout.addLayout(processLayout)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de Post Procesamiento: {str(e)}")
        
    def __inicializarSegmentacion(self) -> None: 
        """
        Inicializa la interfaz de usuario para la segmentacion.

        Este metodo configura los elementos de la interfaz de usuario para la segmentacion,
        incluyendo un boton para iniciar la segmentacion, una barra de progreso y opciones de comparacion.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.segmentacionLabel = QLabel("Segmentación")
            self.segmentacionLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.segmentacionGroupbox = QGroupBox("")
            self.segmentacionLayout = QVBoxLayout()
            self.segmentacionGroupbox.setLayout(self.segmentacionLayout)
            
            # Botón para iniciar la segmentacion
            self.btnSegmentacion = QPushButton("INICIAR SEGMENTACIÓN")
            self.segmentacionLayout.addWidget(self.btnSegmentacion)
            self.btnSegmentacion.clicked.connect(self.__iniciarSegmentacionThread)
            
            # Inicializar la barra de progreso
            self.progreso = QProgressBar()
            self.segmentacionLayout.addWidget(self.progreso)
            
            # Comparacion de resultados con algun valor previo
            comparacionLayout = QHBoxLayout()
            comparacionName = "Número de Rodaballos a Comparar"  
            comparacionLabel = QLabel(comparacionName)
            comparacionLabel.setToolTip("Establece un numero de rodaballos a comparar en caso de que la imagen procesada ya haya sido etiquetada previamente. Si se establece un valor distinto de 0 se obtendrá el error absoluto y error relativo cometidos.")
            self.comparacion = QSpinBox()
            self.comparacion.setMaximum(100000)
            self.comparacion.setMinimum(0)
            self.comparacion.setValue(0)
            comparacionLayout.addWidget(comparacionLabel)
            comparacionLayout.addWidget(self.comparacion)
            self.segmentacionLayout.addLayout(comparacionLayout)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de segmentación: {str(e)}")
        
    def __inicializarCuadro(self) -> None:
        """
        Inicializa la interfaz de usuario para el cuadro de salida.

        Este metodo configura los elementos de la interfaz de usuario para mostrar logs
        y un boton para limpiar el cuadro.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.cuadroLabel = QLabel("Cuadro de salida")
            self.cuadroLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.cuadroGroupbox = QGroupBox("")
            self.cuadroLayout = QVBoxLayout()
            self.cuadroGroupbox.setLayout(self.cuadroLayout)
            
            # Cuadro de texto para mostrar logs
            self.log = QTextEdit()
            self.log.append('<div style="text-align: left; font-weight: bold; font-size: 10pt;"><u>TURBOT SAM<br></u><br></div>')
            self.cuadroLayout.addWidget(self.log)
            
            # Boton para limpiar el cuadro
            self.btnLimpiar = QPushButton("Limpiar Cuadro")
            self.cuadroLayout.addWidget(self.btnLimpiar)
            self.btnLimpiar.clicked.connect(self.__limpiarCuadro)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion del cuadro de texto: {str(e)}")
        
    def __inicializarExportaciones(self) -> None:
        """
        Inicializa la interfaz de usuario para el panel de exportaciones.

        Este metodo configura los elementos de la interfaz de usuario para exportar imagenes y datos.

        Args:
            None

        Returns:
            None
        """
        try:
            # Aniadir titulo
            self.exportacionesLabel = QLabel("Panel de Exportaciones")
            self.exportacionesLabel.setStyleSheet("font-weight: bold;")
            
            # Generar grupo
            self.exportacionesGroupbox = QGroupBox("")
            self.exportacionesLayout = QVBoxLayout()
            self.exportacionesGroupbox.setLayout(self.exportacionesLayout)
            
            # Boton para exportar la imagen con las mascaras
            self.btnExpMascaras = QPushButton("Exportar Imagen Mascaras")
            self.btnExpMascaras.setToolTip("Exporta la imagen original con las máscaras superpuestas (si se ha elegido la opcion de postprocesamiento se exportará el resultado de salida del mismo sino exportará el resultado generado por SAM)")
            self.exportacionesLayout.addWidget(self.btnExpMascaras)
            self.btnExpMascaras.clicked.connect(self.__exportarMascaras)
            
            # Boton para exportar un CSV con las mascaras
            self.btnExpMascarasCSV = QPushButton("Exportar CSV Mascaras")
            expMascarasText = """
            Exporta la información de las máscaras generadas (si se ha elegido la opcion de postprocesamiento se exportará el resultado de salida del mismo sino exportará el resultado generado por SAM). Los campos del CSV por máscara son:
            
            1) area: Número de píxeles dentro de una máscara segmentada.
            2) bbox (Bounding Box): Caja rectangular más pequeña que puede contener completamente el objeto segmentado, esta representado por las coordenadas de la caja delimitadora [Coordenada X (esquina superior izquierda), Coordenada Y (esquina superior izquierda), Ancho caja, Altura caja]
            3) predicted_iou (Predicted Intersection over Union): Es una métrica utilizada para evaluar la precisión de un algoritmo de segmentación. Se calcula como la intersección entre la máscara predicha y la máscara verdadera dividida por la unión de estas dos máscaras. El valor de IoU predicho proporciona una estimación de la calidad de la segmentación
            4) point_coords: Coordenadas específicas de los puntos en la imagen que se utilizan como referencia para la segmentación del objeto.
            5) stability_score: Es una medida de cuán estable es una máscara segmentada bajo cambios en el umbral de binarización utilizado para generar la máscara. Ayuda a evaluar la confiabilidad de una máscara segmentada. Máscaras con alta estabilidad son menos sensibles a cambios en los parámetros de segmentación, lo que las hace más robustas.
            6) crop_box: Es una caja rectangular que define una subregión de la imagen original que se ha recortado para un análisis o procesamiento adicional.
            """      
                        
            self.btnExpMascarasCSV.setToolTip(expMascarasText)
            self.exportacionesLayout.addWidget(self.btnExpMascarasCSV)
            self.btnExpMascarasCSV.clicked.connect(self.__exportarMascarasCSV)
            
            # Boton para exportar la imagen con puntos
            self.btnExpPuntos = QPushButton("Exportar Imagen Puntos")
            self.btnExpPuntos.setToolTip("Exporta la imagen original con los centroides de las máscaras superpuestos (si se ha elegido la opcion de postprocesamiento se exportará el resultado de salida del mismo sino exportará el resultado generado por SAM)")
            self.exportacionesLayout.addWidget(self.btnExpPuntos)
            self.btnExpPuntos.clicked.connect(self.__exportarPuntos)
            
            # Boton para exportar un CSV con los puntos
            self.btnExpPuntosCSV = QPushButton("Exportar CSV Puntos")
            self.btnExpPuntosCSV.setToolTip("Exporta la información de los centroides generados como coordenadas X,Y (si se ha elegido la opcion de postprocesamiento se exportará el resultado de salida del mismo sino exportará el resultado generado por SAM) ")
            self.exportacionesLayout.addWidget(self.btnExpPuntosCSV)
            self.btnExpPuntosCSV.clicked.connect(self.__exportarPuntosCSV)
            
            # Boton para exportar un histograma de las areas de las mascaras
            self.btnExpHistograma = QPushButton("Exportar Histograma")
            self.btnExpHistograma.setToolTip("Exporta un histograma de las areas (ordenadas de mayor a menor) de las mascaras procesadas junto con sus stability scores. Esta opción sólo esta disponible si se ha marcado la opción de postprocesamiento")
            self.exportacionesLayout.addWidget(self.btnExpHistograma)
            self.btnExpHistograma.clicked.connect(self.__exportarHistograma)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de exportaciones: {str(e)}")
        
    def __incicializarVariables(self) -> None:
        """
        Inicializa las variables de la clase.

        Este metodo inicializa todas las variables de la clase utilizadas para almacenar datos y configuraciones.

        Args:
            None

        Returns:
            None
        """
        try:
            self.imagenCargada = None              # Variable para almacenar la imagen
            self.numCuadrantes = None              # Variable para almacenar el numero de cuadrantes seleccionado
            self.mostrarSectores = False           # Variable para almacenar la opción de sectores
            self.imagenZoom = True                 # Variable para almacenar la seleccion de zoom
            self.mascarasGeneradas = None          # Variable para almacenar las mascaras generadas
            self.mascarasGeneradasAux = None       # Variable para almacenar las mascaras generadas como copia para exportarlo 
            self.puntosGenerados = None            # Variable para almacenar los puntos generados
            self.puntosGeneradosAux = None         # Variable para almacenar los puntos generados como copia para exportarlos
            self.mascarasProcesadas = None         # Variable para almacenar las mascaras generadas
            self.mascarasProcesadasAux = None      # Variable para almacenar las mascaras procesadas como copia para exportarlo
            self.puntosProcesados = None           # Variable para almacenar los puntos procesados
            self.puntosProcesadosAux = None        # Variable para almacenar los puntos procesados como copia para exportarlos
            self.porcentajeProgreso = 0            # Variable para almacenar el porcentaje de progreso de la segmentacion
            self.startTime = None                  # Variable para almacenar el tiempo transcurrido
            self.procesamiento = False             # Variable para almacenar la seleccion de postprocesamiento
            self.listaPuntosGenerados = None       # Variable que almacena la lista de las posiciones de los puntos de las mascaras generadas
            self.listaPuntosProcesados = None      # Variable que almacena la lista de las posiciones de los puntos de las mascaras procesadas
            self.listaMascaras = None              # Variable que almacena la lista con informacion de las mascaras generadas
            self.listaMascarasProcesadas = None    # Variable que almacena la lista con informacion de las mascaras procesadas
            
            # Crear un temporizador para llamar a __cargarMascaras periodicamente
            self.timer = QTimer()
            self.timer.timeout.connect(self.__cargarMascaras)
            self.timer.start(1000)                 # Llama a "__cargarMascaras" cada 1000 milisegundos (1 segundo)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error durante el proceso de inicializacion de variables: {str(e)}")
        
    # Funciones para el proceso de carga
    
    def __cargarImagen(self, filename: str) -> None:
        """
        Carga una imagen desde el archivo especificado.

        Esta funcion carga una imagen desde el archivo especificado y realiza algunas
        operaciones como convertir la imagen a escala de grises y abrir la imagen en un visor.

        Args:
            filename (str): La ruta del archivo de la imagen.

        Returns:
            None
        """
        try:
            self.imagenCargada = cv2.imread(filename)
            self.dimensionesImagenCargada = Utils.obtenerDimensionesImagen(self.imagenCargada)
            self.imagenGrises = Utils.convertRGB(self.imagenCargada)
            self.viewer.open(filename)
            self.log.append("<span style='color: green;'>[INFO]</span> Imagen cargada")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al cargar la imagen: {str(e)}")

    def __cargarImagenWidget(self) -> None:
        """
        Abre un cuadro de dialogo para seleccionar una imagen y la carga.

        Esta funcion abre un cuadro de dialogo para que el usuario seleccione una imagen
        y luego llama al metodo __cargarImagen para cargar la imagen seleccionada.

        Args:
            None

        Returns:
            None
        """
        try:
            filename, _ = QFileDialog.getOpenFileName(
                None, "Seleccionar imagen", "", "Imagen (*.tif *.png *.jpg)"
            )
            if filename:
                self.__cargarImagen(filename)  
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error abrir el cuadro de diálogo de cargar imagen: {str(e)}")
            
    def __getImagenCargada(self) -> Union[np.ndarray, None]:
        """
        Obtiene la imagen cargada.

        Retorna la imagen que ha sido cargada en la aplicacion.

        Args:
            None

        Returns:
            self.imagenCargada: La imagen cargada.
        """
        return self.imagenCargada
    
    # Funciones para el proceso de seleccion con/sin zoom
    
    def __toggleZoom(self, state: int) -> None:
        """
        Cambia el estado de la opcion de zoom de la imagen.

        Esta funcion se activa cuando se cambia el estado de la opcion de zoom.
        Actualiza las configuraciones y muestra mensajes de informacion.

        Args:
            state (int): El estado de la opcion de zoom.

        Returns:
            None
        """
        try:
            if state == 2:  
                self.imagenZoom = True
                self.chkNoZoom.setChecked(False)
                self.log.append("<span style='color: green;'>[INFO]</span> Imagen CON zoom seleccionada")
                
                self.params = [
                ("points_per_side", 64),
                ("points_per_batch", 64),
                ("pred_iou_thresh", 0.88),
                ("stability_score_thresh", 0.95),
                ("stability_score_offset", 1),
                ("box_nms_thresh", 0.2),
                ("crop_n_layers", 0),
                ("crop_nms_thresh", 0.3),
                ("crop_overlap_ratio", 0.5),
                ("crop_n_points_downscale_factor", 2),
                ("min_mask_region_area", 0),
                ]
            
                self.__actualizarParametros(self.params)
                self.log.append("<span style='color: green;'>[INFO]</span> Parametros de segmentación actualizados")
                
                self.processParams = [
                ("min_size", 100),
                ("max_size", 0.0015),
                ("min_intensity", 10),
                ]
                
                self.__actualizarParametrosProcesamiento(self.processParams)
                self.log.append("<span style='color: green;'>[INFO]</span> Parametros de postprocesamiento actualizados")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al cambiar el estado a CON Zoom: {str(e)}")

    def __toggleNoZoom(self, state: int) -> None:
        """
        Cambia el estado de la opcion de zoom de la imagen a Sin zoom.

        Esta funcion se activa cuando se cambia el estado de la opcion de zoom a Sin zoom.
        Actualiza las configuraciones y muestra mensajes de informacion.

        Args:
            state (int): El estado de la opción de zoom.

        Returns:
            None
        """
        try:
            if state == 2:  
                self.imagenZoom = False
                self.chkZoom.setChecked(False)
                self.log.append("<span style='color: green;'>[INFO]</span> Imagen SIN zoom seleccionada")
                
                self.params = [
                ("points_per_side", 64),
                ("points_per_batch", 64),
                ("pred_iou_thresh", 0.88),
                ("stability_score_thresh", 0.95),
                ("stability_score_offset", 1),
                ("box_nms_thresh", 0.2),
                ("crop_n_layers", 0),
                ("crop_nms_thresh", 0.3),
                ("crop_overlap_ratio", 0.5),
                ("crop_n_points_downscale_factor", 2),
                ("min_mask_region_area", 0),
                ]
            
                self.__actualizarParametros(self.params)
                self.log.append("<span style='color: green;'>[INFO]</span> Parametros de segmentación actualizados")
                
                self.processParams = [
                ("min_size", 50),
                ("max_size", 0.0005),
                ("min_intensity", 10),
                ]
                
                self.__actualizarParametrosProcesamiento(self.processParams)
                self.log.append("<span style='color: green;'>[INFO]</span> Parametros de postprocesamiento actualizados") 
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al cambiar el estado a SIN Zoom: {str(e)}")
    
    # Funciones para el proceso de seleccion de cuadrantes
    
    def __actualizarSectores(self, state: int) -> None:
        """
        Actualiza el estado de visualizacion de sectores.

        Esta funcion se activa cuando se cambia el estado de visualizacion de sectores.
        Actualiza la variable mostrarSectores y muestra mensajes de informacion.

        Args:
            state (int): El estado de la opcion de visualizacion de sectores.

        Returns:
            None
        """
        try:
            if state == 2:
                self.mostrarSectores = True
                self.log.append("<span style='color: green;'>[INFO]</span> Visualización de sectores activada")
            else:
                self.mostrarSectores = False
                self.log.append("<span style='color: green;'>[INFO]</span> Visualización de sectores desactivada")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al cambiar el estado de la opción de sectores: {str(e)}")
        
    def __actualizarCuadrantes(self, index: int) -> None:
        """
        Actualiza el numero de cuadrantes y los muestra en la imagen si esta cargada.

        Esta funcion se activa cuando se selecciona un indice en el ComboBox de cuadrantes.
        Actualiza el numero de cuadrantes y los muestra en la imagen si esta cargada.

        Args:
            index (int): El indice seleccionado en el ComboBox de cuadrantes.

        Returns:
            None
        """
        try:
            if self.imagenCargada is not None:
                self.numCuadrantes = int(self.cuadrantesCombo.currentText()) if index > 0 else None
                if self.numCuadrantes is not None:
                    labelsCuadrantes  = Utils.generarCuadrantes(self.dimensionesImagenCargada, self.numCuadrantes, self.mostrarSectores)
                    self.__agregarLabel(labelsCuadrantes, "Cuadrantes " + str(self.numCuadrantes))
                    self.log.append("<span style='color: green;'>[INFO]</span> Se han añadido " + str(self.numCuadrantes) + " cuadrantes")
                else:
                    self.log.append("<span style='color: green;'>[INFO]</span> Se ha seleccionado la opción 'Sin Cuadrantes'")
                    
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, cargue una imagen antes de seleccionar los cuadrantes")  
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al actualizar los cuadrantes: {str(e)}")
            
    # Funciones para el proceso de seleccion de parametros de segmentacion 
    
    def __actualizarParametros(self, parametros: List[Tuple[str, Union[int, float]]]) -> None:
        """
        Actualiza los parametros en la interfaz grafica.

        Esta funcion actualiza los valores de los widgets de entrada de parametros
        con los valores proporcionados en la lista de parametros.

        Args:
            parametros (list): Lista de tuplas (nombre del parametro, valor).

        Returns:
            None
        """
        try:
            for paramName, defaultValue in parametros:
                inputWidget = self.paramsInputs[paramName]
                if isinstance(defaultValue, int):
                    inputWidget.setValue(defaultValue)
                else:
                    inputWidget.setValue(defaultValue)     
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al actualizar los parametros: {str(e)}")
                
    # Funciones para el proceso de seleccion de procesamiento
    
    def __toggleProcesamiento(self, state: int) -> None:
        """
        Activa o desactiva la funcion de postprocesamiento.

        Esta funcion se activa cuando se cambia el estado de la opcion de procesamiento.
        Actualiza la variable de procesamiento y muestra mensajes de información.

        Args:
            state (int): El estado de la opción de procesamiento.

        Returns:
            None
        """
        try:
            if state == 2:  
                self.procesamiento = True
                self.log.append("<span style='color: green;'>[INFO]</span> Función de postprocesamiento activada")
            else:
                self.procesamiento = False
                self.log.append("<span style='color: green;'>[INFO]</span> Función de postprocesamiento desactivada")    
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al cambiar el estado de la opción de Post Procesamiento: {str(e)}")
    
    def __actualizarParametrosProcesamiento(self, parametros: List[Tuple[str, Union[int, float]]]) -> None:
        """
        Actualiza los parametros de postprocesamiento en la interfaz grafica.

        Esta funcion actualiza los valores de los widgets de entrada de parametros
        de postprocesamiento con los valores proporcionados en la lista de parametros.

        Args:
            parametros (list): Lista de tuplas (nombre del parametro, valor).

        Returns:
            None
        """
        try:
            for processName, defaultValue in parametros:
                inputWidget = self.processInputs[processName]
                if isinstance(defaultValue, int):
                    inputWidget.setValue(defaultValue)
                else:
                    inputWidget.setValue(defaultValue)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al actualizar los parametros de Post Procesamiento: {str(e)}")
                
    # Funciones para el proceso de cuadro de texto
    
    def __limpiarCuadro(self) -> None:
        """
        Limpia el cuadro de registros.

        Esta funcion borra todos los registros del cuadro de registros y anade un encabezado.

        Args:
            None

        Returns:
            None
        """
        try:
            self.log.clear()
            self.log.append('<div style="text-align: left; font-weight: bold; font-size: 10pt;"><u>TURBOT SAM<br></u><br></div>')
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al limpiar el cuadro de texto: {str(e)}")
        
    def __mostrarResultados(self) -> None:
        """
        Muestra los resultados de la comparacion de rodaballos.

        Esta funcion muestra los resultados de la comparacion de rodaballos, incluyendo el numero
        estimado de rodaballos y los errores absoluto y relativo si se ha establecido un numero de comparacion.

        Args:
            None

        Returns:
            None
        """
        try:
            self.numeroComparacion = self.comparacion.value()
            
            if self.numeroComparacion == 0:
                self.log.append("<span style='color: green;'>[INFO]</span> No se ha establecido ningún número de rodaballos a comparar")
                self.log.append("<span style='color: green;'>[INFO]</span> El numero estimado de rodaballos calculados es: " + str(self.numeroRodCalculado) + "<br>")
            else:
                self.log.append("<span style='color: green;'>[INFO]</span> Se ha establecido la comparación de rodaballos")
                self.errorAbsoluto, self.errorRelativo = Utils.calcularErrores(self.numeroComparacion,self.numeroRodCalculado)
                self.log.append("<span style='color: green;'>[INFO]</span> El numero de rodaballos a comparar es: " + str(self.numeroComparacion))
                self.log.append("<span style='color: green;'>[INFO]</span> El numero estimado de rodaballos calculados es: " + str(self.numeroRodCalculado))
                self.log.append("<span style='color: green;'>[INFO]</span> El error absoluto cometido es de " + str(self.errorAbsoluto) + " rodaballos")
                self.log.append("<span style='color: green;'>[INFO]</span> El error relativo cometido es de " + str(self.errorRelativo) + "%<br>")    
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al mostrar información en el cuadro de texto: {str(e)}")
            
    # Funciones para el proceso de exportaciones
    
    def __exportarMascaras(self) -> None:
        """
        Exporta la imagen con las mascaras generadas o procesadas.

        Esta funcion exporta la imagen con las mascaras generadas o procesadas, dependiendo del estado
        de la opcion de procesamiento. Abre un dialogo de archivo para seleccionar la ubicacion de guardado.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.mascarasProcesadasAux is not None or self.mascarasGeneradasAux is not None:
                if self.procesamiento == True:
                    self.imagenMascaras = label2rgb(self.mascarasProcesadasAux, self.imagenCargada, alpha=0.2)
                else:
                    self.imagenMascaras = label2rgb(self.mascarasGeneradasAux, self.imagenCargada, alpha=0.2)
                    
                # Abrir el diálogo de archivo para seleccionar la ubicación de guardado
                opciones, _ = QFileDialog.getSaveFileName(None, "Guardar Imagen", "", "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)")
                
                if opciones:
                    
                    # Guardar la imagen resultante en la ruta seleccionada
                    plt.imsave(opciones, self.imagenMascaras)
                    self.log.append("<span style='color: green;'>[INFO]</span> Imagen con mascaras alamacenada correctamente.")
                    
                else:
                    self.log.append("<span style='color: yellow;'>[WARNING]</span> No se seleccionó ninguna ubicación para guardar la imagen.")
                
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, inicie un proceso de segmentación antes de exportar los resultados")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al exportar la imagen con las máscaras: {str(e)}")
            
    def __exportarPuntos (self) -> None:
        """
        Exporta la imagen con los puntos generados o procesados.

        Esta funcion exporta la imagen con los puntos generados o procesados, dependiendo del estado
        de la opcion de procesamiento. Abre un dialogo de archivo para seleccionar la ubicacion de guardado.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.puntosProcesadosAux is not None or self.puntosGeneradosAux is not None:
                
                if self.procesamiento == True:
                    self.imagenPuntos = label2rgb(self.puntosProcesadosAux,image = self.imagenCargada, alpha = 0.4)
                else:
                    self.imagenPuntos = label2rgb(self.puntosGeneradosAux, image = self.imagenCargada, alpha = 0.4)
                    
                # Abrir el diálogo de archivo para seleccionar la ubicación de guardado
                opciones, _ = QFileDialog.getSaveFileName(None, "Guardar Imagen", "", "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)")
                
                if opciones:
                    
                    # Guardar la imagen resultante en la ruta seleccionada
                    plt.imsave(opciones, self.imagenPuntos)
                    self.log.append("<span style='color: green;'>[INFO]</span> Imagen con puntos alamacenada correctamente.")
                    
                else:
                    self.log.append("<span style='color: yellow;'>[WARNING]</span> No se seleccionó ninguna ubicación para guardar la imagen.")
                    
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, inicie un proceso de segmentación antes de exportar los resultados")
                
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al exportar la imagen con los puntos: {str(e)}")
            
    def __exportarPuntosCSV(self) -> None:
        """
        Exporta los puntos generados o procesados a un archivo CSV.

        Esta funcion exporta los puntos generados o procesados a un archivo CSV.
        Abre un cuadro de dialogo para seleccionar el directorio y el nombre del archivo a almacenar.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.puntosProcesadosAux is not None or self.puntosGeneradosAux is not None:
                
                listaPuntos = self.listaPuntosProcesados if self.procesamiento else self.listaPuntosGenerados

                # Abre un cuadro de diálogo para seleccionar el directorio y el nombre del archivo
                opciones, _ = QFileDialog.getSaveFileName(None, 'Guardar archivo CSV', '', 'Archivos CSV (*.csv)')
                
                # Si se selecciona un nombre de archivo, escribe los puntos en el archivo CSV
                if opciones:

                    # Escribe los puntos en el archivo CSV
                    with open(opciones, 'w', newline='') as csvfile:
                        # Definir el escritor CSV
                        writer = csv.writer(csvfile)
                        
                        # Escribir la cabecera
                        writer.writerow(['x', 'y'])
                        
                        # Escribir cada conjunto de coordenadas en una fila
                        for punto in listaPuntos:
                            writer.writerow(punto)
                            
                    self.log.append("<span style='color: green;'>[INFO]</span> CSV con puntos alamacenado correctamente.")
                    
                else:
                    self.log.append("<span style='color: yellow;'>[WARNING]</span> No se seleccionó ninguna ubicación para guardar el CSV.")
                    
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, inicie un proceso de segmentación antes de exportar los resultados")    
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al exportar el archivo CSV con los puntos: {str(e)}")
                
    def __exportarMascarasCSV(self) -> None:
        """
        Exporta las mascaras generadas o procesadas a un archivo CSV.

        Esta funcion exporta las mascaras generadas o procesadas a un archivo CSV.
        Abre un cuadro de dialogo para seleccionar el directorio y el nombre del archivo a almacenar.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.puntosProcesadosAux is not None or self.puntosGeneradosAux is not None:
                
                listaMascaras = self.listaMascarasProcesadas if self.procesamiento else self.listaMascaras

                # Abre un cuadro de diálogo para seleccionar el directorio y el nombre del archivo
                opciones, _ = QFileDialog.getSaveFileName(None, 'Guardar archivo CSV', '', 'Archivos CSV (*.csv)')
                
                # Si se selecciona un nombre de archivo, escribe los puntos en el archivo CSV
                if opciones:

                    # Campos del encabezado del archivo CSV
                    encabezado = ['area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box']

                    # Escribir los datos en el archivo CSV
                    with open(opciones, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=encabezado, quoting = csv.QUOTE_NONE, escapechar='\\')
                        writer.writeheader()
                        for objeto in listaMascaras:
                            # Eliminar el campo 'segmentation' del objeto
                            objetoSinSegmentation = {key: val for key, val in objeto.items() if key != 'segmentation'}
                            # Convertir listas en strings
                            objetoStr = {key: str(val).replace('\\', '') if isinstance(val, str) else val for key, val in objetoSinSegmentation.items()}
                            writer.writerow(objetoStr)
                            
                    self.log.append("<span style='color: green;'>[INFO]</span> CSV con mascaras alamacenado correctamente.")
                    
                else:
                    self.log.append("<span style='color: yellow;'>[WARNING]</span> No se seleccionó ninguna ubicación para guardar el CSV.")
                    
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, inicie un proceso de segmentación antes de exportar los resultados")
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al exportar el archivo CSV con la información de las máscaras: {str(e)}")
                
    def __exportarHistograma(self) -> None:
        """
        Exporta un histograma de las areas de las mascaras procesadas junto con sus stability scores.

        Esta funcion exporta un histograma de las areas de las mascaras procesadas junto con sus stability scores.
        Abre un cuadro de dialogo para seleccionar la ubicacion y el nombre del archivo a almacenar.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.puntosProcesadosAux is not None:
                
                listaMascaras = self.listaMascarasProcesadas

                # Extraer el área y predicted_iou de cada objeto
                listaAreas = [mascara['area'] for mascara in listaMascaras]
                listaPredictedIou = [mascara['predicted_iou'] for mascara in listaMascaras]

                # Normalizar los predicted_iou para que estén en el rango [0, 1]
                maxPredictedIou= max(listaPredictedIou)
                listaPredictedIouNorm = [score / maxPredictedIou for score in listaPredictedIou]

                # Ordenar las áreas de manera descendente
                listaAreas = sorted(listaAreas, reverse=True)

                # Generar etiquetas para el eje x
                etiquetasX = [f"M{i+1}" for i in range(len(listaAreas))]
                
                # Abre un cuadro de diálogo para seleccionar el directorio y el nombre del archivo
                opciones, _ = QFileDialog.getSaveFileName(None, 'Guardar Histograma', '', 'Archivos de imagen (*.png)')
                
                # Si se selecciona un nombre de archivo, escribe los puntos en el archivo CSV
                if opciones:

                    # Calcular la línea de tendencia
                    x = np.arange(len(listaAreas))
                    y = np.array(listaAreas)
                    m, b = np.polyfit(x, y, 1)  # Coeficientes de la línea de tendencia
                    lineaTendencia = m * x + b
                    
                    
                    # Generar el histograma y guardarlo como imagen en la ubicación seleccionada
                    fig, ax1 = plt.subplots(figsize=(8, 6))
                    ax1.bar(etiquetasX, listaAreas, color='blue', edgecolor='blue')
                    ax1.set_xlabel('Máscaras')
                    ax1.set_ylabel('Área [Pixeles Contenidos]')
                    ax1.set_title('Histograma de áreas de máscaras')  # Título en negrita
                    ax1.set_xticks([])  

                    # Trazar la línea de tendencia
                    ax1.plot(x, lineaTendencia, color='red', linestyle='-')
                    ax1.grid(True)

                    if self.numeroRodCalculado < 100:
                        tamanoPunto = 4
                        linea = '--'
                    else:
                        tamanoPunto = 1
                        linea = ''
                        
                    # Agregar el segundo eje Y
                    ax2 = ax1.twinx()  
                    ax2.set_ylabel('Predicted IoU [Normalizado]')
                    ax2.plot(x, listaPredictedIouNorm, color='green', linestyle = linea, marker='o', markersize= tamanoPunto, linewidth=0.5)

                    fig.tight_layout()
                    plt.savefig(opciones)
                    
                    
                    plt.tight_layout()
                    plt.savefig(opciones)
                            
                    self.log.append("<span style='color: green;'>[INFO]</span> Histograma alamacenado correctamente.")
                    
                else:
                    self.log.append("<span style='color: yellow;'>[WARNING]</span> No se seleccionó ninguna ubicación para guardar el CSV.")
                    
            else:
                self.log.append("<span style='color: yellow;'>[WARNING]</span> Necesitas realizar un postprocesamiento para generar el histograma")    
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al exportar el histrograma: {str(e)}")
    
    # Funciones para el proceso de segmentacion
    
    def __iniciarSegmentacionThread(self) -> None:
        """
        Inicia la segmentacion en un hilo separado.

        Esta funcion inicia el proceso de segmentacion en un hilo separado para evitar bloquear la interfaz
        de usuario mientras se realiza la segmentacion.

        Args:
            None

        Returns:
            None
        """
        try:
            thread = threading.Thread(target=self.__iniciarSegmentacion)
            thread.start()
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar un hilo: {str(e)}")
        
    def __actualizarBarraProgreso(self, porcentaje: float) -> None:
        """
        Actualiza el valor de la barra de progreso.

        Esta función actualiza el valor de la barra de progreso con el porcentaje proporcionado.

        Args:
            porcentaje (float): El porcentaje de progreso.

        Returns:
            None
        """
        try:
            self.progreso.setValue(round(porcentaje))   
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al establecer el valor del porcentaje de progreso: {str(e)}")
        
    def __iniciarSegmentacion(self) -> None:
        """
        Inicia el proceso de segmentacion.

        Esta funcion inicia el proceso de segmentacion de SAM, comprueba si se ha seleccionado la opcion de
        postprocesamiento y lo ejecuta en tal caso, mostrando los resultados pertinentes.

        Args:
            None

        Returns:
            None
        """
        if self.imagenCargada is not None:
            try:
                # Actualizo la barra de estado
                self.porcentajeProgreso = 0
                self.log.append(f"<br><span style='color: green;'>[INFO]</span> Procesamiento Activado. Progreso: {self.porcentajeProgreso:.2f}%")
                
                
                # Obtener los parametros seleccionados
                points_per_side = self.paramsInputs["points_per_side"].value()
                points_per_batch = self.paramsInputs["points_per_batch"].value()
                pred_iou_thresh = self.paramsInputs["pred_iou_thresh"].value()
                stability_score_thresh = self.paramsInputs["stability_score_thresh"].value()
                stability_score_offset = self.paramsInputs["stability_score_offset"].value()
                box_nms_thresh = self.paramsInputs["box_nms_thresh"].value()
                crop_n_layers = self.paramsInputs["crop_n_layers"].value()
                crop_nms_thresh = self.paramsInputs["crop_nms_thresh"].value()
                crop_overlap_ratio = self.paramsInputs["crop_overlap_ratio"].value()
                crop_n_points_downscale_factor = self.paramsInputs["crop_n_points_downscale_factor"].value()
                min_mask_region_area = self.paramsInputs["min_mask_region_area"].value()
                
                # Crear una instancia de TurbotSAM con los parámetros seleccionados
                turbotSam = TurbotSAM(
                    points_per_side=points_per_side,
                    points_per_batch=points_per_batch,
                    pred_iou_thresh=pred_iou_thresh,
                    stability_score_thresh=stability_score_thresh,
                    stability_score_offset=stability_score_offset,
                    box_nms_thresh=box_nms_thresh,
                    crop_n_layers=crop_n_layers,
                    crop_nms_thresh=crop_nms_thresh,
                    crop_overlap_ratio=crop_overlap_ratio,
                    crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                    min_mask_region_area=min_mask_region_area
                )
                
                # Compruebo si se han seleccionado cuadrantes
                indice = self.cuadrantesCombo.currentIndex()
                
                if indice == 0:
                    try:
                        mascaras = turbotSam.generarMascaras(self.imagenGrises)
                        
                        self.listaMascaras = mascaras
                        
                        self.porcentajeProgreso = 90 if not self.procesamiento else 50
                        self.log.append(f"<span style='color: green;'>[INFO]</span> Mascaras generadas correctamente. Progreso: {self.porcentajeProgreso:.2f}%")
                        
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar las máscaras para la imagen sin cuadrantes: {str(e)}")
                    
                else:
                    
                    # Generamos los cuadrantes
                    try:
                        cuadrantes = Utils.recortarCuadrantes(self.imagenGrises, self.numCuadrantes)
                        
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al recortar los cuadrantes de la imagen para procesarlos: {str(e)}")
                
                    # Recuperamos la lista con las mascaras y el porcentaje de progreso en cada iteracion
                    try:
                        for porcentaje, mascarasPorCuadrante, cuadranteProcesado in turbotSam.generarMascarasPorCuadrante(cuadrantes, self.procesamiento):
                            self.porcentajeProgreso = porcentaje
                            self.log.append(f"<span style='color: green;'>[INFO]</span> Mascaras generadas para el Cuadrante {cuadranteProcesado}. Progreso: {self.porcentajeProgreso:.2f}%")
                            
                        self.log.append(f"<span style='color: green;'>[INFO]</span> Mascaras generadas correctamente para todos los cuadrantes") 
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar las máscaras para la imagen con cuadrantes: {str(e)}")
                    
                    # Generar la imagen con todas las máscaras superpuestas por cuadrantes
                    try:
                        mascaras = ProcesarMascaras.superponerMascaras(mascarasPorCuadrante, self.dimensionesImagenCargada)
                        self.listaMascaras = mascaras
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al superponer las mascaras de los cuadrantes en una misma imagen: {str(e)}")
                try:
                    samImg = ProcesarMascaras.mostrarLabels(mascaras)
                    self.mascarasGeneradas = samImg
                    self.mascarasGeneradasAux = samImg
                except Exception as e:
                    self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar la imagen de etiquetas para las máscaras generadas por SAM: {str(e)}")
                
                
                # Genero los puntos de lo que genera SAM
                try:
                    self.log.append(f"<span style='color: green;'>[INFO]</span> Generando centroides para las mascaras generadas. Progreso: {self.porcentajeProgreso:.2f}%")
                    labels_points_sam, self.listaPuntosGenerados = ProcesarMascaras.pintarCentroidesMascaras(mascaras)
                    self.puntosGenerados = labels_points_sam
                    self.puntosGeneradosAux = labels_points_sam
                    
                    self.porcentajeProgreso = 100 if not self.procesamiento else 60
                    self.log.append(f"<span style='color: green;'>[INFO]</span> Centroides de las mascaras generados correctamente. Progreso: {self.porcentajeProgreso:.2f}%")
                except Exception as e:
                    self.log.append(f"<span style='color: red;'>[ERROR]</span>  Ha ocurrido un error al generar los centroides de las máscaras generadas por SAM: {str(e)}")
                
                # Se inicia el post procesamiento si es necesario
                if self.procesamiento == True:
                    try:
                        self.log.append(f"<br><span style='color: green;'>[INFO]</span> Iniciando Post Procesamiento. Progreso: {self.porcentajeProgreso:.2f}%")
                        
                        # Obtener los parametros de postprocesamiento
                        min_size = self.processInputs["min_size"].value()
                        max_size = self.processInputs["max_size"].value()
                        min_intensity = self.processInputs["min_intensity"].value()
                    
                        # Calcular la imagen promediada en escala de grises a partir de los resultados. Obtenemos la imagen con las mascaras filtrada
                        temp = np.mean(self.imagenGrises, axis=2)
                        
                        #Ejecutamos la función de post procesamiento
                        labelsProcesados = np.zeros(self.imagenGrises.shape[:2], dtype=np.uint16)
                        labelsProcesados, mascarasProcesadas = ProcesarMascaras.procesarMascaras(labelsProcesados, mascaras, temp, min_size, max_size*self.imagenGrises.shape[0]*self.imagenGrises.shape[1], min_intensity)
                        self.listaMascarasProcesadas = mascarasProcesadas
                        self.porcentajeProgreso = 90
                        self.log.append(f"<span style='color: green;'>[INFO]</span> Mascaras de postprocesamiento filtradas. Progreso: {self.porcentajeProgreso:.2f}%")
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al procesar las mascaras en las funcion de post procesamiento: {str(e)}")
                    
                    #Obtenemos las mascaras y el numero
                    try:
                        self.numeroRodCalculado = len(mascarasProcesadas)
                        mascarasImg = ProcesarMascaras.mostrarLabels(mascarasProcesadas)
                        self.mascarasProcesadas = mascarasImg
                        self.mascarasProcesadasAux = mascarasImg
                    except MemoryError as e:
                        self.log.append(f"<span style='color: yellow;'>[WARNING]</span> Debido a la cantidad de máscaras no se pudo asignar memoria suficiente para generar la imagen de segmentación de máscaras. Se generarán sólo los centros de máscaras: {str(e)}")
                        
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar la imagen de etiquetas para las máscaras procesadas: {str(e)}")
                    
                    # Obtenemos los puntos
                    try:
                        self.log.append(f"<span style='color: green;'>[INFO]</span> Generando centroides para las mascaras procesadas. Progreso: {self.porcentajeProgreso:.2f}%")
                        labelsPuntosProcesados, self.listaPuntosProcesados = ProcesarMascaras.pintarCentroidesMascaras(mascarasProcesadas)
                        self.puntosProcesados = labelsPuntosProcesados
                        self.puntosProcesadosAux = labelsPuntosProcesados
                        self.porcentajeProgreso = 100
                        self.log.append(f"<span style='color: green;'>[INFO]</span> Centroides de las mascaras postprocesadas establecidos. Progreso: {self.porcentajeProgreso:.2f}%")
                    except Exception as e:
                        self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al generar los centroides de las máscaras procesadas: {str(e)}")
                else:
                    self.numeroRodCalculado = len(mascaras)
                    
                self.log.append(f"<span style='color: green;'>[INFO]</span> Procesamiento de Segmentación finalizado.<br>")
                
                self.__mostrarResultados()
            except Exception as e:
                self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error en el proceso de segmentación: {str(e)}")
        else:
            self.log.append("<span style='color: yellow;'>[WARNING]</span> Por favor, cargue una imagen antes de iniciar el proceso de segmentación.")  
               
    # Funciones de procesado
    
    def __cargarMascaras(self) -> None:
        """
        Carga las mascaras generadas o los puntos segun la disponibilidad.

        Esta funcion carga las mascaras generadas por SAM o los puntos generados segun su disponibilidad
        en la interfaz grafica. Ademas se encarga de comprobar y actualizar la barra de progreso

        Args:
            None

        Returns:
            None
        """
        try:
            if self.mascarasGeneradas is not None:
                self.__agregarLabel(self.mascarasGeneradas, "Mascaras SAM")
                self.mascarasGeneradas  = None
            
            elif self.puntosGenerados is not None:
                self.__agregarLabel(self.puntosGenerados, "Centros de Mascaras SAM")
                self.puntosGenerados  = None
                
            elif self.mascarasProcesadas is not None:
                self.__agregarLabel(self.mascarasProcesadas, "Mascaras Post Procesamiento")
                self.mascarasProcesadas  = None
                
            elif self.puntosProcesados is not None:
                self.__agregarLabel(self.puntosProcesados, "Centros de Mascaras Post Procesamiento")
                self.puntosProcesados  = None

            else:
                pass
            
            self.__actualizarBarraProgreso(self.porcentajeProgreso)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al agregar las mascaras o puntos generados/procesados al visor: {str(e)}")
            
    def __agregarImagen(self, imagen: Union[np.ndarray, None], titulo: str) -> None:
        """
        Agrega una imagen al visor de Napari con el titulo especificado.

        Args:
            imagen: La imagen a agregar.
            titulo: El título de la imagen que se mostrará en el visor.
        """
        try:
            self.viewer.add_image(imagen, name=titulo)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al añadir la imagen al visor: {str(e)}")
        
    def __agregarLabel(self, label: Union[np.ndarray, None] , titulo: str) -> None:
        """
        Agrega un label al visor de Napari con el título especificado.

        Args:
            label: El label a agregar.
            titulo: El título del label que se mostrara en el visor.
        """
        try:
            self.viewer.add_labels(label, name=titulo)
        except Exception as e:
            self.log.append(f"<span style='color: red;'>[ERROR]</span> Ha ocurrido un error al añadir la imagen de etiquetas: {str(e)}")

    # Funcion de arranque
    
    def run(self) -> None:
        """
        Ejecuta la aplicacion Napari.

        Esta funcion ejecuta la aplicacion Napari para mostrar la interfaz grafica.

        Args:
            None

        Returns:
            None
        """
        napari.run()