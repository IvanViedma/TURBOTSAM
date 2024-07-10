**TURBOT SAM**  
1) Clonamos el repositorio
2) Generamos un entorno virtual con Python 3.8
3) Instalamos las dependencias incluidas en el archivo requirements
4) En el caso de que nuestro PC tenga tarjeta gráfica NVIDEA debemos seguir este paso sino saltamos al siguiente. Para poder usar CUDA, debemos instalar el siguiente comando dependiendo de nuestra  version de CUDA (Si no se dispone de CUDA se empleará la CPU para el procesamiento lo cual supondrá tiempos de ejecución muchos más largos):
   a) CUDA 12.1: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   b) CUDA 11.8: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
5) Ejecutar pip intsall timm
6) Ejecutamos main.py y se abrira la consola con la aplicación TURBOTSAM 
