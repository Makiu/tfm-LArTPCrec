# tfm-LArTPCrec
Repository containing the files used for my Master's Thesis "Reconstrucción automática de imágenes tridimensionales en LArTPCs de gran tamaño" (Automatic reconstruction of three-dimensional images in large LArTPCs)

Contenido de las carpetas:
* files: Archivos root con los sucesos. Deben tener el mismo formato de nombre que los archivos que hay de prueba.
* scripts: Códigos y archivos auxiliares.
* results: Almacena los resultados.

Descripción del código:
El código de este trabajo está formado por tres scripts principales: 
* event.py: Contiene el Algoritmo de Referencia y las herramientas generales de reconstrucción (eliminación de ruido, evaluación, etc)
* cercania.py: Contiene el Algoritmo de Cercanía
* adhoc.py: Contiene el Algoritmo Ad-hoc

Estos dos últimos scripts hacen uso del primero. Al comienzo de estos archivos se encuentra un espacio para controlar que sucesos se reconstruyen, si se almacenan y/o se muestran sus resultados, y establecer los valores de los parámetros de los algoritmos.

Junto a estos tres scripts se encuentran otros archivos secundarios:
* funciones.py: Almacena algunas de las funciones empleadas en los scripts principales
* param.py: Parámetros del detector
* mapping.csv: Contiene la información de la geometría del detector
