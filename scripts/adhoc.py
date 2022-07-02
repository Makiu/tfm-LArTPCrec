# -*- coding: utf-8 -*-
"""Algoritmo ad-hoc

Este script permite al usuario aplicar el algoritmo de referencia a los
sucesos almacenados en archivos root.

Contiene las herramientas necesarias para el preprocesamiento y el
análisis de resultados (evaluación, representaciones gráficas). Estas
están pensadas para usarse también con el resto de algoritmos (importando
la clase Event).

Requiere los siguientes archivos:
    * mapping.csv: Información de los hilos del plano.
    * param.py: Parámetros del sistema.
    * funciones.py: Funciones auxiliares.
    * event.py: Clase Event

"""

#%% Importación de librerías y otros archivos
import numpy as np
import pandas as pd
import uproot
from matplotlib import pyplot as plt

from scipy.spatial import distance, distance_matrix
from sklearn.cluster import AgglomerativeClustering

import itertools
from pathlib import Path
from datetime import datetime

import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(DIR)
sys.path.insert(0, DIR)

mapping = pd.read_csv('scripts/mapping.csv', index_col=0)
from scripts.param import param
from scripts.event import Event
from scripts.funciones import find_chain

#%% Ajustes

# Ajustes generales
general_settings = {'description': 'Ad-hoc',    # str or None : Descripción a mostrar en log.txt
                    'filelist' : 'files',       # str : Carpeta de archivos root
                    'event' : [2,1,1],          # list : Sucesos a reconstruir
                    'save' : True,              # bool : Guardar resultados
                    'show' : False              # bool : Mostrar gráficas
}

# Ajustes de preprocesamiento
pre_settings = {'n' : 10,                       # int
                'm_depos' : 2,                  # int
                'dist_th_depos' : 0.5,          # float
                'm_hits' : 4,                   # int
                'dist_th_hits' : 20.0,          # float
                'alpha' : 100.0,                # float
                'p' : 0.05                      # float
}

# Ajustes del algoritmo
alg_settings = {'V_list' : np.concatenate((np.arange(0.05,2.05,0.1),np.arange(2,5.1))), # array-like
                'dist_th_iso' : 30.0,           # float
                'iso_width': 3,                 # int
                'm_matches' : 8,                # int
                'dist_th_matches' : 2.0         # float
}

settings = {**general_settings, **pre_settings, **alg_settings}

results = pd.DataFrame({})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#%% Definición de clase

class AdHoc(Event):
    """
    Clase usada para añadir el Algoritmo Ad-hoc y la clasificación a la clase Event.
    El prefijo "iso" hace referencia a los sucesos isócronos simples y a las trazas
    isócronas simples, mientras que el prefijo "duo" hace referencia a los sucesos
    isócronos múltiples y las trazas isócronas múltiples.

    ...
    Atributos
    ----------
    iso_matches : dataframe
        matches referentes a una traza isócrona simple
    label : int
        etiqueta de clase predicha
    GIS : float
        grado de isocronía simple
    GIM : float
        grado de isocronía múltiple

    Métodos
    -------
    classify(settings)
        Clasificación del suceso en ordinario, isócrono simple, isócrono múltiple
    iso_match(settings)
        Reconstrucción mediante Algoritmo Ad-hoc
    """
    
    def __init__(self, entry):
        """
        Parámetros
        ----------
        entry : dict
            Diccionario con los datos del TTree del archivo root
        """

        super().__init__(entry)
        self.algorithm = 'adhoc'
        self.iso_matches = pd.DataFrame()
        self.label = 0
        self.GIS = 0
        self.GIM = 0

    def classify(self, settings):
        """Clasificación del suceso en ordinario, isócrono simple, isócrono múltiple

        Actualiza los atributos label, GIS y GIM mediante la
        clasificación basada en histogramas.

        Toma los siguientes ajustes del diccionario alg_settings: 
        * iso_width: ancho de la barra de los histogramas

        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        """

        tpc = self.hits.TPC.mode()[0]   #seleccionamos el TPC más poblado
        self.GIS = np.zeros(3)   #un número para cada plano
        self.GIM = np.zeros(3)
            
        ## RECONOCIMIENTO DE TRAZAS
        for plane in range(3):
            selection = self.hits.loc[(self.hits.Plane == plane) & (self.hits.TPC == tpc) ,:].copy()
            og_len = len(selection) #n de hits total

            if len(selection)==0:
                self.GIS[plane] = np.nan
                self.GIM[plane] = np.nan
                continue

            selection['iso_status'] = 0 #columnas que indican la pertenencia del hit a traza isócrona
            selection['duo_status'] = 0

            # Cálculo del histograma
            hist, edges = np.histogram(selection.Time, bins=np.append(np.arange(selection.Time.min(), selection.Time.max(), settings['iso_width']), selection.Time.max()))
            
            # Búsqueda de traza isócrona simple
            for i,h in enumerate(hist):
                if h >= 12:
                    selection.loc[(selection.Time >= edges[i]) & (selection.Time <= edges[i+1]),'iso_status'] = 1

            iso_selection = (selection.loc[selection.iso_status == 1]
                                    .drop(columns=['iso_status','duo_status', 'Integral','AdjTime'])
                            )
            self.GIS[plane] = len(iso_selection) / og_len

            # Búsqueda de traza isócrona múltiple
            chains = find_chain(hist, height=3, length=20)

            for chain in chains:
                selection.loc[(selection.Time >= edges[chain[0]]) & (selection.Time <= edges[chain[1]]),'duo_status'] = 1

            
            duo_selection = (selection.loc[selection.duo_status == 1]
                                    .drop(columns=['iso_status', 'duo_status', 'Integral','AdjTime'])
                            )
            self.GIM[plane] = len(duo_selection) / og_len
            
        ## CLASIFICACIÓN
        if (np.count_nonzero(self.GIS[:]) >= 2) or (np.count_nonzero(self.GIM[:]) >= 2):
            self.GIS = np.nanmax(self.GIS)
            self.GIM = np.nanmax(self.GIM)

            if (self.GIS <= 0.05) & (self.GIM <= 0.05):
                self.label = 0
            elif self.GIS > self.GIM:
                self.label = 1
            else:
                self.label = 2
        else:
            self.GIS = 0
            self.GIM = 0
            self.label = 0
            
    def iso_match(self, settings):
        """Matching de la traza isócrona según el Algoritmo Ad-hoc

        La identificación de la traza isócrona se repite, considerando esta vez la posibilidad
        de diferentes trazas isócronas en el mismo suceso. En este caso cada una se asignará a
        un clúster diferente.

        Toma los siguientes ajustes del diccionario alg_settings: 
        * dist_th_iso: distancia máxima entre hits de un mismo clúster
        * iso_width: ancho de las barras de los histogramas para la identificación de la traza
        
        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        """

        dist_th = settings['dist_th_iso']

        tpc = self.hits.TPC.mode()[0]   #seleccionamos el TPC más poblado

        ## IDENTIFICACIÓN DE TRAZAS ISÓCRONAS SIMPLES
        iso = [[],[],[]]        #lista con una componente para cada plano. Cada componente es una 
                                #lista con las trazas isócronas que hay en un mismo plano.
        direction = [[],[],[]]  #lista con una componente para cada plano. Cada componente es una
                                #lista con las direcciones de las trazas isócronas de iso.

        # Búsqueda de clústers
        for plane in range(3):
            selection = self.hits.loc[(self.hits.Plane == plane) & (self.hits.TPC == tpc) ,:].copy()

            if len(selection)==0:
                direction[plane].append(None)
                iso[plane].append(pd.DataFrame())
                continue

            selection['iso_status'] = 0

            # Cálculo del histograma
            hist, edges = np.histogram(selection.Time, bins=np.append(np.arange(selection.Time.min(), selection.Time.max(), settings['iso_width']), selection.Time.max()))

            # Búsqueda de trazas isócronas simples
            for i,h in enumerate(hist):
                if h >= 12:
                    selection.loc[(selection.Time >= edges[i]) & (selection.Time <= edges[i+1]),'iso_status'] = 1
                        
            iso_selection = (selection.loc[selection.iso_status == 1]
                                    .drop(columns=['iso_status', 'Integral','AdjTime'])
                            )

            if len(iso_selection) < 10:
                direction[plane].append(None)
                iso[plane].append(pd.DataFrame())
                continue

            # Clústering y eliminación de ruido
            cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single', distance_threshold=dist_th)
            cluster.fit_predict(iso_selection.loc[:,['w','Time']])
            
            iso_selection['cluster_id'] = cluster.labels_
            iso_selection.loc[np.isin(iso_selection.cluster_id, np.where(np.bincount(iso_selection.cluster_id)<8)[0]), 'cluster_id'] = -1
            iso_selection.query('cluster_id!=-1',inplace=True)

            if len(iso_selection) < 10:
                direction[plane].append(None)
                iso[plane].append(pd.DataFrame())
                continue
            
            iso_selection.loc[:,'cluster_id'].replace({label : i for i, label in enumerate(iso_selection.groupby('cluster_id').Time.mean().sort_values().index)}, inplace=True)
            #reasignación de etiquetas de clústers a 0,1,2,...

            # Determinación del sentido de recorrido para cada traza (almacenado en direction)
            for cluster in iso_selection.cluster_id.unique():

                iso_cluster = (iso_selection.loc[iso_selection.cluster_id == cluster,:]
                                            .drop(columns=['iso_status', 'cluster_id'], errors='ignore')
                                            .reset_index(drop=True)
                                )

                end1 = iso_cluster.loc[(iso_cluster.w.idxmin())]        #posiciones de los extremos
                end2 = iso_cluster.loc[(iso_cluster.w.idxmax())]
                cm = selection.loc[:,['w','Time']].sum()/len(selection) #posición del centro

                dist1 = distance.euclidean(cm, end1.loc[['w','Time']])
                dist2 = distance.euclidean(cm, end2.loc[['w','Time']])
                
                if dist1 < dist2:
                    direction[plane].append(True)
                else:
                    direction[plane].append(False)

                iso_cluster = (iso_cluster.sort_values('w',ascending=direction[plane][-1])  #el clúster también se ordena
                                            .reset_index(drop=True)
                                )

                iso[plane].append(iso_cluster)  #iso almacena para cada plano las trazas ordenadas

        # REORDENACIÓN DE CLÚSTERS (para agrupar las distintas proyecciones de cada traza)
        #este paso asegura que las trazas de los tres planos en la lista iso se correspondan
        #correctamente en el caso de que haya un número desigual de trazas en cada plano.

        if (len(iso[0]) != len(iso[1])) | (len(iso[0]) != len(iso[2])):   #en el caso de que el nº de trazas en cada plano sea diferente hay que ajustarlos
            max_clusters = len(max(iso,key=len))
            ref_times = [cluster.Time.mean() for cluster in max(iso,key=len)]   #se toma como referencia el plano con mayor número de trazas y nos basamos
                                                                                #en el tiempo medio de cada clúster

            for i in range(3):
                if len(iso[i]) < max_clusters:
                    try:
                        times = [cluster.Time.mean() for cluster in iso[i]]
                        newpos = np.argmin(distance_matrix([[t] for t in ref_times], [[t] for t in times]),axis=0)  #minimizamos la distancia en tiempo entre las diferentes proyecciones
                    except AttributeError:
                        newpos = []

                    iso_temp = [pd.DataFrame() for _ in range(max_clusters)]    #creamos clústers vacíos para homogeneizar el formato
                    dir_temp = [None for _ in range(max_clusters)]

                    for j, val in enumerate(newpos):
                        iso_temp[val] = pd.concat([iso_temp[val], iso[i][j]], axis=0)
                        iso_temp[val] = (iso_temp[val].sort_values('w',ascending=direction[i][j])
                                                        .reset_index(drop=True)
                                        )
                        dir_temp[val] = direction[i][j]

                    iso[i] = iso_temp
                    direction[i] = dir_temp

        ## MATCHING
        iso = list(map(list, zip(*iso)))
        direction = list(map(list, zip(*direction)))

        for cluster in range(len(iso)): #para cada traza isócrona
            iso_eq = [] #en esta lista se almacenan los df con los hits preparados para el matching

            # Hits de la traza más numerosa
            iso_long = sorted(iso[cluster], key=len)[-1]
            wlong = (iso_long.loc[:,'w'].mean(), iso_long.loc[:,'w'].std())
            iso_long['wnorm'] = (iso_long.loc[:,'w'] - wlong[0])/wlong[1]

            # Hits de las trazas menos numerosas
            for i in range(len(iso[cluster])-1):
                iso_short = sorted(iso[cluster], key=len)[i]
                iso_short['status'] = 0

                if len(iso_short) == 0:
                    continue

                wshort = (iso_short.loc[:,'w'].mean(), iso_short.loc[:,'w'].std())
                iso_short['wnorm'] = (iso_short.loc[:,'w'] - wshort[0])/wshort[1]

                iso_short = iso_short.set_index('wnorm')
                iso_short = (pd.concat([iso_short, pd.DataFrame(index=iso_long.wnorm, data={'status':[1]*len(iso_long)})])
                            .sort_index(ascending=False)
                            .interpolate(method='index', limit_direction='both')
                            .query('status==1')
                            .drop(columns='status')
                            .sort_values('w',ascending=direction[cluster][iso_short.Plane.unique()[0]])
                            .reset_index()
                            )
                iso_eq.append(iso_short)

            iso_eq.append(iso_long)
            
            # Matching
            for iso_1, iso_2 in list(itertools.combinations(iso_eq, 2)):    
                iso_matches = pd.concat([iso_1.add_suffix('_1'),iso_2.add_suffix('_2')], axis=1)

                iso_matches = (iso_matches.rename(columns={'w_1':'w'+str(int(iso_1.Plane.unique()[0])+1),
                                                            'w_2':'w'+str(int(iso_2.Plane.unique()[0])+1),
                                                            'Time_1':'Ind'+str(int(iso_1.Plane.unique()[0])+1)+'Time',
                                                            'Time_2':'Ind'+str(int(iso_2.Plane.unique()[0])+1)+'Time',
                                                            'TPC_1':'TPC'})
                                            .drop(['wnorm_1','wnorm_2','status_1','status_2','TPC_2','Plane_1','Plane_2'], axis=1, errors='ignore')
                            )
                
                if 'w1' not in iso_matches.columns:
                    iso_matches['w1'] = np.nan
                    iso_matches['Ind1Time'] = np.nan
                elif 'w2' not in iso_matches.columns:
                    iso_matches['w2'] = np.nan
                    iso_matches['Ind2Time'] = np.nan
                else:
                    iso_matches['w3'] = np.nan
                    iso_matches['Ind3Time'] = np.nan

                iso_matches.rename(columns={'Ind3Time':'ColTime'}, inplace=True)
                iso_matches['cluster_id'] = cluster
                self.iso_matches = pd.concat([self.iso_matches, iso_matches], axis=0)

        # OBTENCIÓN DE COORDENADAS XYZ
        if len(self.iso_matches) > 0:
            NaN_pos = np.where(self.iso_matches.w1.isna(),
                               1,
                               np.where(self.iso_matches.w2.isna(),
                                        2,
                                        np.where(self.iso_matches.w3.isna(),
                                                 3,
                                                 0)))

            self.iso_matches['X'] = np.where(NaN_pos == 3,
                                             ((-param['twin'] + 0.5*self.iso_matches.Ind1Time-self.nuvS[0]*1e-3)*param['v'] - param['a1'])*(1-2*self.iso_matches.TPC),
                                             ((-param['twin'] + 0.5*self.iso_matches.ColTime-self.nuvS[0]*1e-3 - param['p']*(1/param['v12']+1/param['v23']))*param['v'] - param['a1'])*(1-2*self.iso_matches.TPC)
                                            )

            self.iso_matches['Y'] = np.where(NaN_pos == 1,
                                             (param['p']/(np.tan(param['theta']))*(self.iso_matches.w2/np.cos(param['theta']) - self.iso_matches.w3) + param['b'])*(1-2*self.iso_matches.TPC),
                                             np.where(NaN_pos == 2,
                                                      (param['p']/(np.tan(param['theta']))*(-self.iso_matches.w1/np.cos(param['theta']) + self.iso_matches.w3) + param['b'])*(1-2*self.iso_matches.TPC),
                                                      (param['p']/(2*np.sin(param['theta']))*(-self.iso_matches.w1 + self.iso_matches.w2) + param['b'])*(1-2*self.iso_matches.TPC)
                                                     )
                                            )

            self.iso_matches['Z'] = np.where(NaN_pos == 3,
                                             param['p']/(2*np.cos(param['theta']))*(self.iso_matches.w1 + self.iso_matches.w2) + param['c1'],
                                             param['p']*self.iso_matches.w3 + param['c2']
                                            )

            self.iso_matches = self.iso_matches.loc[:,['X','Y','Z']].groupby([self.iso_matches.index, self.iso_matches.cluster_id]).mean()

        self.matches = pd.concat([self.matches, self.iso_matches], axis=0)

        print(f'Evento [{self.runID},{self.subrunID},{self.eventID}] completado: {len(self.matches)} matches en {self.runtime} s. Isocronía: {self.label}') 

#%% Ejecución del algoritmo ad-hoc

if __name__ == '__main__':

    # Selección de lista de sucesos
    p = Path(settings['filelist'])
    filelist = p.glob('*.root')
    if len(settings['event']) == 1:
        pathlist = p.glob('*' + 'R' + str(settings['event'][0]) + '-' + str(settings['event'][0]) + '_*.root')
    elif len(settings['event']) > 1:
        pathlist = p.glob('*' + 'R' + str(settings['event'][0]) + '-' + str(settings['event'][0]) + '_SR' + str(settings['event'][1]) + '-' + str(settings['event'][1])  + '.root')

    # Creación de directorios de resultados
    if settings['save']:
        Path('results').mkdir(exist_ok=True)
        
        results_path = Path('results') / (timestamp + '_AH')
        results_path.mkdir(exist_ok=True)

        img_path = results_path / 'img'
        img_path.mkdir(exist_ok=True)
    else:
        img_path = None


    for entry in uproot.iterate(pathlist, library="np", step_size=1):
    
        ## SELECCIÓN DEL SUCESO
        if len(settings['event']) == 3:
            if not (entry['EventID'] == settings['event'][2]):
                continue
        
        event = AdHoc(entry)    # instanciación de la clase

        if len(event.hits) < 100: continue  # filtro de sucesos vacíos
        
        
        ## ALGORITMO AD-HOC
        event.pre(settings)                                 # Preprocesamiento
        event.ref_match(settings['V_list'])                 # Matching
        event.clean()                                       # Eliminación de duplicados
        event.classify(settings)                            # Clasificación
        if event.label == 1:
            event.filter(settings)                          # Filtrado
            event.iso_match(settings)                       # Matching adhoc
            event.clean()                                   # Eliminación de duplicados
        else:
            print(f'Evento [{event.runID},{event.subrunID},{event.eventID}] completado: {len(event.matches)} matches en {event.runtime} s')
        

        ## EVALUACIÓN Y REPRESENTACIÓN GRÁFICA
        results = event.evaluate(results)
        # event.space_plot(alpha=0.1, rec=True, savepath=img_path)
        # event.time_plot(rec=True, dec=False, savepath=img_path)
        event.hybrid_plot(rec=True, dec=False, show_results=None, savepath=img_path)


    print('Resultados: ')
    print(results)

    # Almacenamiento de resultados
    # Crea una carpeta que incluye:
    # * img: Subcarpeta con las imágenes
    # * log.txt: Información de los parámetros de la reconstrucción
    # * metrics.csv: Evaluación de las reconstrucciones
    if settings['save']:
        eval_path = results_path / 'metrics.csv'
        results.to_csv(eval_path, header=True, index=False)
        
        log_path = results_path / 'log.txt'

        with open(results_path / 'log.txt', 'w') as file:
            for k,v in settings.items():
                if isinstance(v, np.ndarray):
                    file.write(f'{k}: {v.tolist()}, \n')
                else:
                    file.write(f'{k} : {v}, \n')

if settings['show']:
    plt.show()
