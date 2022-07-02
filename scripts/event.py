# -*- coding: utf-8 -*-

"""Algoritmo de referencia y otras funcionalidades básicas

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

"""

#%% Importación de librerías y otros archivos

import numpy as np
import pandas as pd
import uproot
from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree, distance_matrix
from sklearn.cluster import AgglomerativeClustering

import time
from pathlib import Path
from datetime import datetime

import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(DIR)
sys.path.insert(0, DIR)

from scripts.param import param
mapping = pd.read_csv('scripts/mapping.csv', index_col=0)

#%% Ajustes

# Ajustes generales
general_settings = {'description': 'Referencia', # str or None : Descripción a mostrar en log.txt
                    'filepath' : 'files',       # str : Carpeta de archivos root
                    'event' : [2,1,1],          # list : Sucesos a reconstruir
                    'save' : True,              # bool : Guardar resultados
                    'show' : False              # bool : Mostrar gráficas
}

#Cómo seleccionar sucesos (item 'event'):
#según el número de elementos en la lista se interpreta de una manera u otra:
# * 0 elementos: todos los sucesos en la ruta
# * 1 elemento: todos los sucesos de la misma run
# * 2 elementos: todos los sucesos de la misma run y subrun
# * 3 elementos: solo el suceso de la run, subrun, event específica

# Ajustes de preprocesamiento
pre_settings = {'n' : 10,                       # int
                'm_depos' : 15,                 # int
                'dist_th_depos' : 0.5,          # float
                'm_hits' : 4,                   # int
                'dist_th_hits' : 20.0,          # float
                'alpha' : 100.0,                # float
                'p' : 0.05,                     # float
}

# Ajustes del algoritmo
alg_settings = {'V_list' : np.concatenate((np.arange(0.05,2.05,0.1),np.arange(2,5.1)))   # array-like
}

settings = {**general_settings, **pre_settings, **alg_settings}

results = pd.DataFrame({})                              # Dataframe donde se almacenan los resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    # Tiempo de inicio

#%% Definición de clase

class Event:
    """
    Clase usada para el procesamiento de un suceso, incluyendo el Algoritmo de
    Referencia.

    ...

    Atributos
    ----------
    algorithn : str
        el algoritmo empleado en la reconstrucción
    runtime : float
        duración de la reconstrucción
    eventID, subrunID, runID : int, int, int
        identificación del suceso
    timestamp : str
        tiempo de inicio de la reconstrucción
    nuvE : float
        energía del neutrino primario (en GeV)
    nuvS : array float
        posición y tiempo de la interacción del neutrino (en cm, ns)
    depos : dataframe
        tabla con la información de las deposiciones de energía (i.e. puntos verdaderos)
    hits : dataframe
        tabla con la información de los hits
    matches : dataframe
        tabla con la información de los matches

    Métodos
    -------
    pre(settings)
        Eliminación de ruido mediante clustering jerárquico (+ filtro en energía)
    ref_match(V_list=[1])
        Reconstrucción mediante Algoritmo de Referencia
    filter(settings, m=None, dist_th=None)
        Filtrado de matches mediante clustering jerárquico
    fill(settings)
        Promediado local de matches
    clean()
        Eliminación de matches duplicados
    evaluate(results, q=95)
        Evaluación mediante las cuatro métricas (cobertura, precisión, fscore, distancia)
    space_plot(alpha=0.008, rec=False, inplot=True, sideplot=True, savepath=None)
        Representación en el espacio XYZ
    time_plot(rec=False, dec=False, savepath=None)
        Representación en el espacio de tiempo vs canal de cada plano de detección
    hybrid_plot(rec=False, dec=False, show_results=None, savepath=None)
        Representación híbrida entre espacio de tiempo vs canal y espacio XYZ
    """
    
    def __init__(self, entry):
        """
        Parámetros
        ----------
        entry : dict
            Diccionario con los datos del TTree del archivo root
        """

        self.algorithm = 'referencia'
        self.runtime = 0

        self.eventID = entry['EventID'][0]
        self.subrunID = entry['SubRunID'][0]
        self.runID = entry['RunID'][0]
        self.timestamp = timestamp

        self.nuvE = entry['TrueVEnergy']
        self.nuvS = np.concatenate((entry['TrueVt'],entry['TrueVx'],entry['TrueVy'],entry['TrueVz']))
        
        HitsIntegral = entry['HitsIntegral'][0]
        HitsPeakTime = entry['HitsPeakTime'][0]
        HitsChannel = entry['HitsChannel'][0].astype('int32')
        
        # Depos
        self.depos = pd.DataFrame({'T':entry['EnDepT'][0],
                                   'E':entry['EnDepE'][0],
                                   'X':entry['EnDepX'][0],
                                   'Y':entry['EnDepY'][0],
                                   'Z':entry['EnDepZ'][0]})
        self.depos = (self.depos.assign(TPC=lambda df: df.X > 0)
                                .astype({'TPC':'int64'})
                                .assign(w1 = lambda df: 1/param['p']*(-np.sin(param['theta'])*(df.Y - param['b'])*(1-2*df.TPC) + np.cos(param['theta'])*(df.Z - param['c1'])),
                                        w2 = lambda df: 1/param['p']*(np.sin(param['theta'])*(df.Y - param['b'])*(1-2*df.TPC) + np.cos(param['theta'])*(df.Z - param['c1'])),
                                        w3 = lambda df: 1/param['p']*(df.Z - param['c2']),
                                        ticks1 = lambda df: 2*(df['T']*1e-3 + (param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin']),
                                        ticks2 = lambda df: 2*(df['T']*1e-3 + param['p']/param['v12'] + (param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin']),
                                        ticks3 = lambda df: 2*(df['T']*1e-3 + param['p']*(1/param['v12'] + 1/param['v23']) + (param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin'])
                                       )
                                .sort_values('T')
                                .reset_index(drop=True)
                     )
        
        # Hits
        self.hits = pd.DataFrame({'ID':HitsChannel, 'Time':HitsPeakTime, 'Integral':HitsIntegral})

        self.hits = (pd.merge(self.hits, mapping, how='inner',on='ID')
                       .drop(['s','slope','intercept'], axis=1)
                       .sort_values('Time')
                       .reset_index(drop=True)
                    )

        shift01 = 2*param['p']/param['v12'] # en ticks        
        shift12 = 2*param['p']/param['v23'] # en ticks

        self.hits['AdjTime'] = np.where(self.hits.Plane==0,                     # tiempo ajustado, teniendo en cuenta los retrasos temporales entre los planos
                                        self.hits.Time + shift01 + shift12,
                                        np.where(self.hits.Plane==1,
                                                 self.hits.Time + shift12,
                                                 self.hits.Time))

        # Matches
        self.matches = pd.DataFrame(columns=['Ind1Time', 'Ind2Time', 'ColTime', 'w1', 'w2', 'w3', 'TPC'])
    
    def pre(self, settings):
        """Eliminación de ruido mediante clústering jerárquico (+ filtro en energía, varianza)

        Toma los siguientes ajustes del diccionario pre_settings: 
        * p: cuantil del filtro de energía
        * n: número de vecinos donde buscar al vecino
        * m_depos, m_hits: número mínimo de puntos en un clúster
        * dist_th_depos, dist_th_depos: distancia máxima entre dos puntos de la
            misma cadena

        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        """

        p = settings['p']
        n, m_depos, dist_th_depos, m_hits, dist_th_hits = (settings['n'], 
                                                           settings['m_depos'], settings['dist_th_depos'], 
                                                           settings['m_hits'], settings['dist_th_hits'])

        ## ELIMINACIÓN DE RUIDO EN DEPOS
        # Filtro de energía
        self.depos.loc[:,'is_E_inlier'] = np.where(self.depos.E > self.depos.E.quantile(p), 1, 0)
        self.depos = (self.depos.query('is_E_inlier == 1')
                                .drop(columns='is_E_inlier')
                     )

        if len(self.depos) > n:

            # Clústering jerárquico
            cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single', distance_threshold=dist_th_depos)
            cluster.fit_predict(self.depos.loc[:,['X','Y','Z']])    #formación de clústers
            
            self.depos['cluster_id'] = cluster.labels_
            self.depos['cluster_status'] = np.isin(cluster.labels_, np.where(np.bincount(cluster.labels_)>=m_depos)[0])
            self.depos = self.depos.query('cluster_status')     #filtro de clústers poco numerosos

            # Filtro de varianzas
            var = self.depos.loc[:,['X','Y','Z','cluster_id']].groupby(by='cluster_id').var().mean(axis=1)
            self.depos.cluster_status = np.isin(self.depos.cluster_id, var[var > var.max()/settings['alpha']].index)
            self.depos = (self.depos.query('cluster_status')
                                    .drop(columns=['cluster_id','cluster_status'], errors='ignore')
                                    .reset_index(drop=True)
                         )

        ## ELIMINACIÓN DE RUIDO EN HITS

        filtered_hits = pd.DataFrame()

        for tpc in self.hits.TPC.unique():
            for j in range(3):

                selection = (self.hits.loc[(self.hits.TPC == tpc) & (self.hits.Plane == j),:]
                                 .copy()
                                 .reset_index(drop=True)
                            )

                # Filtro de energía
                selection.loc[:,'is_E_inlier'] = np.where(selection.Integral > selection.Integral.quantile(p), 1, 0)
                selection = (selection.query('is_E_inlier == 1')
                                      .drop(columns='is_E_inlier')
                            )   

                if len(selection) < n:
                    continue

                # Clústering jerárquico
                cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single', distance_threshold=dist_th_hits)
                cluster.fit_predict(selection.loc[:,['w','Time']])

                selection['cluster_id'] = cluster.labels_
                selection['cluster_status'] = np.isin(cluster.labels_, np.where(np.bincount(cluster.labels_)>=m_hits)[0])
                selection = (selection.query('cluster_status')
                                      .drop(columns='cluster_status')
                )

                # Filtro de varianzas
                var = selection.loc[:,['w','Time','cluster_id']].groupby(by='cluster_id').var().mean(axis=1)
                selection['cluster_status'] = np.isin(selection.cluster_id, var[var > var.max()/settings['alpha']].index)
                selection = (selection.query('cluster_status')
                                      .drop(columns=['ID','cluster_id','cluster_status'], errors='ignore')
                                      .reset_index(drop=True)
                            )

                filtered_hits = pd.concat([filtered_hits, selection], axis=0)
        
        self.hits = filtered_hits.copy()

    def ref_match(self, V_list=[1]):
        """Algoritmo de referencia

        Devuelve los matches obtenidos a partir de los hits presentes.

        Parámetros
        ----------
        V_list : array-like
            lista con los tamaños de ventana disponibles
        """

        start = time.time()
                
        self.matches = pd.DataFrame(columns=['Ind1Time', 'Ind2Time', 'ColTime', 'w1', 'w2', 'w3', 'TPC'])
        #df donde almacenar los nuevos matches

        for tpc in self.hits.TPC.unique():
            for hit in self.hits.loc[self.hits.TPC == tpc,:].itertuples(index=False):
                flag = 0    # variable auxiliar que indica si en la iteración se ha encontrado match
                ref_plane = hit.Plane
                neigh_plane = [i for i in range(3) if i != ref_plane]   #planos candidatos

                selected_hits = pd.DataFrame([hit], columns=self.hits.columns)
                #df donde se almacenan los hits que forman match en esta iteración

                for tol in V_list:
                    #df con los hits vecinos del primer plano candidato
                    bag_1 = self.hits[(self.hits.Plane == neigh_plane[0]) &
                                       (self.hits.TPC == tpc) &
                                       (abs(self.hits.AdjTime-hit.AdjTime) < tol)]
                    #df con los hits vecinos del segundo plano candidato
                    bag_2 = self.hits[(self.hits.Plane == neigh_plane[1]) &
                                       (self.hits.TPC == tpc) &
                                       (abs(self.hits.AdjTime-hit.AdjTime) < tol)]

                    #criterio de aceptación de matches
                    if len(bag_1) < 1:
                        if len(bag_2) == 1:
                            flag = 1
                            selected_hits = pd.concat([selected_hits, bag_2])
                    elif len(bag_1) == 1:
                        flag = 1
                        if len(bag_2) == 1:
                            selected_hits = pd.concat([selected_hits, bag_1, bag_2])
                            break
                        else:
                            selected_hits = pd.concat([selected_hits, bag_1])
                            if len(bag_2) > 1:
                                break
                    elif len(bag_1) > 1:
                        if len(bag_2) < 1:
                            continue
                        elif len(bag_2) == 1:
                            flag = 1
                            selected_hits = pd.concat([selected_hits, bag_2])
                            break
                        elif len(bag_2) > 1:
                            break
                                            
                if flag:
                    selected_hits.drop_duplicates(subset='Plane', keep='first', inplace=True, ignore_index=True)
                    match = pd.DataFrame({'Ind1Time':selected_hits.loc[selected_hits.Plane==0,'Time'].reset_index(drop=True),
                                          'Ind2Time':selected_hits.loc[selected_hits.Plane==1,'Time'].reset_index(drop=True),
                                          'ColTime':selected_hits.loc[selected_hits.Plane==2,'Time'].reset_index(drop=True),
                                          'w1':selected_hits.loc[selected_hits.Plane==0, 'w'].reset_index(drop=True),
                                          'w2':selected_hits.loc[selected_hits.Plane==1, 'w'].reset_index(drop=True),
                                          'w3':selected_hits.loc[selected_hits.Plane==2, 'w'].reset_index(drop=True),
                                          'TPC':tpc})
                    self.matches = pd.concat([self.matches, match], axis=0, ignore_index=True)

        # Obtención de coordenadas XYZ
        if len(self.matches) > 0:
            # Vector que indica que hilo no se utiliza en cada match
            NaN_pos = np.where(self.matches.w1.isna(),
                               1,
                               np.where(self.matches.w2.isna(),
                                        2,
                                        np.where(self.matches.w3.isna(),
                                                 3,
                                                 0)))

            #se muestra la ecuación en la memoria a la que corresponde cada línea
            self.matches['X'] = np.where(NaN_pos == 3,
                                    ((-param['twin'] + 0.5*self.matches.Ind1Time-self.nuvS[0]*1e-3)*param['v'] - param['a1'])*(1-2*self.matches.TPC),     # ec.1
                                    ((-param['twin'] + 0.5*self.matches.ColTime - self.nuvS[0]*1e-3 - param['p']*(1/param['v12'] + 1/param['v23']))*param['v'] - param['a1'])*(1-2*self.matches.TPC)  # ec.3
            )

            self.matches['Y'] = np.where(NaN_pos == 1,
                                    (param['p']/(np.tan(param['theta']))*(self.matches.w2/np.cos(param['theta']) - self.matches.w3) + param['b'])*(1-2*self.matches.TPC),  # ec.6
                                    np.where(NaN_pos == 2,
                                             (param['p']/(np.tan(param['theta']))*(-self.matches.w1/np.cos(param['theta']) + self.matches.w3) + param['b'])*(1-2*self.matches.TPC),    #ec.5
                                             (param['p']/(2*np.sin(param['theta']))*(-self.matches.w1 + self.matches.w2) + param['b'])*(1-2*self.matches.TPC)  #ec.4
                                            )
            )

            self.matches['Z'] = np.where(NaN_pos == 3,
                                    param['p']/(2*np.cos(param['theta']))*(self.matches.w1 + self.matches.w2) + param['c1'],  #ec.7
                                    param['p']*self.matches.w3 + param['c2'] #ec.8
            )   
        else:
            self.matches['X'] = []
            self.matches['Y'] = []
            self.matches['Z'] = []
        
        self.runtime = np.round(time.time() - start,4)  #duración de la reconstrucción
        if self.algorithm == 'referencia':
            print(f'Evento [{self.runID},{self.subrunID},{self.eventID}] completado: {len(self.matches)} matches en {self.runtime} s')

    def filter(self, settings, m=None, dist_th=None):
        """Filtrado de matches mediante clustering jerárquico

        Puede tomar los parámetros adicionales m y dist_th para poder aplicar el mismo
        método con diferentes parámetros en la misma ejecución (relevante para AC).

        Toma los siguientes ajustes del diccionario alg_settings: 
        * m_matches: Número mínimo de matches en un clúster
        * dist_th_matches: Máxima distancia entre dos matches de un mismo clúster

        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        m : int or None
            Número mínimo de matches en un clúster (segunda aplicación)
        dist_th_matches : float or None
            Máxima distancia entre dos matches de un mismo clúster
        """

        if m==None or dist_th==None:    # Si alguno de estos es None, tomar valores de settings
            m, dist_th = settings['m_matches'], settings['dist_th_matches']

        if len(self.matches) > settings['n']:
            cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single', distance_threshold=dist_th)
            cluster.fit_predict(self.matches.loc[:,['X','Y','Z']])
            
            self.matches['cluster_id'] = cluster.labels_
            self.matches['cluster_status'] = np.isin(cluster.labels_, np.where(np.bincount(cluster.labels_)>=m)[0])
            self.matches = self.matches.query('cluster_status')

            self.matches = self.matches.drop(columns=['cluster_id','cluster_status'], errors='ignore')

    def fill(self, settings):
        """Promediado local de matches

        Toma los siguientes ajustes del diccionario alg_settings: 
        * fill_dist: Máxima distancia entre vecinos a promediar

        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        """

        # Cálculo de los vecinos de cada punto por debajo de fill_dist
        self.matches.reset_index(drop=True, inplace=True)
        kdt = KDTree(self.matches[['X','Y','Z']])
        pairs = kdt.query_pairs(r=settings['fill_dist'], output_type='ndarray')
        pairs = pd.DataFrame(pairs, columns=['index_A','index_B'])  #df con índices de parejas de matches

        #df con parejas de matches
        match_pairs = (self.matches[['X','Y','Z']].reset_index()
                                                  .merge(self.matches[['X','Y','Z']].reset_index(), how='cross', suffixes=('_A','_B'))
                                                  .merge(pairs,how='right',on=['index_A','index_B'])
                                                  .drop(columns=['index_A','index_B'])
                      )

        # Cálculo de los nuevos puntos
        match_pairs['X'] = match_pairs[['X_A','X_B']].mean(axis=1)
        match_pairs['Y'] = match_pairs[['Y_A','Y_B']].mean(axis=1)
        match_pairs['Z'] = match_pairs[['Z_A','Z_B']].mean(axis=1)
        
        # Reconstrucción de la información en los espacios de tiempo vs canal
        match_pairs = (match_pairs.drop(columns=['X_A','Y_A','Z_A','X_B','Y_B','Z_B'])
                                  .drop_duplicates()
                                  .astype('float32')
                                  .assign(TPC=lambda df: df.X > 0,
                                          w1 = lambda df: 1/param['p']*(-np.sin(param['theta'])*(df.Y - param['b'])*(1-2*df.TPC) + np.cos(param['theta'])*(df.Z - param['c1'])),
                                          w2 = lambda df: 1/param['p']*(np.sin(param['theta'])*(df.Y - param['b'])*(1-2*df.TPC) + np.cos(param['theta'])*(df.Z - param['c1'])),
                                          w3 = lambda df: 1/param['p']*(df.Z - param['c2']),
                                          Ind1Time = lambda df: 2*((param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin']),
                                          Ind2Time = lambda df: 2*(param['p']/param['v12'] + (param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin']),
                                          ColTime = lambda df: 2*(param['p']*(1/param['v12'] + 1/param['v23']) + (param['a1'] - np.abs(df['X']))/param['v'] + self.nuvS[0]*1e-3 + param['twin']))
                                  .astype({'TPC':'int8'})
                      )
        
        # Inclusión de nuevos matches
        self.matches = pd.concat([self.matches, match_pairs], axis=0)

    def clean(self):
        """Eliminación de matches duplicados

        Para determinar duplicidad se basa en XYZ. También reordena según los tiempos de 
        detección de los hits.
        """

        self.matches = (self.matches.drop_duplicates(subset=['X','Y','Z'])
                                    .sort_values(by=['Ind1Time','Ind2Time','ColTime'])
                                    .reset_index(drop=True))

    def evaluate(self, results, q=95.0):
        """Evaluación mediante las cuatro métricas (cobertura, precisión, fscore, distancia)

        Almacena los nuevos resultados en results. Incluye también datos sobre la clasificación
        de isocronía si el algoritmo es adhoc (donde se encuentra el algoritmo de clasificación).

        Parámetros
        ----------
        results : dataframe
            tabla con los resultados de cada suceso
        q : float
            cuantil empleado en la distancia de Hausdorff modificada
        """

        if (len(self.matches) > 0) and (len(self.depos) > 0):
            
            # Matriz de distancias entre matches y depos
            dist_mat = distance_matrix(self.matches.loc[:,['X','Y','Z']], self.depos.loc[:,['X','Y','Z']])

            dist_min_depos = dist_mat.min(axis=0) #lista de distancias entre cada depos y su match más cercano
            dist_dir = np.percentile(dist_min_depos, q)
            coverage = sum(dist_min_depos <= 0.3)/len(dist_min_depos)*100

            dist_min_matches = dist_mat.min(axis=1) #lista de distancias entre cada match y su depos más cercano
            dist_inv = np.percentile(dist_min_matches, q)
            precision = sum(dist_min_matches <= 0.3)/len(dist_min_matches)*100

            fscore = 2*coverage*precision/(coverage + precision)
            dist =  max(dist_dir, dist_inv)
        else:
            coverage, precision, fscore, dist = 0, 0, 0, np.nan

        current_results = pd.DataFrame({'ID': [str(self.runID) + '_' + str(self.subrunID) + '_' + str(self.eventID)],
                                        'matches': [len(self.matches)],
                                        'tiempo': [self.runtime],
                                        'coverage': [coverage],
                                        'precision': [precision],
                                        'fscore': [fscore],
                                        'distance': [dist]
                                        })
        
        if self.algorithm == 'adhoc':
            current_results['label'] = self.label               # Etiqueta predicha
            current_results['GIS'] = self.GIS                   # GIS
            current_results['GIM'] = self.GIM                   # GIM

        results = pd.concat([results, current_results], axis=0)
        return results

    def space_plot(self, alpha=0.008, rec=False, inplot=True, sideplot=True, savepath=None):
        """Representación en el espacio XYZ

        Representa la imagen 3D del suceso y sus proyecciones en los tres planos cartesianos.

        Parámetros
        ----------
        alpha : float
            grado de transparencia de las proyecciones en la imagen 3D
        rec : bool
            si representar los puntos reconstruidos o no
        inplot : bool
            si representar las proyecciones en los planos cartesianos en la misma imagen 3D
        sideplot : bool
            si representar las proyecciones en los planos cartesianos en otros subplots dedicados
        savepath : Path, str or None
            ruta donde almacenar la imagen (si es None no se almacena)
        """

        if rec:
            if len(self.matches) == 0:
                rec = False
        
        fig = plt.figure(tight_layout=True, figsize=(24,15))
        gs = gridspec.GridSpec(3, 2, width_ratios=(2,1))
        
        ax = fig.add_subplot(gs[:, 0], projection='3d')
        
        # Puntos verdaderos en la imagen 3D
        ax.scatter(self.depos.X, self.depos.Y, self.depos.Z, c='darkorange', s=8)

        # Reconstrucción en la imagen 3D
        if rec:
            ax.scatter(self.matches['X'], self.matches['Y'], self.matches['Z'], marker='+', s=5, c='k')
        
        # Proyecciones en la imagen 3D
        if inplot:
            ax.plot(self.depos.X, self.depos.Z, linestyle='', marker='^', markersize=5, color='lightcoral', zdir='y', zs=200, alpha=alpha)
            ax.plot(self.depos.Y, self.depos.Z, linestyle='', marker='^', markersize=5, color='greenyellow', zdir='x', zs=-200, alpha=alpha)
            ax.plot(self.depos.X, self.depos.Y, linestyle='', marker='^', markersize=5, color='cornflowerblue', zdir='z', zs=0, alpha=alpha)
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
 
        # Proyecciones en subplots
        if sideplot:
            for i in range(3):
                ax = fig.add_subplot(gs[i, 1])
                
                if i==0:
                    ax.scatter(self.depos.Y, self.depos.Z, s=5, c='greenyellow')
                    if rec:
                        ax.scatter(self.matches['Y'], self.matches['Z'], marker='+', s=5, c='k', zorder=2)

                    ax.set_xlabel('Y (cm)')
                    ax.set_ylabel('Z (cm)')
                    
                elif i==1:
                    ax.scatter(self.depos.X, self.depos.Y, s=5, c='cornflowerblue')
                    if rec:
                        ax.scatter(self.matches['X'], self.matches['Y'], marker='+', s=5, c='k', zorder=2)                      

                    ax.set_xlabel('X (cm)')
                    ax.set_ylabel('Y (cm)')

                else:
                    ax.scatter(self.depos.X, self.depos.Z, s=5, c='lightcoral')
                    if rec:
                        ax.scatter(self.matches['X'], self.matches['Z'], marker='+', s=5, c='k', zorder=2)

                    ax.set_xlabel('X (cm)')
                    ax.set_ylabel('Z (cm)')
        
        # Almacenamiento
        if savepath is not None:
            fig.savefig(savepath / ('s' + '_' + 'R' + str(self.runID) + '_SR' + str(self.subrunID) + '_E' + str(self.eventID)), bbox_inches='tight')
            
    def time_plot(self, rec=False, dec=False, savepath=None):
        """Representación en el espacio de tiempo vs canal de cada plano de detección

        Muestra los hits y también puede mostrar los matches y los depos.

        Parámetros
        ----------
        rec : bool
            si representar los puntos reconstruidos o no
        dec : bool
            si representar los puntos verdaderos/depos o no
        savepath : Path, str or None
            ruta donde almacenar la imagen (si es None no se almacena)
        """

        if rec:
            if len(self.matches) == 0:
                rec = False
                    
        fig = plt.figure(figsize=(24,15))
        gs = fig.add_gridspec(1,6, wspace=0)
        axs = gs.subplots(sharex=False, sharey=True)
        column_label = ['Ind1Time', 'Ind2Time', 'ColTime']
        
        for i in range(6):
            if i<3: #TPC 0
                current_hits = self.hits[(self.hits.Plane==i) & (self.hits.TPC==0)]

                # Hits
                axs[i].plot(current_hits.w, current_hits.Time,
                            marker='o', fillstyle=None, color='royalblue', linestyle='', alpha=0.5, zorder=1)
                
                # Matches
                if rec and (0 in self.matches.TPC.unique()):
                    current_matches = self.matches.loc[self.matches.TPC==0,:]
                    axs[i].plot(current_matches.loc[:,'w'+str(i+1)], current_matches.loc[:,column_label[i]],
                                marker='x', color='mediumblue', linestyle='', zorder=2)
                    
                # Depos
                if dec:
                    axs[i].plot(self.depos.loc[self.depos.TPC == 0,'w'+str(i+1)], self.depos.loc[self.depos.TPC == 0,'ticks'+str(i+1)],
                                'ko', linestyle='', zorder=0)
                
                plane_name = 'Inducción '+str(i) if i<2 else 'Colección'
                axs[i].set_title(plane_name, fontsize=20)
                axs[i].tick_params(axis='both', which='major', labelsize=20)
                
            else:   #TPC 1
                current_hits = self.hits[(self.hits.Plane==i-3) & (self.hits.TPC==1)]

                # Hits
                axs[i].plot(current_hits.w, current_hits.Time,
                            marker='o', fillstyle=None, color='indianred', linestyle='', alpha=0.5, zorder=1)
                
                # Matches
                if rec and (1 in self.matches.TPC.unique()):
                    current_matches = self.matches.loc[self.matches.TPC==1,:]
                    axs[i].plot(current_matches.loc[:,'w'+str(i-2)], current_matches.loc[:,column_label[i-3]],
                                marker='x', color='darkred', linestyle='', zorder=2)
                    
                # Depos
                if dec:
                    axs[i].plot(self.depos.loc[self.depos.TPC == 1,'w'+str(i-2)], self.depos.loc[self.depos.TPC == 1,'ticks'+str(i-2)],
                                'ko', linestyle='', zorder=0)
                    
                plane_name = 'Inducción '+str(i-3) if i<5 else 'Colección'
                axs[i].set_title(plane_name, fontsize=20)
                axs[i].tick_params(axis='both', which='major', labelsize=20)   
                
        fig.suptitle('Canal', y=0.06, fontsize=30, verticalalignment='bottom')
        axs[0].set_ylabel('Tiempo (ticks)', fontsize=30)
        axs[0].tick_params(axis='both', which='major', labelsize=20)

        # Almacenamiento
        if savepath is not None:
            fig.savefig(savepath / ('t' + '_' + 'R' + str(self.runID) + '_SR' + str(self.subrunID) + '_E' + str(self.eventID)), bbox_inches='tight')
    
    def hybrid_plot(self, rec=False, dec=False, show_results=None, savepath=None):
        """Representación híbrida entre espacio de tiempo vs canal y espacio XYZ

        Muestra las proyecciones de los planos cartesianos y los espacios de tiempo 
        vs canal del TPC más poblado.

        Parámetros
        ----------
        rec : bool
            si representar los puntos reconstruidos o no
        dec : bool
            si representar los puntos verdaderos/depos o no
        show_results : dataframe or None
            métricas a mostrar mediante texto
        savepath : Path, str or None
            ruta donde almacenar la imagen (si es None no se almacena)
        """

        if rec:
            if len(self.matches) == 0:
                rec = False
                    
        fig = plt.figure(figsize=(20,22))
        # gs = gridspec.GridSpec(4, 3, height_ratios=(0.22,0.22,0.22,0.33))
        outer = gridspec.GridSpec(2,1, height_ratios=(2, 1), hspace=0.2)
        gs1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec = outer[0], hspace = 0)
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec = outer[1])

        column_label = ['Ind1Time', 'Ind2Time', 'ColTime']
        plane_name = ['Inducción 1 ', 'Inducción 2', 'Colección']
        tpc = self.depos.TPC.value_counts().idxmax()

        ## Gráficas de tiempo vs canal
        for i in range(3):
            if i==0:
                ax = fig.add_subplot(gs1[i])
            else:
                ax = fig.add_subplot(gs1[i], sharex=ax)

            current_hits = self.hits[(self.hits.Plane==i) & (self.hits.TPC==tpc)]

            # Hits
            ax.plot(current_hits.Time, current_hits.w,
                    marker='o', fillstyle=None, color='orange', linestyle='', alpha=0.5, zorder=1)
            
            # Matches
            if rec:
                current_matches = self.matches.loc[self.matches.TPC==tpc,:]
                ax.plot(current_matches.loc[:,column_label[i]], current_matches.loc[:,'w'+str(i+1)],
                        marker='x', color='orangered', linestyle='', zorder=2)
                
            # Depos
            if dec:
                ax.plot(self.depos.loc[self.depos.TPC == tpc,'ticks'+str(i+1)], self.depos.loc[self.depos.TPC == tpc,'w'+str(i+1)],
                        'ko', linestyle='', zorder=0)

            ax.set_ylabel('w', fontsize=20, labelpad=0)
            ax.tick_params(axis='y', which='major', labelsize=20)

            if i!=2:
                ax.tick_params(axis='x', which='major', labelsize=15, labelbottom=False)
            else:
                ax.tick_params(axis='x', which='major', labelsize=15)
                ax.set_xlabel('Tiempo (ticks)', fontsize=20)

            ax.text(0.5,0.99, plane_name[i], fontsize=20, horizontalalignment='center', verticalalignment='top', transform = ax.transAxes)
            ax.yaxis.set_label_coords(-0.05, 0.5)
        
        ## Gráficas de las proyecciones cartesianas XYZ
        ax1 = fig.add_subplot(gs2[0])
        ax2 = fig.add_subplot(gs2[1])
        ax3 = fig.add_subplot(gs2[2])
        
        # Depos
        ax1.scatter(self.depos.Y, self.depos.Z, s=5, c='greenyellow')
        ax2.scatter(self.depos.X, self.depos.Y, s=5, c='cornflowerblue')
        ax3.scatter(self.depos.X, self.depos.Z, s=5, c='lightcoral')

        # Matches
        if rec:
            ax1.scatter(self.matches['Y'], self.matches['Z'], marker='+', s=5, c='k', zorder=2, alpha=0.5)
            ax2.scatter(self.matches['X'], self.matches['Y'], marker='+', s=5, c='k', zorder=2, alpha=0.5)                      
            ax3.scatter(self.matches['X'], self.matches['Z'], marker='+', s=5, c='k', zorder=2, alpha=0.5)

        # Resultados
        if show_results is not None:
            str_results = show_results.iloc[-1,1:].transpose().to_string(header=False)
            t = ax1.text(0.05, -0.3, str_results, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, backgroundcolor='white', alpha=0.5)
            t.set_bbox(dict(facecolor='black', alpha=0.2, edgecolor='black'))

        ax1.set_xlabel('Y (cm)', fontsize=20)
        ax1.set_ylabel('Z (cm)', fontsize=20)
        ax2.set_xlabel('X (cm)', fontsize=20)
        ax2.set_ylabel('Y (cm)', fontsize=20)
        ax3.set_xlabel('X (cm)', fontsize=20)
        ax3.set_ylabel('Z (cm)', fontsize=20)

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=14)

        # Almacenamiento
        if savepath is not None:
            fig.savefig(savepath / ('h' + '_' + 'R' + str(self.runID) + '_SR' + str(self.subrunID) + '_E' + str(self.eventID)), bbox_inches='tight')
            
#%% Ejecución del algoritmo de referencia

if __name__ == '__main__':

    # Selección de lista de sucesos
    p = Path(settings['filepath'])
    pathlist = p.glob('*.root')
    if len(settings['event']) == 1:
        pathlist = p.glob('*' + 'R' + str(settings['event'][0]) + '-' + str(settings['event'][0]) + '_*.root')
    elif len(settings['event']) > 1:
        pathlist = p.glob('*' + 'R' + str(settings['event'][0]) + '-' + str(settings['event'][0]) + '_SR' + str(settings['event'][1]) + '-' + str(settings['event'][1])  + '.root')

    # Creación de directorios de resultados
    if settings['save']:
        Path('results').mkdir(exist_ok=True)

        results_path = Path('results') / (timestamp + '_REF')
        results_path.mkdir(exist_ok=True)

        img_path = results_path / 'img'
        img_path.mkdir(exist_ok=True)
    else:
        img_path = None


    for entry in uproot.iterate(pathlist, library='np', step_size=1):
    
        ## SELECCIÓN DEL SUCESO
        if len(settings['event']) == 3:
            if not (entry['EventID'] == settings['event'][2]):
                continue
        
        event = Event(entry) # instanciación de la clase

        if len(event.hits) < 100: continue  # filtro de sucesos vacíos
        

        ## ALGORITMO DE REFERENCIA
        event.pre(settings)                                 # Preprocesamiento
        event.ref_match(settings['V_list'])                 # Matching
        event.clean()                                       # Eliminación de duplicados


        ## EVALUACIÓN Y REPRESENTACIÓN GRÁFICA
        results = event.evaluate(results)
        # event.space_plot(alpha=0.1, rec=True, savepath=img_path)                      # Gráfica en XYZ
        # event.time_plot(rec=True, dec=False, savepath=img_path)                       # Gráfica en tiempo vs canal
        event.hybrid_plot(rec=True, dec=False, show_results=None, savepath=img_path)    # Gráfica híbrida


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
 