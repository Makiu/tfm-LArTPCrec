# -*- coding: utf-8 -*-
"""Algoritmo de cercanía

Este script permite al usuario aplicar el algoritmo de cercanía a los
sucesos almacenados en archivos root.

Hace uso de la clase Event del archivo event.py para el preprocesamiento
y el postprocesamiento.

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

from pathlib import Path
from datetime import datetime
import time

import os
import sys
DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, DIR)

mapping = pd.read_csv('scripts/mapping.csv')
from scripts.funciones import intersect, filter_by_tscore, reconstruct
from scripts.param import param
from scripts.event import Event

#%% Ajustes

# Ajustes generales
general_settings = {'description': 'Cercania',  # str or None : Descripción a mostrar en log.txt
                    'filelist' : 'files',       # str : Carpeta de archivos root
                    'event' : [1,1,1],          # list : Sucesos a reconstruir
                    'save' : True,              # bool : Guardar resultados
                    'show' : False              # bool : Mostrar gráficas
}

# Ajustes de preprocesamiento
pre_settings = {'n' : 10,                       # int
                'm_depos' : 15,                 # int
                'dist_th_depos' : 0.5,          # float
                'm_hits' : 4,                   # int
                'dist_th_hits' : 20.0,          # float
                'alpha' : 100.0,                # float
                'p' : 0.05                      # float
}

# Ajustes del algoritmo
alg_settings = {'V_list' : np.concatenate((np.arange(0.05,2.05,0.1),np.arange(2,5.1))), # array-like
                'm_matches' : 8,                # int
                'dist_th_matches' : 2.0,        # float
                'r_0' : 1.0,                    # float
                'tau_0' : 1.0,                  # float
                'r_factor' : 3.0,               # float
                'tau_factor' : 20.0,            # float
                'granularity' : 5,              # int
                'fill_dist' : 2.0,              # float
                'm_matches_2' : 8,              # int
                'dist_th_matches_2' : 2,        # float
}

settings = {**general_settings, **pre_settings, **alg_settings}

results = pd.DataFrame({})                              # Dataframe donde se almacenan los resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    # Tiempo de inicio

#%% Definición de clase

class Cercania(Event):
    """
    Clase usada para añadir el Algoritmo de Cercanía a la clase Event.

    ...

    Métodos
    -------
    match(settings)
        Reconstrucción mediante Algoritmo de Cercanía    
    """
    
    def __init__(self, entry):
        """
        Parámetros
        ----------
        entry : dict
            Diccionario con los datos del TTree del archivo root
        """

        super().__init__(entry)
        self.algorithm = 'cercania'
        self.hits['tscore'] = np.ones(len(self.hits)) * 9999

    def match(self, settings):
        """Matching del algoritmo de referencia

        Devuelve los matches obtenidos a partir de los hits presentes mediante ventanas
        3D y el criterio de t-score.

        El df matches se emplea también para colocar las ventanas en cada iteración.

        Toma los siguientes ajustes del diccionario alg_settings: 
        * r_0, r_factor: rango de la lista R
        * tau_0, tau_factor: rango de la lista T
        * granularity: granularidad de las listas R y T

        Parámetros
        ----------
        settings : dict
            diccionario con los ajustes relevantes
        """

        start = time.time()
        self.matches['status'] = np.zeros(len(self.matches), dtype=np.int8)
        #variable que indica el tamaño de ventana a usar en cada match (el índice en las listas R,T)
        
        self.hits.drop(columns='Integral', inplace=True)
        self.hits = self.hits.astype({'w':'float32', 'Plane':'int8', 'TPC':'int8',
                                      'Time':'float32', 'AdjTime':'float32',                
                                      'tscore':'float32'})

        # Cálculo del tscore para los matches de AR
        shift01 = 2*param['p']/param['v12'] # en ticks        
        shift12 = 2*param['p']/param['v23'] #en ticks

        AdjTime1 = self.matches.Ind1Time + shift01 + shift12
        AdjTime2 = self.matches.Ind2Time + 0 + shift12
        AdjTime3 = self.matches.ColTime + 0 + 0

        t21 = np.abs(AdjTime2 - AdjTime1)
        t31 = np.abs(AdjTime3 - AdjTime1)
        t32 = np.abs(AdjTime3 - AdjTime2)
        self.matches['tscore'] = np.nanmax(np.concatenate([[t21],[t31],[t32]], axis=0), axis=0)
        self.matches = self.matches.astype({'Ind1Time':'float32', 'Ind2Time':'float32', 'ColTime':'float32',
                                            'w1':'float32', 'w2':'float32', 'w3':'float32', 'TPC':'int8',
                                            'X':'float32','Y':'float32','Z':'float32',
                                            'status':'int8','tscore':'float32'})

        # Obtención de las listas R y T
        granularity = settings['granularity']
        R_list = settings['r_0'] * np.linspace(1,settings['r_factor'],granularity)
        T_list = settings['tau_0'] * np.linspace(1,settings['tau_factor'],granularity)
        
        # Matching
        while True:
            
            # Identificación de matches disponibles
            #en cada iteración se aplica una ventana (con un tamaño) a cada match disponible.
            #un match disponible es aquel que se ha descubierto y en el que no se ha examinado
            #con una ventana con el tamaño máximo. Estos se identifican con un valor de la columna
            #status inferior a "granularity"

            current_matches = self.matches.query('status < @granularity').copy()

            if len(current_matches) == 0:
                break   #si no quedan matches disponibles, finalizar matching

            # Obtención de todos los canales de cada match disponible (para colocar las ventanas)
            current_matches.loc[current_matches.w1.isna(),'w1'] = \
                1/param['p']*(-np.sin(param['theta'])*(current_matches.Y - param['b'])*(1-2*current_matches.TPC) + np.cos(param['theta'])*(current_matches.Z - param['c1']))
            current_matches.loc[current_matches.w2.isna(),'w2'] = \
                1/param['p']*(np.sin(param['theta'])*(current_matches.Y - param['b'])*(1-2*current_matches.TPC) + np.cos(param['theta'])*(current_matches.Z - param['c1']))
            current_matches.loc[current_matches.w3.isna(),'w3'] = \
                1/param['p']*(current_matches.Z - param['c2'])

            # Pipeline principal
            candidates = (current_matches.assign(T = lambda df: np.where(df.ColTime.isna(), df[['Ind1Time','Ind2Time']].mean(axis=1), df.ColTime),  #tiempo medio de cada ventana
                                                 tmin = lambda df: df['T'] - [T_list[i] for i in df.status],                                      #tiempos mín y máx de cada ventana
                                                 tmax = lambda df: df['T'] + [T_list[i] for i in df.status],
                                                 w1min = lambda df: df['w1'] - [R_list[i] // param['p'] for i in df.status],                        #canales mín y máx en los tres planos
                                                 w1max = lambda df: df['w1'] + [R_list[i] // param['p'] for i in df.status],
                                                 w2min = lambda df: df['w2'] - [R_list[i] // param['p'] for i in df.status],
                                                 w2max = lambda df: df['w2'] + [R_list[i] // param['p'] for i in df.status],
                                                 w3min = lambda df: df['w3'] - [R_list[i] // param['p'] for i in df.status],
                                                 w3max = lambda df: df['w3'] + [R_list[i] // param['p'] for i in df.status])
                                         .merge(self.hits, how='cross', suffixes=('_m','_h'))                                                       #obtención de los hilos relevantes de cada ventana
                                         .query('(AdjTime > tmin) & (AdjTime < tmax) & (TPC_m == TPC_h)')                                           #aplicación del filtro en tiempo y TPC
                                         .query('((w > w1min) & (w < w1max) & (Plane == 0)) | ((w > w2min) & (w < w2max) & (Plane == 1)) | ((w > w3min) & (w < w3max) & (Plane == 2))') #aplicación del filtro en canales
                                         .merge(mapping, how='left', left_on=['w','Plane','TPC_h'], right_on=['w','Plane','TPC'])                   #obtención de la pendiente y ordenada en el origen de cada hilo
                                         .loc[:,['X','Y','Z','status','w','Plane','TPC','Time','AdjTime','tscore_h','slope','intercept']]
                                         .astype({'TPC':'int8','slope':'float32','intercept':'float32'})
                                         .groupby(by=['X','Y','Z','status'])
                                         .apply(intersect, by='Plane')                                                                              #obtención de las intersecciones entre dos hilos
                         )
            
            if len(candidates) == 0: break

            candidates = (candidates.drop(columns=[label+'_B' for label in ['X','Y','Z','status','TPC']])
                                    .rename(columns={label+'_A':label for label in ['X','Y','Z','status','TPC']})
                                    .astype({'X':'float64','Y':'float64','Z':'float64'})
                                    .assign(yc=lambda df: - (df.intercept_B - df.intercept_A) / (df.slope_B - df.slope_A),                          #cálculo de la posición de cada intersección
                                            zc=lambda df: df.slope_A * df.yc + df.intercept_A,
                                            d=lambda df: np.sqrt((df.yc - df.Y)**2 + (df.zc - df.Z)**2))                                            #distancia entre intersección y centro de ventana
                                    .query('d<@R_list[status]')                                                                                     #eliminar intersecciones fuera de la ventana
                                    .drop(columns=['slope_A','slope_B','intercept_A','intercept_B','yc','zc','d'])
                                    .assign(tscore=lambda df: np.abs(df.AdjTime_A - df.AdjTime_B))                                                  #cálculo del tscore
                                    .query('(tscore_h_A > tscore)&(tscore_h_B > tscore)')                                                           #criterio de tscore (entre candidatos actuales)
                                    .rename(columns={'tscore_h_A':'tscore_A','tscore_h_B':'tscore_B'})
                                    .assign(w1=lambda df: np.where(df.Plane_A == 0, df.w_A, np.nan),                                                #cambio de formato
                                            w2=lambda df: np.where(df.Plane_A == 1, df.w_A, np.where(df.Plane_B == 1, df.w_B, np.nan)),
                                            w3=lambda df: np.where(df.Plane_B == 2, df.w_B, np.nan),
                                            Ind1Time=lambda df: np.where(df.Plane_A == 0, df.Time_A, np.nan),
                                            Ind2Time=lambda df: np.where(df.Plane_A == 1, df.Time_A, np.where(df.Plane_B == 1, df.Time_B, np.nan)),
                                            ColTime=lambda df: np.where(df.Plane_B == 2, df.Time_B, np.nan),
                                            tscore_A=lambda df: df.tscore,
                                            tscore_B=lambda df: df.tscore)
                                    .pipe(filter_by_tscore)                                                                                         #criterio de tscore (con todos los matches)                                                                                     
                                    .pipe(reconstruct, param, self.nuvS[0])                                                                        #obtención de las coords XYZ
                                    .astype({'w1':'float32', 'w2':'float32', 'w3':'float32', 'newX':'float32','newY':'float32','newZ':'float32','newstatus':'int8'})
                                    .reset_index(drop=True)
                          )
                         
            candidates = candidates.loc[(candidates.merge(self.matches.drop_duplicates(subset=['X','Y','Z']), how='left', left_on=['newX','newY','newZ'], right_on=['X','Y','Z'], indicator=True)['_merge'] == 'left_only').values,:]
            #eliminar matches obtenidos en esta iteración que se han obtenido ya previamente

            # Actualización de matches (actualizar status y añadir nuevos matches)
            self.matches.status += 1    #por defecto en todos los matches se prueba la siguiente ventana más grande en la siguiente iteración
            self.matches.loc[self.matches.status > granularity, 'status'] = granularity
            self.matches.loc[(self.matches.merge(candidates.drop_duplicates(subset=['X','Y','Z']), how='left', left_on=['X','Y','Z'], right_on=['X','Y','Z'], indicator=True)['_merge'] == 'both').values,'status'] = granularity
            #en los matches donde se hayan encontrado nuevos matches no se aplican más ventanas

            self.matches = pd.concat([self.matches, candidates.drop(columns=['X','Y','Z','status']).rename(columns={'newX':'X','newY':'Y','newZ':'Z','newstatus':'status'}).loc[:,self.matches.columns]], axis=0, ignore_index=True)
            
            # Actualización de hits (para poder realizar el criterio de tscore)
            labels = self.hits.columns
            self.hits = (self.hits.merge(candidates.rename(columns={'TPC':'TPC_A'})[[label+'_A' for label in labels]], how='left', left_on=labels[:-1].tolist(), right_on=[label+'_A' for label in labels][:-1])
                                  .merge(candidates.rename(columns={'TPC':'TPC_B'})[[label+'_B' for label in labels]], how='left', left_on=labels[:-1].tolist(), right_on=[label+'_B' for label in labels][:-1])
                        )
            self.hits['tscore'] = self.hits.loc[:,['tscore','tscore_A','tscore_B']].min(axis=1, skipna=True)
            self.hits = self.hits.iloc[:,:6]
            self.hits = self.hits.sort_values('tscore').drop_duplicates(subset=['w','Plane','TPC','Time'], keep='last')

            del candidates

        self.matches = filter_by_tscore(self.matches)   #criterio de tscore final

        self.matches.drop(columns=['status','tscore'])

        self.runtime = np.round(time.time() - start + self.runtime,4)
        print(f'Evento [{self.runID},{self.subrunID},{self.eventID}] completado: {len(self.matches)} matches en {time.time()-start} s')
       

#%% Ejecución del algoritmo de cercanía

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
        results_path = Path('resultados') / (timestamp + '_C')
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
            
        event = Cercania(entry) # instanciación de la clase

        if len(event.hits) < 100: continue  # filtro de sucesos vacíos


        ## ALGORITMO DE CERCANÍA
        event.pre(settings)                                                             # Preprocesamiento
        event.ref_match(settings['V_list'])                                             # Matching inicial (AR)
        event.clean()                                                                   # Eliminación de duplicados (AR)
        event.filter(settings=settings)                                                 # Primer filtrado
        event.match(settings)                                                           # Matching principal
        event.filter(settings, settings['m_matches_2'], settings['dist_th_matches_2'])  # Segundo filtrado
        event.fill(settings)                                                            # Promediado local
        event.clean()                                                                   # Eliminación de duplicados
        
        event.matches.to_csv('temp/matches_2_1_c.csv', index=False)
        ## EVALUACIÓN Y REPRESENTACIÓN GRÁFICA
        results = event.evaluate(results)
        # event.space_plot(alpha=0.1, rec=True, savepath=img_path)                      # Gráfica en XYZ
        # event.time_plot(rec=True, dec=True, savepath=img_path)                        # Gráfica en tiempo vs canal
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
