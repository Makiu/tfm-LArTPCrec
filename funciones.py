# -*- coding: utf-8 -*-
"""Funciones

Este script contiene funciones empleadas en el resto de scripts.
"""

import numpy as np
import pandas as pd

def intersect(df, by='Plane'):
    """Intersección de hilos encontrados en cada ventana

        Toma los siguientes ajustes del diccionario alg_settings: 
        * fill_dist: Máxima distancia entre vecinos a promediar

        Parámetros
        ----------
        df : dataframe
            dataframe dividido por grupos para cada ventana
        by : str
            nombre de la variable de los planos
    """

    new_df = (df.merge(df, how='cross', suffixes=('_A','_B'))
                .query(by+'_A < '+by+'_B')
             )
    return new_df

def filter_by_tscore(matches):
    """Criterio de t-score

        Aplica el criterio de t-score a los matches recibidos.

        Parámetros
        ----------
        matches : dataframe
            tabla con los matches
    """

    for i,timename in zip(['1','2','3'],['Ind1Time','Ind2Time','ColTime']):
        matches = matches.sort_values(by='tscore')
        na_matches = matches.loc[matches['w'+i].isna(),:]   #se reservan los matches con NA
        matches = (matches.dropna(axis=0, subset='w'+i)
                          .groupby(by=['w'+i,timename])
                          .head(2)                          #parámetro n=2
                )
        matches = pd.concat([matches, na_matches], axis=0)

    return matches

def reconstruct(matches, param, t0):
    """Obtención de las coordenadas XYZ

        Obtiene las coords XYZ a partir de los canales de los hilos

        Parámetros
        ----------
        matches : dataframe
            tabla con los matches a transformar
        param : dict
            parámetros del detector
        t0 : float
            tiempo de inicio del suceso
    """

    matches_X = np.where(matches.Plane_B == 2,
                         matches.Time_B,
                         np.mean([matches.AdjTime_A, matches.AdjTime_B], axis=0))
    matches_X = ((-param['twin'] + 0.5*matches_X-t0*1e-3 - param['p']*(1/param['v12']+1/param['v23']))*param['v'] - param['a1'])*(1-2*matches.TPC)

    matches_Y = np.where(matches.Plane_B == 2, 
                            np.where(matches.Plane_A == 0,
                                    param['p']/(np.tan(param['theta']))*(-matches.w_A/np.cos(param['theta']) + matches.w_B) + param['b'],
                                    param['p']/(np.tan(param['theta']))*(matches.w_A/np.cos(param['theta']) - matches.w_B) + param['b']
                                    ),
                            param['p']/(2*np.sin(param['theta']))*(-matches.w_A + matches.w_B) + param['b']
                        )
    matches_Y = matches_Y*(1 - 2*matches.TPC)
    
    matches_Z = np.where(matches.Plane_B == 2, 
                            param['p']*matches.w_B + param['c2'], 
                            param['p']/(2*np.cos(param['theta']))*(matches.w_A + matches.w_B) + param['c1']
                        )

    matches['newX'] = matches_X
    matches['newY'] = matches_Y
    matches['newZ'] = matches_Z
    matches['newstatus'] = 0
    
    return matches

def find_chain(hist, height=3.0, length=5):
    """Búsqueda de cadenas en histogramas de tiempo

        Búsqueda de cadenas para la clasificación de sucesos
        isócronos múltiples

        Parámetros
        ----------
        hist : array-like
            lista de alturas de las barras del histograma
        height : float
            altura mínima de las barras de una traza isócrona múltiple
        length : int
            número mínimo de barras para una traza isócrona múltiple
    """

    ishigh = np.concatenate(([0], (hist > height).view(np.int8), [0]))
    absdiff = np.abs(np.diff(ishigh))
    
    chains = np.where(absdiff == 1)[0].reshape(-1, 2)

    if chains.size > 0:
        short_chains = np.apply_along_axis(np.diff,axis=1, arr=chains).T
        short_chains = np.where(short_chains[0] < length)
        chains = np.delete(chains, short_chains, axis=0)

    return chains