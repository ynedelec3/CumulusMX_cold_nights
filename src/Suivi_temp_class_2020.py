#!/usr/bin/env python
# coding: utf-8

# Creation de graphes colores par classes (python module tslearn, classification KMeans)
# (TODO) Les graphes sont enregistres pour etre transferes sur une page web
# Deux grandeurs sont tracees : temperature exterieure et cumul des temperatures negatives

import sys
import random

import pandas as pd
import numpy as np

import datetime as dt
import time

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns

from sklearn import cluster

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

autostop = 15.
df_stop = pd.read_csv('C:\\CumulusMX\\web\\realtimewikiT.txttmp', sep = ';', index_col=False)
if (float(df_stop.temp[0].replace(',', '.')) > autostop) :
    sys.exit()

seuilh = 1.
seuilb = -7.
nclasses = 5
#looprange = 330
looprange = 1
#sleep1 = 180
sleep1 = 2
sleep2 = 1

palette = 'Set1'


df_nov = pd.read_csv('C:\\CumulusMX\\data\\nov20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_dec = pd.read_csv('C:\\CumulusMX\\data\\déc20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_jan = pd.read_csv('C:\\CumulusMX\\data\\janv21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_fev = pd.read_csv('C:\\CumulusMX\\data\\févr21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_mar = pd.read_csv('C:\\CumulusMX\\data\\mars21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_avr = pd.read_csv('C:\\CumulusMX\\data\\avr21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))

plt.ion()

sns.set(rc={'figure.figsize':(20., 12.)})
sns.set_style("darkgrid", {"grid.color": "0.2", "axes.facecolor": ".9", "axes.facecolor": "0.", "figure.facecolor": "white"})
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]

df_act = pd.read_csv("C:\\CumulusMX\\data\\mai21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act], ignore_index=True)
df.drop(np.arange(3, 28), axis = 1, inplace = True)
df['t'] = df[0] + ' ' + df[1]
df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
df.drop([0, 1], axis = 1, inplace = True)
df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
df.rename(columns={2 : 'temp'}, inplace = True)
df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 10:00', '%d/%m/%Y %H:%M'))]
df['cumul'] = [min(x, 0) for x in df.temp]
df['cumul'] = df['cumul'].cumsum()
for name, group in df.groupby('date'): 
    df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))
nuitsi = nuits.copy()
# nuitsi = nuits.loc[(nuits['heure'] <= dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0))].copy()

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.ylim([-1000, 0])
plt.title('2020 Cumuls - Nuits complètes, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul_2020.png')
time.sleep(sleep2)
plt.close()

# g = sns.lineplot(x = 'heure', y = 'cumul', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
# mng = plt.get_current_fig_manager()
# mng.canvas.set_window_title('Courbe températures nocturnes')
# mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
# mng.window.state('iconic')
# xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
# g.set_xticks(xlabels)
# g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
# plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
# plt.axhline(0, c='white', lw=1)
# plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
# plt.ylim([-1000, 0])
# plt.title('Cumuls - Nuits entamées, coloration chronologique')
# plt.pause(0.001)
# plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumuli_2020.png')
# time.sleep(sleep2)
# plt.close()

g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.title('2020 Températures - Nuits complètes, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_2020.png')
time.sleep(sleep2)
plt.close()

# g = sns.lineplot(x = 'heure', y = 'temp', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
# mng = plt.get_current_fig_manager()
# mng.canvas.set_window_title('Courbe températures nocturnes')
# mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
# mng.window.state('iconic')
# xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
# g.set_xticks(xlabels)
# g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
# plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
# plt.axhline(0, c='white', lw=1)
# plt.axhline(-2, c='white', lw=1, ls = '--')
# plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
# plt.title('Températures - Nuits entamées, coloration chronologique')
# plt.pause(0.001)
# plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_tempi_2020.png')
# time.sleep(sleep2)
# plt.close()

for i in range(looprange) :
    time.sleep(sleep1)
    plt.close()
    df_act = pd.read_csv("C:\\CumulusMX\\data\\mai21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
    df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act], ignore_index=True)
    df.drop(np.arange(3, 28), axis = 1, inplace = True)
    df['t'] = df[0] + ' ' + df[1]
    df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
    df.drop([0, 1], axis = 1, inplace = True)
    df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
    df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
    df.rename(columns={2 : 'temp'}, inplace = True)
    df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
    df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 10:00', '%d/%m/%Y %H:%M'))]
    df['cumul'] = [min(x, 0) for x in df.temp]
    df['cumul'] = df['cumul'].cumsum()
    for name, group in df.groupby('date'): 
        df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
    nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))
    nuitsi = nuits.copy()
    # nuitsi = nuits.loc[(nuits['heure'] <= dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0))].copy()

    cumuldata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'cumul':'min'}).rename(columns={'t' : 't', 'cumul' : 'cumul'})
    cumuldatai = nuitsi.groupby([nuitsi.date, nuitsi.heure]).agg({'t' : 'min', 'cumul':'min'}).rename(columns={'t' : 't', 'cumul' : 'cumul'})
    tempsdata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'temp':'min'}).rename(columns={'t' : 't', 'temp' : 'temp'})
    tempsdatai = nuitsi.groupby([nuitsi.date, nuitsi.heure]).agg({'t' : 'min', 'temp':'min'}).rename(columns={'t' : 't', 'temp' : 'temp'})
    dates = nuits.groupby(by = 'date').agg({'date' : 'first'}).reset_index(drop=True)
    idatescomp = [i for i in range(dates.size)]
    # iauj = dates.index[(dates.date == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%y"))].tolist()[0]
    # idatescomp.remove(iauj)
    # datescomp = dates.drop(index = iauj)
    datescomp = dates.copy()
    listedates = datescomp.date.to_list()
    # listedates.append(dates.iloc[iauj].date)
    cumulseries = [cumuldata.loc[dates.iloc[i]].reset_index(drop=True).cumul.to_list() for i in idatescomp]
    cumulseriesi = [cumuldatai.loc[dates.iloc[i]].reset_index(drop=True).cumul.to_list() for i in idatescomp]
    tempseries = [tempsdata.loc[dates.iloc[i]].reset_index(drop=True).temp.to_list() for i in idatescomp]
    tempseriesi = [tempsdatai.loc[dates.iloc[i]].reset_index(drop=True).temp.to_list() for i in idatescomp]
    formatted_cumul = to_time_series_dataset(cumulseries)
    formatted_cumuli = to_time_series_dataset(cumulseriesi)
    formatted_temps = to_time_series_dataset(tempseries)
    formatted_tempsi = to_time_series_dataset(tempseriesi)
    # cumulaujseries = cumuldata.loc[dates.iloc[iauj]].reset_index(drop=True).cumul.to_list()
    cumulaujseries = cumuldata.reset_index(drop=True).cumul.to_list()
    # tempaujseries = tempsdata.loc[dates.iloc[iauj]].reset_index(drop=True).temp.to_list()
    tempaujseries = tempsdata.reset_index(drop=True).temp.to_list()
    formatted_cumulauj = to_time_series_dataset(cumulaujseries)
    formatted_tempsauj = to_time_series_dataset(tempaujseries)

    modelc = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
    modelci = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
    modelc.fit(formatted_cumul)
    modelci.fit(formatted_cumuli)
    labelsc = modelc.labels_
    labelsci = modelci.labels_
    labelcauj = modelc.predict(formatted_cumulauj)
    labelcauji = modelci.predict(formatted_cumulauj)
    labelsc = np.concatenate((labelsc, labelcauj))
    labelsci = np.concatenate((labelsci, labelcauji))

    modelt = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
    modelti = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
    modelt.fit(formatted_temps)
    modelti.fit(formatted_tempsi)
    labelst = modelt.labels_
    labelsti = modelti.labels_
    labeltauj = modelt.predict(formatted_tempsauj)
    labeltauji = modelti.predict(formatted_tempsauj)
    labelst = np.concatenate((labelst, labeltauj))
    labelsti = np.concatenate((labelsti, labeltauji))
    nlabels = len(listedates)

    sizes_classes = [1 for i in range(nlabels)]
    sizes_classes[-1] = 3

    nuits['classec'] = nuits.date
    nuitsi['classec'] = nuitsi.date
    for d, c in zip(listedates, labelsc):
        nuits['classec'] = nuits['classec'].replace([d], c)
    for d, c in zip(listedates, labelsci):
        nuitsi['classec'] = nuitsi['classec'].replace([d], c)
    tric = zip(listedates, labelsc)
    trici = zip(listedates, labelsci)
    vraies_datesc = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in tric]
    vraies_datesc = sorted(vraies_datesc, key = lambda x: x[0])
    vraies_datesci = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in trici]
    vraies_datesci = sorted(vraies_datesci, key = lambda x: x[0])
    sorteddatesc = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datesc]
    sorteddatesci = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datesci]
    couleurs_classesc = [ts[1] for ts in vraies_datesc]
    couleurs_classesci = [ts[1] for ts in vraies_datesci]
    for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
        couleurs_classesc = [color if i == label else i for i in couleurs_classesc]
    for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
        couleurs_classesci = [color if i == label else i for i in couleurs_classesci]

    nuits['classet'] = nuits.date
    nuitsi['classet'] = nuitsi.date
    for d, c in zip(listedates, labelst):
        nuits['classet'] = nuits['classet'].replace([d], c)
    for d, c in zip(listedates, labelsti):
        nuitsi['classet'] = nuitsi['classet'].replace([d], c)
    trit = zip(listedates, labelst)
    triti = zip(listedates, labelsti)
    vraies_datest = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in trit]
    vraies_datest = sorted(vraies_datest, key = lambda x: x[0])
    vraies_datesti = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in triti]
    vraies_datesti = sorted(vraies_datesti, key = lambda x: x[0])
    sorteddatest = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datest]
    sorteddatesti = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datesti]
    couleurs_classest = [ts[1] for ts in vraies_datest]
    couleurs_classesti = [ts[1] for ts in vraies_datesti]
    for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
        couleurs_classest = [color if i == label else i for i in couleurs_classest]
    for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
        couleurs_classesti = [color if i == label else i for i in couleurs_classesti]

    # Class areas
    minimaxc = []
    for i in range(nclasses) :
        dfmmc = nuits.loc[nuits['classec'] == i].groupby(pd.Grouper(key = 'heure')).agg({'temp': ['min', 'max'], 'cumul': ['min', 'max']}).dropna()
        minimaxc.append(dfmmc)
        dfmmc.to_csv('C:\\CumulusMX\\webfiles\\images\\mmc' + str(i) + '.csv', sep = ';', header = None)
    minimaxt = []
    for i in range(nclasses) :
        dfmmt = nuits.loc[nuits['classet'] == i].groupby(pd.Grouper(key = 'heure')).agg({'temp': ['min', 'max'], 'cumul': ['min', 'max']}).dropna()
        minimaxt.append(dfmmt)
        dfmmt.to_csv('C:\\CumulusMX\\webfiles\\images\\mmt' + str(i) + '.csv', sep = ';', header = None)
    #print(minimaxc[0]['cumul']['min'].to_list())
    
    
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    
    g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    plt.ylim([-1000, 0])
    plt.title('2020 Cumuls - Nuits complètes, coloration chronologique')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul_2020.png')
    time.sleep(sleep2)
    plt.close()

    #g = sns.lineplot(x = 'heure', y = 'cumul', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
    #mng = plt.get_current_fig_manager()
    #mng.canvas.set_window_title('Courbe températures nocturnes')
    #mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    #mng.window.state('iconic')
    #g.set_xticks(xlabels)
    #g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    #plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    #plt.axhline(0, c='white', lw=1)
    #plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    #plt.ylim([-1000, 0])
    #plt.title('Cumuls - Nuits entamées, coloration chronologique')
    #plt.pause(0.001)
    #plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumuli_2020.png')
    #time.sleep(sleep2)
    #plt.close()

    g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axhline(-2, c='white', lw=1, ls = '--')
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    plt.title('2020 Températures - Nuits complètes, coloration chronologique')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_2020.png')
    time.sleep(sleep2)
    plt.close()

    #g = sns.lineplot(x = 'heure', y = 'temp', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
    #mng = plt.get_current_fig_manager()
    #mng.canvas.set_window_title('Courbe températures nocturnes')
    #mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    #mng.window.state('iconic')
    #xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    #g.set_xticks(xlabels)
    #g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    #plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    #plt.axhline(0, c='white', lw=1)
    #plt.axhline(-2, c='white', lw=1, ls = '--')
    #plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    #plt.title('Températures - Nuits entamées, coloration chronologique')
    #plt.pause(0.001)
    #plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_tempi_2020.png')
    #time.sleep(sleep2)
    #plt.close()

    fig, axs = plt.subplots()
    g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None, ax = axs)
    for i in range(nclasses) :
        #print(sns.color_palette(palette, nlabels)[i])
        axs.fill_between(minimaxc[i].index.to_list(), minimaxc[i]['cumul']['min'].to_list(), minimaxc[i]['cumul']['max'].to_list(), color = sns.color_palette(palette, nclasses)[i], alpha=0.3)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    plt.ylim([-1000, 0])
    plt.title('2020 Cumuls - Nuits complètes, coloration par classe de cumul')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_cumul_2020.png')
    time.sleep(sleep2)
    plt.close()

    # g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classesci, size = 'date', sizes = sizes_classes, estimator=None)
    # mng = plt.get_current_fig_manager()
    # mng.canvas.set_window_title('Courbe températures nocturnes')
    # mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    # mng.window.state('iconic')
    # g.set_xticks(xlabels)
    # g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    # plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    # plt.axhline(0, c='white', lw=1)
    # plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    # plt.ylim([-1000, 0])
    # plt.title('Cumuls - Nuits entamées, coloration par classe de cumul')
    # plt.pause(0.001)
    # plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumuli_cumul_2020.png')
    # time.sleep(sleep2)
    # plt.close()

    fig, axs = plt.subplots()
    g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None, ax = axs)
    for i in range(nclasses) :
        #print(sns.color_palette(palette, nlabels)[i])
        axs.fill_between(minimaxc[i].index.to_list(), minimaxc[i]['temp']['min'].to_list(), minimaxc[i]['temp']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axhline(-2, c='white', lw=1, ls = '--')
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    plt.title('2020 Températures - Nuits complètes, coloration par classe de cumul')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_temp_2020.png')
    time.sleep(sleep2)
    plt.close()

    # g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classesci, size = 'date', sizes = sizes_classes, estimator=None)
    # mng = plt.get_current_fig_manager()
    # mng.canvas.set_window_title('Courbe températures nocturnes')
    # mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    # mng.window.state('iconic')
    # g.set_xticks(xlabels)
    # g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    # plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    # plt.axhline(0, c='white', lw=1)
    # plt.axhline(-2, c='white', lw=1, ls = '--')
    # plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    # plt.title('Températures - Nuits entamées, coloration par classe de cumul')
    # plt.pause(0.001)
    # plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumuli_temp_2020.png')
    # time.sleep(sleep2)
    # plt.close()

    fig, axs = plt.subplots()
    g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None, ax = axs)
    for i in range(nclasses) :
        #print(sns.color_palette(palette, nlabels)[i])
        axs.fill_between(minimaxt[i].index.to_list(), minimaxt[i]['temp']['min'].to_list(), minimaxt[i]['temp']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axhline(-2, c='white', lw=1, ls = '--')
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    plt.title('2020 Températures - Nuits complètes, coloration par classe de températures')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_temp_2020.png')
    time.sleep(sleep2)
    plt.close()

    # g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classesti, size = 'date', sizes = sizes_classes, estimator=None)
    # mng = plt.get_current_fig_manager()
    # mng.canvas.set_window_title('Courbe températures nocturnes')
    # mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    # mng.window.state('iconic')
    # xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    # g.set_xticks(xlabels)
    # g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    # plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    # plt.axhline(0, c='white', lw=1)
    # plt.axhline(-2, c='white', lw=1, ls = '--')
    # plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    # plt.title('Températures - Nuits entamées, coloration par classe de températures')
    # plt.pause(0.001)
    # plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_tempi_temp_2020.png')
    # time.sleep(sleep2)
    # plt.close()

    fig, axs = plt.subplots()
    g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None, ax = axs)
    for i in range(nclasses) :
        #print(sns.color_palette(palette, nlabels)[i])
        axs.fill_between(minimaxt[i].index.to_list(), minimaxt[i]['cumul']['min'].to_list(), minimaxt[i]['cumul']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    plt.ylim([-1000, 0])
    plt.title('2020 Cumuls - Nuits complètes, coloration par classe de température')
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_cumul_2020.png')
    time.sleep(sleep2)
    plt.close()

    # g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classesti, size = 'date', sizes = sizes_classes, estimator=None)
    # mng = plt.get_current_fig_manager()
    # mng.canvas.set_window_title('Courbe températures nocturnes')
    # mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    # mng.window.state('iconic')
    # xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    # g.set_xticks(xlabels)
    # g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    # plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    # plt.axhline(0, c='white', lw=1)
    # plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    # plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='white', lw=1, ls = '--')
    # plt.ylim([-1000, 0])
    # plt.title('Cumuls - Nuits entamées, coloration par classe de température')
    # plt.pause(0.001)
    # plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_tempi_cumul_2020.png')
    # time.sleep(sleep2)
    # plt.close()