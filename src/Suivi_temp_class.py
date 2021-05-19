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

autostop = 5.
df_stop = pd.read_csv("C:\\CumulusMX\\web\\realtimewikiT.txttmp", sep = ';', index_col=False)
if (float(df_stop.temp[0].replace(',', '.')) > autostop) :
    sys.exit()

seuilh = 1.
seuilb = -7.

df_nov = pd.read_csv('C:\\CumulusMX\\data\\nov20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_dec = pd.read_csv('C:\\CumulusMX\\data\\déc20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_jan = pd.read_csv('C:\\CumulusMX\\data\\janv21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_fev = pd.read_csv('C:\\CumulusMX\\data\\févr21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_mar = pd.read_csv('C:\\CumulusMX\\data\\mars21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df_avr = pd.read_csv('C:\\CumulusMX\\data\\mars21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))

plt.ion()

sns.set(rc={'figure.figsize':(20., 12.)})
sns.set_style("darkgrid", {"grid.color": "0.2", "axes.facecolor": ".9", "axes.facecolor": "0.", "figure.facecolor": "white"})
palette = 'Set1'
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]

df_act = pd.read_csv("C:\\CumulusMX\\data\\mai21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act])
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
nuitsi = nuits.loc[(nuits['heure'] <= dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0))].copy()

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.ylim([-1000, 0])
plt.title('Cumuls - Nuits complètes, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
time.sleep(1.)
plt.close()

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.ylim([-1000, 0])
plt.title('Cumuls - Nuits entamées, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumuli.png')
time.sleep(1.)
plt.close()

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
plt.title('Températures - Nuits complètes, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
time.sleep(1.)
plt.close()

g = sns.lineplot(x = 'heure', y = 'temp', data = nuitsi, hue = 'date', palette = 'viridis', estimator=None)
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
plt.title('Températures - Nuits entamées, coloration chronologique')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_tempi.png')
time.sleep(1.)

for i in range(330) :
    time.sleep(180)
    plt.close()
    df_act = pd.read_csv("C:\\CumulusMX\\data\\mai21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
    df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act])
    #print(df.size)
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
    nuitsi = nuits.loc[(nuits['heure'] <= dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0))].copy()

    ###g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    ##mng = plt.get_current_fig_manager()
    ##mng.canvas.set_window_title('Courbe températures nocturnes')
    ##mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    ##mng.window.state('iconic')
    ###xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    ###g.set_xticks(xlabels)
    ###g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    ###plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
    ###plt.axhline(0, c='white', lw=1)
    ###plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    ###plt.pause(0.001)
    ##plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
    ###time.sleep(5.)
    ###plt.close()

    ###g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    ##mng = plt.get_current_fig_manager()
    ##mng.canvas.set_window_title('Courbe températures nocturnes')
    ##mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    ##mng.window.state('iconic')
    ###xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    ###g.set_xticks(xlabels)
    ###g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    #plt.legend(bbox_to_anchor=(0.694, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', title = str(i))
    ###plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    ###plt.axhline(0, c='white', lw=1)
    ###plt.axhline(-2, c='white', lw=1, ls = '--')
    ###plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    ###plt.pause(0.001)
    ##plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
    #plt.clf()
    ###plt.close()

nclasses = 5
cumuldata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'cumul':'min'}).rename(columns={'t' : 't', 'cumul' : 'cumul'})
tempsdata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'temp':'min'}).rename(columns={'t' : 't', 'temp' : 'temp'})
dates = nuits.groupby(by = 'date').agg({'date' : 'first'}).reset_index(drop=True)
idatescomp = [i for i in range(dates.size)]
iauj = dates.index[(dates.date == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%y"))].tolist()[0]
idatescomp.remove(iauj)
datescomp = dates.drop(index = iauj)
listedates = datescomp.date.to_list()
listedates.append(dates.iloc[iauj].date)
cumulseries = [cumuldata.loc[dates.iloc[i]].reset_index(drop=True).cumul.to_list() for i in idatescomp]
tempseries = [tempsdata.loc[dates.iloc[i]].reset_index(drop=True).temp.to_list() for i in idatescomp]
formatted_cumul = to_time_series_dataset(cumulseries)
formatted_temps = to_time_series_dataset(tempseries)
cumulaujseries = cumuldata.loc[dates.iloc[iauj]].reset_index(drop=True).cumul.to_list()
tempaujseries = tempsdata.loc[dates.iloc[iauj]].reset_index(drop=True).temp.to_list()
formatted_cumulauj = to_time_series_dataset(cumulaujseries)
formatted_tempsauj = to_time_series_dataset(tempaujseries)
modelc = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
modelc.fit(formatted_cumul)
labelsc = modelc.labels_
labelcauj = modelc.predict(formatted_cumulauj)
labelsc = np.concatenate((labelsc, labelcauj))
modelt = TimeSeriesKMeans(n_clusters=nclasses, metric="dtw", max_iter=10)
modelt.fit(formatted_temps)
labelst = modelt.labels_
labeltauj = modelt.predict(formatted_tempsauj)
labelst = np.concatenate((labelst, labeltauj))
nlabels = len(listedates)
sizes_classes = [1 for i in range(nlabels)]
sizes_classes[-1] = 3

nuits['classec'] = nuits.date
for d, c in zip(listedates, labelsc):
    nuits['classec'] = nuits['classec'].replace([d], c)
tric = zip(listedates, labelsc)
vraies_datesc = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in tric]
vraies_datesc = sorted(vraies_datesc, key = lambda x: x[0])
sorteddatesc = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datesc]
couleurs_classesc = [ts[1] for ts in vraies_datesc]
for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
    couleurs_classesc = [color if i == label else i for i in couleurs_classesc]
#couleurs_classesc.append(sns.color_palette(palette, nclasses + 1)[nclasses])

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None)
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_cumul.png')
plt.close()

g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None)
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_temp.png')
plt.close()

nuits['classet'] = nuits.date
for d, c in zip(listedates, labelst):
    nuits['classet'] = nuits['classet'].replace([d], c)
trit = zip(listedates, labelst)
vraies_datest = [(dt.datetime.strptime(ts[0], "%d/%m/%y"), ts[1]) for ts in trit]
vraies_datest = sorted(vraies_datest, key = lambda x: x[0])
sorteddatest = [dt.datetime.strftime(ts[0], "%d/%m/%y") for ts in vraies_datest]
couleurs_classest = [ts[1] for ts in vraies_datest]
for color, label in zip(sns.color_palette(palette, nclasses), range(nclasses)):
    couleurs_classest = [color if i == label else i for i in couleurs_classest]
#couleurs_classest.append(sns.color_palette(palette, nclasses + 1)[nclasses])

g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None)
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_temp.png')
plt.close()

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None)
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_cumul.png')
plt.close()