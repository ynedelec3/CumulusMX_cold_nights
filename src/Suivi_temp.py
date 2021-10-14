#!/usr/bin/env python
# coding: utf-8

# Creation de graphes colores chronologiquement
# Les graphes sont enregistres pour etre transferes sur une page web
# Deux grandeurs sont tracees : temperature exterieure et cumul des temperatures negatives

import sys

import pandas as pd
import numpy as np

import datetime as dt
import time

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns

autostop = 15.
df_stop = pd.read_csv("C:\\CumulusMX\\web\\realtimewikiT.txttmp", sep = ';', index_col=False)
if (float(df_stop.temp[0].replace(',', '.')) > autostop) :
    sys.exit()

seuilh = 10.
seuilb = -7.

df_oct = pd.read_csv('C:\\CumulusMX\\data\\oct21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_nov = pd.read_csv('C:\\CumulusMX\\data\\nov20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_dec = pd.read_csv('C:\\CumulusMX\\data\\déc20log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_jan = pd.read_csv('C:\\CumulusMX\\data\\janv21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_fev = pd.read_csv('C:\\CumulusMX\\data\\févr21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_mar = pd.read_csv('C:\\CumulusMX\\data\\mars21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df_avr = pd.read_csv('C:\\CumulusMX\\data\\avr21log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))


plt.ion()

sns.set(rc={'figure.figsize':(20., 12.)})
sns.set_style("darkgrid", {"grid.color": "0.2", "axes.facecolor": ".9", "axes.facecolor": "0.", "figure.facecolor": "white"})

pd.set_option('mode.chained_assignment', None)

df_act = pd.read_csv("C:\\CumulusMX\\data\\oct21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
#df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act])
df = df_act
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

g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
#plt.legend(bbox_to_anchor=(0.1, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', title = 'init')
plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
time.sleep(5.)
#plt.clf()
plt.close()

g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
mng = plt.get_current_fig_manager()
mng.canvas.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
g.set_xticks(xlabels)
g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
#plt.legend(bbox_to_anchor=(0.694, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', title = 'init')
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
time.sleep(5.)
#plt.clf()

for i in range(330) :
    time.sleep(180)
    plt.close()
    df_act = pd.read_csv("C:\\CumulusMX\\data\\oct21log.txt", sep = ';', header=None, index_col=False, names = np.arange(0, 28))
    #df = pd.concat([df_nov, df_dec,df_jan, df_fev, df_mar, df_avr, df_act])
    df = df_act
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

    g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    #plt.legend(bbox_to_anchor=(0.1, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', title = str(i))
    plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
    plt.axhline(0, c='white', lw=1)
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
    time.sleep(5.)
    #plt.clf()
    plt.close()

    g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = 'viridis', estimator=None)
    mng = plt.get_current_fig_manager()
    mng.canvas.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    g.set_xticks(xlabels)
    g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    #plt.legend(bbox_to_anchor=(0.694, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', title = str(i))
    plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
    plt.axhline(0, c='white', lw=1)
    plt.axhline(-2, c='white', lw=1, ls = '--')
    plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
    plt.pause(0.001)
    plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
    #plt.clf()
    plt.close()
