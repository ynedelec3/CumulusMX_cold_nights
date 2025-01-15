#!/usr/bin/env python
# coding: utf-8

# Creation de graphes colores chronologiquement
# Superposition des classes de temperatures de l'an passe, classement par rapport aux cumuls
# Les graphes sont enregistres pour etre transferes sur une page web
# Deux grandeurs sont tracees : temperature exterieure et cumul des temperatures negatives

import sys
import gc

import pandas as pd # type: ignore
import numpy as np # type: ignore

import datetime as dt
import time
from dateutil.relativedelta import relativedelta # type: ignore

import matplotlib.pyplot as plt # type: ignore
from matplotlib.dates import DateFormatter # type: ignore
from pandas.plotting import register_matplotlib_converters # type: ignore
register_matplotlib_converters()

import seaborn as sns # type: ignore

from sklearn import cluster # type: ignore

from tslearn.utils import to_time_series_dataset # type: ignore
from tslearn.clustering import TimeSeriesKMeans # type: ignore
from tslearn.metrics import dtw # type: ignore


autostop = 10.
trace_mmc = "cur"
#trace_mmc = "prev"

df_stop = pd.read_csv("C:\\CumulusMX\\realtime.txt", sep = ' ', header=None, usecols =[2], names = ['temp'])
#print(float(df_stop.temp[0]))
if (float(df_stop.temp[0]) > autostop) :
    sys.exit()

seuilh = 1.
seuilb = -7.

pos_heure = 20.
pos_heure_c = 35.

#looprange = 390
looprange = 1

#sleep1 = 180.
sleep1 = 5.
sleep2 = 2.
plt_pause1 = 0.001

nclasses = 5

palette = "Set1"

#print(sns.color_palette(palette, nclasses))

current_month = int(dt.date.today().strftime('%m'))
#current_year = int(dt.date.today().strftime('%y'))
working_year = int((dt.date.today() + relativedelta(months=3)).strftime('%y')) - 1

newlog = 0
pref_month = ['oct', 'nov', 'déc', 'janv', 'févr', 'mars', 'avr', 'mai']
if working_year == 24:    
    newlog = 1
    pref_month = ['oct', 'nov', 'déc', '01', '02', '03', '04', '05']
if working_year > 24:    
    newlog = 2
    pref_month = ['10', '11', '12', '01', '02', '03', '04', '05']


num_month = [10, 11, 12, 1, 2, 3, 4, 5]
offset_month = [0, 0, 0, 1, 1, 1, 1, 1]

month_dataframe = ['df_oct', 'df_nov', 'df_dec', 'df_jan', 'df_fev', 'df_mar', 'df_avr', 'df_mai']

count_month = num_month.index(current_month)


i = 0
while i  < count_month :
    if newlog == 0:
        logfile = pref_month[i] + str(working_year + offset_month[i])
        sep = ';'
        dec = ','
    if newlog == 1:
        if i <= 2:
            logfile = pref_month[i] + str(working_year + offset_month[i])
            sep = ';'
            dec = ','
        else:
            logfile = str(2000 + working_year + offset_month[i]) + pref_month[i]
            sep = ','
            dec = '.'
    if newlog == 2:
        logfile = str(2000 + working_year + offset_month[i]) + pref_month[i]
        sep = ','
        dec = '.'
    globals()[month_dataframe[i]] = pd.read_csv('C:\\CumulusMX\\data\\' + logfile + 'log.txt', sep = sep, decimal = dec, header=None, index_col=False, names = np.arange(0, 28))
    i += 1

mmc = []
min_classe = []
#mmt = []
if trace_mmc == "cur":
    for i in range(nclasses) :
        dfmmc = pd.read_csv('C:\\CumulusMX\\webfiles\\images\\mmc' + str(working_year) + str(i) + '.csv', sep = ';', header = None, names = ['t', 'tmin', 'tmax', 'cmin', 'cmax'], parse_dates = ['t'])
        min_classe.append(dfmmc['cmin'].min())
        mmc.append(dfmmc)
        #dfmmt = pd.read_csv('C:\\CumulusMX\\webfiles\\images\\mmt' + str(i) + '.csv', sep = ';', header = None, names = ['t', 'tmin', 'tmax', 'cmin', 'cmax'], parse_dates = ['t'])
        #mmt.append(dfmmt)
    rang_classe = [0] * len(min_classe)
    for i, x in enumerate(sorted(range(len(min_classe)), key=lambda y: min_classe[y])):
        rang_classe[x] = i

if trace_mmc == "prev":
    for i in range(nclasses) :
        dfmmc = pd.read_csv('C:\\CumulusMX\\webfiles\\images\\mmc' + str(working_year - 1) + str(i) + '.csv', sep = ';', header = None, names = ['t', 'tmin', 'tmax', 'cmin', 'cmax'], parse_dates = ['t'])
        min_classe.append(dfmmc['cmin'].min())
        mmc.append(dfmmc)
        #dfmmt = pd.read_csv('C:\\CumulusMX\\webfiles\\images\\mmt' + str(i) + '.csv', sep = ';', header = None, names = ['t', 'tmin', 'tmax', 'cmin', 'cmax'], parse_dates = ['t'])
        #mmt.append(dfmmt)
    rang_classe = [0] * len(min_classe)
    for i, x in enumerate(sorted(range(len(min_classe)), key=lambda y: min_classe[y])):
        rang_classe[x] = i

sns.set(rc={'figure.figsize':(20., 12.)})
sns.set_style("darkgrid", {"grid.color": "0.2", "axes.facecolor": ".9", "axes.facecolor": "0.", "figure.facecolor": "white"})

pd.set_option('mode.chained_assignment', None)

xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00'), pd.Timestamp('01/02/1900 12:00')]
    
df0 = globals()[month_dataframe[0]]
j = 1
while j  < count_month :
    df0 = pd.concat([df0, globals()[month_dataframe[j]]], ignore_index=True)
    j += 1

if newlog == 0:
    logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
    sep = ';'
    dec = ','
if newlog == 1:
    if count_month <= 2:
        logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
        sep = ';'
        dec = ','
    else:
        logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
        sep = ','
        dec = '.'
if newlog == 2:
    logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
    sep = ','
    dec = '.'

globals()[month_dataframe[count_month]] = pd.read_csv('C:\\CumulusMX\\data\\' + logfile + 'log.txt', sep = sep, decimal = dec, header=None, index_col=False, names = np.arange(0, 28))
df = pd.concat([df0, globals()[month_dataframe[count_month]]], ignore_index=True)
df.drop(np.arange(3, 28), axis = 1, inplace = True)
df['t'] = df[0] + ' ' + df[1]
df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
df.drop([0, 1], axis = 1, inplace = True)
df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
df.rename(columns={2 : 'temp'}, inplace = True)
#df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 12:00', '%d/%m/%Y %H:%M'))]
df['cumul'] = [min(x, 0) for x in df.temp]
df['cumul'] = df['cumul'].cumsum()
for name, group in df.groupby('date'): 
    df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))

ndate = nuits.groupby(by=['date']).ngroups
chron_palette = sns.mpl_palette("viridis", n_colors = ndate - 1)
chron_palette.append((1., 0.5, 0.05))

gcum = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
if trace_mmc == "cur" or trace_mmc == "prev" :
    for i in range(nclasses) :
        gcum.fill_between(mmc[i]['t'].to_list(), mmc[i]['cmin'].to_list(), mmc[i]['cmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gcum.set_xticks(xlabels)
gcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.ylim([-1000, 0])
plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 3)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure_c, dt.datetime.today().strftime("%H:%M"))
plt.pause(plt_pause1)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
plt.clf()
plt.cla()
gc.collect()

gtemp = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
if trace_mmc == "cur" or trace_mmc == "prev" :
    for i in range(nclasses) :
        gtemp.fill_between(mmc[i]['t'].to_list(), mmc[i]['tmin'].to_list(), mmc[i]['tmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gtemp.set_xticks(xlabels)
gtemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.354, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 5)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure, dt.datetime.today().strftime("%H:%M"))
plt.pause(plt_pause1)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
plt.clf()
plt.cla()
gc.collect()
time.sleep(sleep1)

i = 0

while i < looprange and (dt.datetime.now() <= dt.datetime.now().replace(hour=12) or dt.datetime.now() >= dt.datetime.now().replace(hour=18)) :
    # TODO Check change of year in night is correctly managed
    current_month0 = int(dt.date.today().strftime('%m'))
    #current_year0 = int(dt.date.today().strftime('%y'))
    if current_month0 == current_month :
        i += 1
        
        if newlog == 0:
            logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
            sep = ';'
            dec = ','
        if newlog == 1:
            if count_month <= 2:
                logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
                sep = ';'
                dec = ','
            else:
                logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
                sep = ','
                dec = '.'
        if newlog == 2:
            logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
            sep = ','
            dec = '.'

        globals()[month_dataframe[count_month]] = pd.read_csv('C:\\CumulusMX\\data\\' + logfile + 'log.txt', sep = sep, decimal = dec, header=None, index_col=False, names = np.arange(0, 28))
        df = pd.concat([df0, globals()[month_dataframe[count_month]]], ignore_index=True)
        df.drop(np.arange(3, 28), axis = 1, inplace = True)
        df['t'] = df[0] + ' ' + df[1]
        df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
        df.drop([0, 1], axis = 1, inplace = True)
        df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
        df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
        df.rename(columns={2 : 'temp'}, inplace = True)
        #df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
        df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 12:00', '%d/%m/%Y %H:%M'))]
        df['cumul'] = [min(x, 0) for x in df.temp]
        df['cumul'] = df['cumul'].cumsum()
        for name, group in df.groupby('date'): 
            df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
        nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))
        
        ndate = nuits.groupby(by=['date']).ngroups
        chron_palette = sns.mpl_palette("viridis", n_colors = ndate - 1)
        chron_palette.append((1., 0.5, 0.05))

        gcum = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
        if trace_mmc == "cur" or trace_mmc == "prev" :
            for i in range(nclasses) :
                gcum.fill_between(mmc[i]['t'].to_list(), mmc[i]['cmin'].to_list(), mmc[i]['cmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
        mng = plt.get_current_fig_manager()
        mng.set_window_title('Courbe températures nocturnes')
        mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
        mng.window.state('iconic')
        gcum.set_xticks(xlabels)
        gcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
        plt.ylim([-1000, 0])
        plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 3)
        plt.axhline(0, c='white', lw=1)
        plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
        plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
        plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure_c, dt.datetime.today().strftime("%H:%M"))
        plt.pause(plt_pause1)
        plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
        plt.clf()
        plt.cla()
        gc.collect()

        gtemp = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
        if trace_mmc == "cur" or trace_mmc == "prev" :
            for i in range(nclasses) :
                gtemp.fill_between(mmc[i]['t'].to_list(), mmc[i]['tmin'].to_list(), mmc[i]['tmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
        mng = plt.get_current_fig_manager()
        mng.set_window_title('Courbe températures nocturnes')
        mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
        mng.window.state('iconic')
        gtemp.set_xticks(xlabels)
        gtemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
        plt.legend(bbox_to_anchor=(0.354, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 5)
        plt.axhline(0, c='white', lw=1)
        plt.axhline(-2, c='white', lw=1, ls = '--')
        plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
        plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
        plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure, dt.datetime.today().strftime("%H:%M"))
        plt.pause(plt_pause1)
        plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
        plt.clf()
        plt.cla()
        gc.collect()
        time.sleep(sleep1)
    else :
        current_month = current_month0
        count_month = num_month.index(current_month)
        df0 = pd.concat([df0, globals()[month_dataframe[count_month - 1]]], ignore_index=True)
        
        i += 1

        if newlog == 0:
            logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
            sep = ';'
            dec = ','
        if newlog == 1:
            if count_month <= 2:
                logfile = pref_month[count_month] + str(working_year + offset_month[count_month])
                sep = ';'
                dec = ','
            else:
                logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
                sep = ','
                dec = '.'
        if newlog == 2:
            logfile = str(2000 + working_year + offset_month[count_month]) + pref_month[count_month]
            sep = ','
            dec = '.'

        globals()[month_dataframe[count_month]] = pd.read_csv('C:\\CumulusMX\\data\\' + logfile + 'log.txt', sep = sep, decimal = dec, header=None, index_col=False, names = np.arange(0, 28))
        df = pd.concat([df0, globals()[month_dataframe[count_month]]], ignore_index=True)
        df.drop(np.arange(3, 28), axis = 1, inplace = True)
        df['t'] = df[0] + ' ' + df[1]
        df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
        df.drop([0, 1], axis = 1, inplace = True)
        df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
        df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
        df.rename(columns={2 : 'temp'}, inplace = True)
        #df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
        df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 12:00', '%d/%m/%Y %H:%M'))]
        df['cumul'] = [min(x, 0) for x in df.temp]
        df['cumul'] = df['cumul'].cumsum()
        for name, group in df.groupby('date'): 
            df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
        nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))
        
        ndate = nuits.groupby(by=['date']).ngroups
        chron_palette = sns.mpl_palette("viridis", n_colors = ndate - 1)
        chron_palette.append((1., 0.5, 0.05))

        gcum = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
        if trace_mmc == "cur" or trace_mmc == "prev" :
            for i in range(nclasses) :
                gcum.fill_between(mmc[i]['t'].to_list(), mmc[i]['cmin'].to_list(), mmc[i]['cmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
        mng = plt.get_current_fig_manager()
        mng.set_window_title('Courbe températures nocturnes')
        mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
        mng.window.state('iconic')
        gcum.set_xticks(xlabels)
        gcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
        plt.ylim([-1000, 0])
        plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 3)
        plt.axhline(0, c='white', lw=1)
        plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
        plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
        plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure_c, dt.datetime.today().strftime("%H:%M"))
        plt.pause(plt_pause1)
        plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
        plt.clf()
        plt.cla()
        gc.collect()

        gtemp = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
        if trace_mmc == "cur" or trace_mmc == "prev" :
            for i in range(nclasses) :
                gtemp.fill_between(mmc[i]['t'].to_list(), mmc[i]['tmin'].to_list(), mmc[i]['tmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
        mng = plt.get_current_fig_manager()
        mng.set_window_title('Courbe températures nocturnes')
        mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
        mng.window.state('iconic')
        gtemp.set_xticks(xlabels)
        gtemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
        plt.legend(bbox_to_anchor=(0.354, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 5)
        plt.axhline(0, c='white', lw=1)
        plt.axhline(-2, c='white', lw=1, ls = '--')
        plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
        plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
        plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure, dt.datetime.today().strftime("%H:%M"))
        plt.pause(plt_pause1)
        plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
        plt.clf()
        plt.cla()
        gc.collect()
        time.sleep(sleep1)



cumuldata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'cumul':'min'}).rename(columns={'t' : 't', 'cumul' : 'cumul'})
tempsdata = nuits.groupby([nuits.date, nuits.heure]).agg({'t' : 'min', 'temp':'min'}).rename(columns={'t' : 't', 'temp' : 'temp'})
dates = nuits.groupby(by = 'date').agg({'date' : 'first'}).reset_index(drop=True)
idatescomp = [i for i in range(dates.size)]
datescomp = dates.copy()
listedates = datescomp.date.to_list()
cumulseries = [cumuldata.loc[dates.iloc[i]].reset_index(drop=True).cumul.to_list() for i in idatescomp]
tempseries = [tempsdata.loc[dates.iloc[i]].reset_index(drop=True).temp.to_list() for i in idatescomp]
formatted_cumul = to_time_series_dataset(cumulseries)
formatted_temps = to_time_series_dataset(tempseries)
cumulaujseries = cumuldata.reset_index(drop=True).cumul.to_list()
tempaujseries = tempsdata.reset_index(drop=True).temp.to_list()
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

# Class areas
classlistmm = range(nclasses)
classlist = range(nclasses)
#classlist = [4]

minimaxc = []
for i in classlistmm :
    dfmmc = nuits.loc[nuits['classec'] == i].sort_values(by = 'heure').groupby(pd.Grouper(key = 'heure')).agg({'temp': ['min', 'max'], 'cumul': ['min', 'max']}).dropna()
    minimaxc.append(dfmmc)
    dfmmc.to_csv('C:\\CumulusMX\\webfiles\\images\\mmc' + str(working_year) + str(i) + '.csv', sep = ';', header = None)
minimaxt = []
for i in classlistmm :
    dfmmt = nuits.loc[nuits['classet'] == i].sort_values(by = 'heure').groupby(pd.Grouper(key = 'heure')).agg({'temp': ['min', 'max'], 'cumul': ['min', 'max']}).dropna()
    minimaxt.append(dfmmt)
    dfmmt.to_csv('C:\\CumulusMX\\webfiles\\images\\mmt' + str(working_year) + str(i) + '.csv', sep = ';', header = None)

gcumcum = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None)
for i in classlist :
    gcumcum.fill_between(minimaxc[i].index.to_list(), minimaxc[i]['cumul']['min'].to_list(), minimaxc[i]['cumul']['max'].to_list(), color = sns.color_palette(palette, nclasses)[i], alpha=0.3)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gcumcum.set_xticks(xlabels)
gcumcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.ylim([-1000, 0])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.title('20' + str(working_year) + ' Cumuls - Nuits complètes, coloration par classe de cumul')
plt.pause(plt_pause1)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_cumul_' + str(working_year) + '.png')
plt.clf()
plt.cla()
gc.collect()
time.sleep(sleep2)

gtempcum = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classesc, size = 'date', sizes = sizes_classes, estimator=None)
for i in classlist :
    gtempcum.fill_between(minimaxc[i].index.to_list(), minimaxc[i]['temp']['min'].to_list(), minimaxc[i]['temp']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gtempcum.set_xticks(xlabels)
gtempcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.title('20' + str(working_year) + ' Températures - Nuits complètes, coloration par classe de cumul')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_cumul_temp_' + str(working_year) + '.png')
plt.clf()
plt.cla()
gc.collect()
time.sleep(sleep2)

gcumtemp = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None)
for i in classlist :
    gcumtemp.fill_between(minimaxt[i].index.to_list(), minimaxt[i]['cumul']['min'].to_list(), minimaxt[i]['cumul']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gcumtemp.set_xticks(xlabels)
gcumtemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.05, 0.8), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.ylim([-1000, 0])
plt.title('20' + str(working_year) + ' Cumuls - Nuits complètes, coloration par classe de température')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_cumul_' + str(working_year) + '.png')
plt.clf()
plt.cla()
gc.collect()

gtemptemp = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = couleurs_classest, size = 'date', sizes = sizes_classes, estimator=None)
for i in classlist :
    gtemptemp.fill_between(minimaxt[i].index.to_list(), minimaxt[i]['temp']['min'].to_list(), minimaxt[i]['temp']['max'].to_list(), color = sns.color_palette(palette, nlabels)[i], alpha=0.3)
mng = plt.get_current_fig_manager()
mng.set_window_title('Courbe températures nocturnes')
mng.window.wm_iconbitmap("C:\\data\\NedelecDev\\nbpython38\\suivi_temp.ico")
mng.window.state('iconic')
gtemptemp.set_xticks(xlabels)
gtemptemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
plt.axhline(0, c='white', lw=1)
plt.axhline(-2, c='white', lw=1, ls = '--')
plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
plt.title('20' + str(working_year) + ' Températures - Nuits complètes, coloration par classe de températures')
plt.pause(0.001)
plt.savefig('C:\\CumulusMX\\webfiles\\images\\class_temp_temp_' + str(working_year) + '.png')
plt.clf()
plt.cla()
gc.collect()
time.sleep(sleep2)

