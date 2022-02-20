#!/usr/bin/env python
# coding: utf-8

# Creation de graphes colores chronologiquement
# Superposition des classes de temperatures de l'an passe, classement par rapport aux cumuls
# Les graphes sont enregistres pour etre transferes sur une page web
# Deux grandeurs sont tracees : temperature exterieure et cumul des temperatures negatives

import sys
import gc

import pandas as pd
import numpy as np

import datetime as dt
import time
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import seaborn as sns

autostop = 15.
df_stop = pd.read_csv("C:\\CumulusMX\\web\\realtimewikiT.txttmp", sep = ';', index_col=False)
if (float(df_stop.temp[0].replace(',', '.')) > autostop) :
    sys.exit()

seuilh = 0.5
seuilb = -7.

pos_heure = 20.
pos_heure_c = 35.

looprange = 390

sleep1 = 180.
#sleep1 = 5.
sleep2 = 2.
plt_pause1 = 0.001

nclasses = 5

#palette = 'Set1'
palette = "coolwarm"

current_month = int(dt.date.today().strftime('%m'))
working_year = int((dt.date.today() + relativedelta(months=3)).strftime('%y')) - 1

num_month = [10, 11, 12, 1, 2, 3, 4]
offset_month = [0, 0, 0, 1, 1, 1, 1]
count_month = num_month.index(current_month)
pref_month = ['oct', 'nov', 'déc', 'janv', 'févr', 'mar', 'avr']
month_dataframe = ['df_oct', 'df_nov', 'df_dec', 'df_jan', 'df_fev', 'df_mar', 'df_avr']

i = 0
while i  < count_month :
    globals()[month_dataframe[i]] = pd.read_csv('C:\\CumulusMX\\data\\' + pref_month[i] + str(working_year + offset_month[i]) + 'log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
    i += 1

mmc = []
min_classe = []
#mmt = []
for i in range(nclasses) :
    dfmmc = pd.read_csv('C:\\CumulusMX\\webfiles\\images\\mmc' + str(i) + '.csv', sep = ';', header = None, names = ['t', 'tmin', 'tmax', 'cmin', 'cmax'], parse_dates = ['t'])
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

xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
    
# df = globals()[month_dataframe[0]]
# i = 1
# while i  <= count_month :
#     df = pd.concat([df, globals()[month_dataframe[i]]], ignore_index=True)
#     i += 1
# df.drop(np.arange(3, 28), axis = 1, inplace = True)
# df['t'] = df[0] + ' ' + df[1]
# df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
# df.drop([0, 1], axis = 1, inplace = True)
# df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
# df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
# df.rename(columns={2 : 'temp'}, inplace = True)
# df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
# df = df.loc[(df['heure'] <= dt.datetime.strptime('02/01/1900 12:00', '%d/%m/%Y %H:%M'))]
# df['cumul'] = [min(x, 0) for x in df.temp]
# df['cumul'] = df['cumul'].cumsum()

# for name, group in df.groupby('date'):
#     df.loc[df['date'] == name, ['cumul']] = df.loc[df['date'] == name, ['cumul']] - float(group.head(1).cumul)
# nuits = df.groupby(by=['date']).filter(lambda x: (x['temp'].min() < seuilh and x['temp'].min() > seuilb) or x['t'].min().strftime("%d/%m/%Y") == (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%d/%m/%Y"))

# ndate = nuits.groupby(by=['date']).ngroups
# chron_palette = sns.mpl_palette("viridis", n_colors = ndate - 1)
# chron_palette.append((1., 0.5, 0.05))

# g = sns.lineplot(x = 'heure', y = 'cumul', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
# for i in range(nclasses) :
#     g.fill_between(mmc[i]['t'].to_list(), mmc[i]['cmin'].to_list(), mmc[i]['cmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
# mng = plt.get_current_fig_manager()
# mng.set_window_title('Courbe températures nocturnes')
# mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
# mng.window.state('iconic')
# xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00'), pd.Timestamp('01/02/1900 12:00')]
# g.set_xticks(xlabels)
# g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
# plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
# plt.axhline(0, c='white', lw=1)
# plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
# plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
# plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure_c, dt.datetime.today().strftime("%H:%M"))
# plt.pause(plt_pause1)
# plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp_cumul.png')
# plt.clf()
# plt.cla()
# gc.collect()

# g = sns.lineplot(x = 'heure', y = 'temp', data = nuits, hue = 'date', palette = chron_palette, estimator=None)
# for i in range(nclasses) :
#     g.fill_between(mmc[i]['t'].to_list(), mmc[i]['tmin'].to_list(), mmc[i]['tmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
# mng = plt.get_current_fig_manager()
# mng.set_window_title('Courbe températures nocturnes')
# mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
# mng.window.state('iconic')
# #xlabels = [pd.Timestamp('01/01/1900 18:00'), pd.Timestamp('01/01/1900 20:00'), pd.Timestamp('01/01/1900 22:00'), pd.Timestamp('01/02/1900 00:00'), pd.Timestamp('01/02/1900 02:00'), pd.Timestamp('01/02/1900 04:00'), pd.Timestamp('01/02/1900 06:00'), pd.Timestamp('01/02/1900 08:00'), pd.Timestamp('01/02/1900 10:00')]
# g.set_xticks(xlabels)
# g.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
# plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
# plt.axhline(0, c='white', lw=1)
# plt.axhline(-2, c='white', lw=1, ls = '--')
# plt.axvline(pd.Timestamp('01/02/1900 00:00'), c='white', lw=1)
# plt.axvline(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), c='grey', lw=1, ls = '--')
# plt.text(dt.datetime.strptime('01/01/1900 ' + (dt.datetime.today() + dt.timedelta(hours=-18, minutes=0, seconds=0)).strftime("%H:%M"), '%d/%m/%Y %H:%M') + dt.timedelta(hours=18, minutes=0, seconds=0), pos_heure, dt.datetime.today().strftime("%H:%M"))
# plt.pause(plt_pause1)
# plt.savefig('C:\\CumulusMX\\webfiles\\images\\suivi_temp.png')
# plt.clf()
# plt.cla()
# gc.collect()
df0 = globals()[month_dataframe[0]]
j = 1
while j  < count_month :
    df0 = pd.concat([df0, globals()[month_dataframe[j]]], ignore_index=True)
    j += 1


for i in range(looprange) :
    globals()[month_dataframe[count_month]] = pd.read_csv('C:\\CumulusMX\\data\\' + pref_month[count_month] + str(working_year + offset_month[count_month]) + 'log.txt', sep = ';', header=None, index_col=False, names = np.arange(0, 28))
    df = pd.concat([df0, globals()[month_dataframe[count_month]]], ignore_index=True)
    df.drop(np.arange(3, 28), axis = 1, inplace = True)
    df['t'] = df[0] + ' ' + df[1]
    df['t'] = df['t'].apply(lambda x : dt.datetime.strptime(x, '%d/%m/%y %H:%M') - dt.timedelta(hours=18, minutes=0, seconds=0))
    df.drop([0, 1], axis = 1, inplace = True)
    df['date'] = df['t'].apply(lambda x : dt.datetime.strftime(x, '%d/%m/%y'))
    df['heure'] = df['t'].apply(lambda x : dt.datetime.combine(dt.date(1900, 1, 1), x.time()) + dt.timedelta(hours=18, minutes=0, seconds=0))
    df.rename(columns={2 : 'temp'}, inplace = True)
    df['temp'] = df['temp'].apply(lambda x : float(x.replace(',', '.')))
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
    for i in range(nclasses) :
        gcum.fill_between(mmc[i]['t'].to_list(), mmc[i]['cmin'].to_list(), mmc[i]['cmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
    mng = plt.get_current_fig_manager()
    mng.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    gcum.set_xticks(xlabels)
    gcum.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.05, 0.7), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 2)
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
    for i in range(nclasses) :
        gtemp.fill_between(mmc[i]['t'].to_list(), mmc[i]['tmin'].to_list(), mmc[i]['tmax'].to_list(), linewidth = nclasses - rang_classe[i], color = sns.color_palette(palette, nclasses)[rang_classe[i]], alpha=0.2)
    mng = plt.get_current_fig_manager()
    mng.set_window_title('Courbe températures nocturnes')
    mng.window.wm_iconbitmap("D:\\NedelecDev\\nbpython38\\suivi_temp.ico")
    mng.window.state('iconic')
    gtemp.set_xticks(xlabels)
    gtemp.set_xticklabels([d.strftime('%H:%M') for d in xlabels])
    plt.legend(bbox_to_anchor=(0.394, 1.), loc=2, edgecolor = None, facecolor = 'black', fancybox = 0, framealpha = 0, labelcolor='white', ncol = 4)
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

