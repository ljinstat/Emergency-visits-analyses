
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LabelKFold

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

dataframe = pd.read_csv("/data/data/Urgences/data/PREDURG_rpu2014-degrade_20160428.csv", sep=";")
dataframe_15 = pd.read_csv("/data/users/ling.jin/PREDURG_rpu2015-degrade_20160613-CLEAN.csv",
                        sep=";")#, index_col=0)


# In[2]:

from pandas.tseries.resample import TimeGrouper


# In[3]:

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9


# In[4]:

from IPython.display import display, HTML


# # Construire la série temporelle

# In[5]:

#Date et heure d’entrée 2015
ENT=dataframe['DHM_ENT'].tolist()
dataframe_or_ent=pd.to_datetime(ENT)
df0 = pd.DataFrame({'Patient':dataframe.ID_RPU,'Date':dataframe_or_ent, 'Sum':1})
#print(df0_shuffle.head(10))


# In[6]:

#Date et heure d’entrée 2015
ENT=dataframe_15['DHM_ENT'].tolist()
dataframe_or_ent_15=pd.to_datetime(ENT)
df0_15 = pd.DataFrame({'Patient':dataframe_15.ID_RPU,'Date':dataframe_or_ent_15, 'Sum':1})
#print(df0.head(10))
#print(df0_shuffle.head(10))


# In[7]:

#supprimer les dates de 2013 et 2015
#df0: les donnees originales
#df0_shuffle: les donnees apres shuffle
#df_day: les donnees par jour
from pandas.tseries.resample import TimeGrouper
df0_shuffle = df0.reindex(np.random.permutation(len(df0)))
df_day = df0_shuffle.set_index('Date').groupby(TimeGrouper('6H')).sum()[1:-1].reset_index()
#print(df_day)


# In[8]:

#supprimer les dates de 2013 et 2015
#df0: les donnees originales
#df0_shuffle: les donnees apres shuffle
#df_day: les donnees par jour
from pandas.tseries.resample import TimeGrouper
df0_shuffle_15 = df0_15.reindex(np.random.permutation(len(df0_15)))
df_day_15 = df0_shuffle_15.set_index('Date').groupby(TimeGrouper('6H')).sum()[1:-1].reset_index()


# In[9]:

#print(df0_shuffle)


# In[10]:

#Il faut set_index()
#print(df_day.dtypes)
#print(df_day.index)


# In[11]:

ts = df_day["Sum"] 


# In[12]:

ts_15 = df_day_15["Sum"] 


# In[13]:

#ts.head(10)


# In[14]:

fig, ax = plt.subplots()
plot2014 = plt.plot(ts, label = '2014')
plot2015 = plt.plot(ts_15, label = '2015')
plt.title('Nombre de passages 2014 et 2015')
plt.legend(loc=0)


# ## Prévoir les statistiques simples
# ### Prévoir nombre moyen de passages de chaque jour 2014

# In[15]:

#2.On prend le nombre moyenne comme la prevision
#(si on separer l'ensemble d'apprantissage et validations par demi donnees-> le nombre est presque un demi de nombre actuel)
#train_day_halfdata->la moyenne de nombre de passage journalier
#validation_day_halfdata-> toutes les donnees par jour

validation_day = df_day.copy()
validation_day['hour'] = validation_day.Date.apply(lambda x: x.hour)


# In[16]:

#la nuit profonde minuit a 6h
sum_nuit_pro_validation = (validation_day['Sum'][validation_day['hour'] == 0]).mean()
print(sum_nuit_pro_validation)
#matinee 7h-12h
sum_matinee_validation = (validation_day['Sum'][validation_day['hour'] == 6]).mean()
print(sum_matinee_validation)
#debut d'apres midi 12h-18h
sum_d_am_validation = (validation_day['Sum'][validation_day['hour'] == 12]).mean()
print(sum_d_am_validation)
#fin d'apres midi 18h-23h
sum_f_am_validation = (validation_day['Sum'][validation_day['hour'] == 18]).mean()
print(sum_f_am_validation)


# In[17]:

train_day = df_day.copy()


# In[18]:

train_day['day_mean'] = 0
train_day['day_mean'][validation_day['hour'] == 0] = sum_nuit_pro_validation
train_day['day_mean'][validation_day['hour'] == 6] = sum_matinee_validation
train_day['day_mean'][validation_day['hour'] == 12] = sum_d_am_validation
train_day['day_mean'][validation_day['hour'] == 18] = sum_f_am_validation


# In[19]:

pre_validation_day = train_day['day_mean']
res_validation_day = pd.DataFrame({'Date':validation_day.Date, 'Sum':abs(validation_day.Sum - train_day.day_mean)})
fig, ax = plt.subplots()
Validation_plt  = plt.plot(validation_day.Sum, color = 'blue', label = 'Validation')
Résidu_plt  = plt.plot(res_validation_day.Sum, color = 'red', label = 'Résidu')
Moyenne_plt  = plt.plot(pre_validation_day, color='green', label = 'Moyenne')
plt.title('Prédiction de nombre moyen de passages par 6 heures et les Résidus 2014')
plt.legend(loc=0)


# In[67]:

pre_validation_day = train_day['day_mean']
res_validation_day = pd.DataFrame({'Date':validation_day.Date, 'Sum':abs(validation_day.Sum - train_day.day_mean)})

fig, ax = plt.subplots()
Validation_plt  = plt.plot(validation_day.Sum[0:28], color = 'blue', label = 'Validation')
Résidu_plt  = plt.plot(res_validation_day.Sum[0:28], color = 'red', label = 'Résidu')
Moyenne_plt  = plt.plot(pre_validation_day[0:28], color='green', label = 'Moyenne')
plt.title('Prédiction de nombre moyen de passages par 6 heures et les Résidus 2014 (la première semaine)')
plt.legend(loc=0)


# In[68]:

#erreur de la validation(supprimer la derniere point)
Erreur_validation = (res_validation_day.Sum)
fig, ax1 = plt.subplots()
Erreur_Validation_plt = plt.plot(Erreur_validation[:-1], color = 'blue')
plt.title("Résidus absolus 2014")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[69]:

type(Erreur_validation)


# In[70]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2014")


# In[72]:

#Les Résidu tres grandes
grande_erreur_day = res_validation_day[res_validation_day.Sum > 1000].dropna()
print(display(grande_erreur_day.sort_values(by = 'Sum').reset_index()))


# In[25]:

res_validation_day['Pourcentage'] = res_validation_day.Sum/df_day.Sum
res_validation_day['total'] = df_day.Sum


# In[26]:

res_validation_day = res_validation_day.reset_index()
grande_erreur_day = grande_erreur_day.reset_index()


# In[27]:

pourcentage_sum_day =  res_validation_day[res_validation_day.Date.isin(grande_erreur_day.Date)]
print(display(pourcentage_sum_day.sort_values(by = 'Sum')))


# ### Prévoir nombre moyen de passages de chaque jour 2015

# In[28]:

#2.On prend le nombre moyenne comme la prevision
#(si on separer l'ensemble d'apprantissage et validations par demi donnees-> le nombre est presque un demi de nombre actuel)
#train_day_halfdata->la moyenne de nombre de passage journalier
#validation_day_halfdata-> toutes les donnees par jour

validation_day_15 = df_day_15.copy()
validation_day_15['hour'] = validation_day_15.Date.apply(lambda x: x.hour)


# In[31]:

#la nuit profonde minuit a 6h
sum_nuit_pro_validation_15 = (validation_day_15['Sum'][validation_day_15['hour'] == 0]).mean()
print(sum_nuit_pro_validation_15)
#matinee 7h-12h
sum_matinee_validation_15 = (validation_day_15['Sum'][validation_day_15['hour'] == 6]).mean()
print(sum_matinee_validation_15)
#debut d'apres midi 12h-18h
sum_d_am_validation_15 = (validation_day_15['Sum'][validation_day_15['hour'] == 12]).mean()
print(sum_d_am_validation)
#fin d'apres midi 18h-23h
sum_f_am_validation_15 = (validation_day_15['Sum'][validation_day_15['hour'] == 18]).mean()
print(sum_f_am_validation_15)


# In[32]:

train_day_15 = df_day_15.copy()


# In[33]:

train_day_15['day_mean'] = 0
train_day_15['day_mean'][validation_day_15['hour'] == 0] = sum_nuit_pro_validation_15
train_day_15['day_mean'][validation_day_15['hour'] == 6] = sum_matinee_validation_15
train_day_15['day_mean'][validation_day_15['hour'] == 12] = sum_d_am_validation_15
train_day_15['day_mean'][validation_day_15['hour'] == 18] = sum_f_am_validation_15


# In[34]:

pre_validation_day_15 = train_day_15.day_mean
res_validation_day_15 = pd.DataFrame({'Date':validation_day_15.Date, 'Sum':abs(validation_day_15.Sum - train_day_15.day_mean)})
fig, ax = plt.subplots()
Validation_plt_15  = plt.plot(validation_day_15.Sum, color = 'blue', label = 'Validation')
Résidu_plt_15  = plt.plot(res_validation_day_15.Sum, color = 'red', label = 'Résidu')
Moyenne_plt_15  =  plt.plot(pre_validation_day_15, color='green', label = 'Moyenne')
plt.title('Prédiction de nombre moyen de passages par 6 heures et les Résidus 2015')
plt.legend(loc=0)


# In[73]:

pre_validation_day_15 = train_day_15.day_mean
res_validation_day_15 = pd.DataFrame({'Date':validation_day_15.Date, 'Sum':abs(validation_day_15.Sum - train_day_15.day_mean)})
fig, ax = plt.subplots()
Validation_plt_15  = plt.plot(validation_day_15.Sum[0:28], color = 'blue', label = 'Validation')
Résidu_plt_15  = plt.plot(res_validation_day_15.Sum[0:28], color = 'red', label = 'Résidu')
Moyenne_plt_15  =  plt.plot(pre_validation_day_15[0:28], color='green', label = 'Moyenne')
plt.title('Prédiction de nombre moyen de passages par 6 heures et les Résidus 2015 (la première semaine)')
plt.legend(loc=0)


# In[74]:

#erreur de la validation(supprimer la derniere point)
Erreur_validation_15 = (res_validation_day_15).Sum
fig, ax1 = plt.subplots()
Erreur_Validation_plt_15 = plt.plot(Erreur_validation_15[:-1], color = 'blue')
plt.title("Résidus absolus 2015")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[75]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation_15, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2015")


# In[76]:

#Les Résidu tres grandes
grande_erreur_day_15 = res_validation_day_15[res_validation_day_15 > 1200].dropna()
print(display(grande_erreur_day_15.sort_values(by = 'Sum').reset_index()))


# In[39]:

res_validation_day_15['Pourcentage'] = res_validation_day_15.Sum/df_day_15.Sum
res_validation_day_15['total'] = df_day_15.Sum


# In[40]:

res_validation_day_15 = res_validation_day_15.reset_index()
grande_erreur_day_15 = grande_erreur_day_15.reset_index()


# In[41]:

pourcentage_sum_day_15 =  res_validation_day_15[res_validation_day_15.Date.isin(grande_erreur_day_15.Date)]
print(display(pourcentage_sum_day_15.sort_values(by = 'Sum')))


# ### Prévoir nombre moyen de passages de chaque jour de la semaine 2014

# In[42]:

sum_p_day = df_day.reset_index()
sum_p_day['Weekday'] = sum_p_day.Date.apply(lambda x: x.weekday())
#[sum_p_day.weekday(d) for d in sum_p_day.index]
lun = sum_p_day[sum_p_day['Weekday'] == 0].mean().Sum
mar = sum_p_day[sum_p_day['Weekday'] == 1].mean().Sum
mer = sum_p_day[sum_p_day['Weekday'] == 2].mean().Sum
jeu = sum_p_day[sum_p_day['Weekday'] == 3].mean().Sum
ven = sum_p_day[sum_p_day['Weekday'] == 4].mean().Sum
sam = sum_p_day[sum_p_day['Weekday'] == 5].mean().Sum
dim = sum_p_day[sum_p_day['Weekday'] == 6].mean().Sum


# In[43]:

#plot Nombre moyen de passage selon le jour de la semaine
#bar plot

semaine_plt = [lun, mar, mer, jeu, ven, sam, dim]
print(semaine_plt)
ax = plt.subplot(111)
ax.grid()
barlist = ax.bar(range(len(semaine_plt)), semaine_plt, width=0.7,align='center',color='blue' )

plt.title("Nombre moyen de passages selon le jour de la semaine 2014")
plt.xlabel(" ")
plt.ylabel("Nombre")

plt.xticks(range(7),['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim'])

plt.legend(loc=1)


plt.show()


# In[44]:

lun_var = np.var(sum_p_day[sum_p_day['Weekday'] == 0].Sum)
mar_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 1].Sum)
mer_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 2].Sum)
jeu_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 3].Sum)
ven_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 4].Sum)
sam_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 5].Sum)
dim_var =  np.var(sum_p_day[sum_p_day['Weekday'] == 6].Sum)


# In[45]:

print(lun_var)


# In[46]:

#plot Nombre moyen de passage selon le jour de la semaine
#bar plot

semaine_plt = [lun, mar, mer, jeu, ven, sam, dim]
semaine_var_plt = [lun_var/100, mar_var/100, mer_var/100, jeu_var/100, ven_var/100, sam_var/100, dim_var/100]
print(semaine_plt)
ax = plt.subplot(111)
ax.grid()
barlist = ax.bar(range(len(semaine_plt)), semaine_plt, width=0.7,align='center',color='blue', label = 'Nombre de passages' )
plt.plot(semaine_var_plt, label = 'Variance/100')

plt.title("Nombre moyen de passages selon le jour de la semaine 2014")
plt.xlabel(" ")
plt.ylabel("Nombre")

plt.xticks(range(7),['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim'])

plt.legend(loc=1)


plt.show()


# In[47]:

#
#(1.séparer l'ensemble d'apprantissage et validations  par demi annee->Il n'est pas correct
train_week = sum_p_day.copy()
validation_week = sum_p_day.copy()


# In[48]:

train_week.head()


# In[49]:

#nombre moyen de passages de chaque jour de la semaine dans l'ensemble d'apprentissage
lun_train = train_week[train_week['Weekday'] == 0].mean().Sum
mar_train = train_week[train_week['Weekday'] == 1].mean().Sum
mer_train = train_week[train_week['Weekday'] == 2].mean().Sum
jeu_train = train_week[train_week['Weekday'] == 3].mean().Sum
ven_train = train_week[train_week['Weekday'] == 4].mean().Sum
sam_train = train_week[train_week['Weekday'] == 5].mean().Sum
dim_train = train_week[train_week['Weekday'] == 6].mean().Sum


# In[50]:

#Ajouter un column de predicion a validation_week
validation_week['prediction'] = 0
validation_week['prediction'][validation_week['Weekday']== 0] = lun
validation_week['prediction'][validation_week['Weekday']== 1] = mar
validation_week['prediction'][validation_week['Weekday']== 2] = mer
validation_week['prediction'][validation_week['Weekday']== 3] = jeu
validation_week['prediction'][validation_week['Weekday']== 4] = ven
validation_week['prediction'][validation_week['Weekday']== 5] = sam
validation_week['prediction'][validation_week['Weekday']== 6] = dim


# In[51]:

pre_validation_week = validation_week['prediction']
res_validation_week = abs(pre_validation_week - validation_week.Sum)
fig, ax = plt.subplots()
Validation_plt_week = plt.plot(validation_week.Sum, color = 'blue', label = 'Validation')
Prediction_plt_week = plt.plot(pre_validation_week, color = 'green', label = 'Prediction')
Résidu_plt_week = plt.plot(res_validation_week, color = 'red', label = 'Résidu')
plt.title('Prédiction de nombre moyen de passages hebdomadaire et les Résidus 2014')
plt.legend(loc=0)


# In[53]:

#erreur de la validation (supprimer la derniere point)
Erreur_validation_week = (res_validation_week)
fig, ax = plt.subplots()
Erreur_Validation_plt_week = plt.plot(Erreur_validation_week[:-1], color = 'blue')
plt.title("Résidus absolus 2014")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[54]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation_week, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2014")


# In[57]:

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res_validation_week.reset_index()
grande_erreur_week = validation_week[res_validation_week > 2000].dropna()
print(display(pd.concat([grande_erreur_week.Date, res_validation_week[res_validation_week > 2000]],axis = 1)))


# In[58]:

print(df_day[df_day.Sum == df_day.Sum.min()])


# In[59]:

print(df_day[df_day.Sum == df_day.Sum.max()])


# ### Prévoir nombre moyen de passages de chaque jour de la semaine 2015

# In[60]:

sum_p_day_15 = df_day_15.reset_index()
sum_p_day_15['Weekday'] = sum_p_day_15.Date.apply(lambda x: x.weekday())
#[sum_p_day.weekday(d) for d in sum_p_day.index]
lun_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 0].mean().Sum
mar_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 1].mean().Sum
mer_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 2].mean().Sum
jeu_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 3].mean().Sum
ven_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 4].mean().Sum
sam_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 5].mean().Sum
dim_15 = sum_p_day_15[sum_p_day_15['Weekday'] == 6].mean().Sum


# In[61]:

#plot Nombre moyen de passage selon le jour de la semaine
#bar plot

semaine_plt_15 = [lun_15, mar_15, mer_15, jeu_15, ven_15, sam_15, dim_15]
print(semaine_plt_15)
ax = plt.subplot(111)
ax.grid()
barlist_15 = ax.bar(range(len(semaine_plt_15)), semaine_plt_15, width=0.7,align='center',color='blue' )

plt.title("Nombre moyen de passages selon le jour de la semaine 2015")
plt.xlabel(" ")
plt.ylabel("Nombre")

plt.xticks(range(7),['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim'])

plt.legend(loc=1)


plt.show()


# In[62]:

lun_var_15 = np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 0].Sum)
mar_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 1].Sum)
mer_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 2].Sum)
jeu_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 3].Sum)
ven_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 4].Sum)
sam_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 5].Sum)
dim_var_15 =  np.var(sum_p_day_15[sum_p_day_15['Weekday'] == 6].Sum)


# In[63]:

#plot Nombre moyen de passage selon le jour de la semaine
#bar plot

semaine_plt_15 = [lun_15, mar_15, mer_15, jeu_15, ven_15, sam_15, dim_15]
semaine_var_plt_15 = [lun_var_15/200, mar_var_15/200, mer_var_15/200, jeu_var_15/200, ven_var_15/200, sam_var_15/200, dim_var_15/200]
print(semaine_plt_15)
ax = plt.subplot(111)
ax.grid()
barlist = ax.bar(range(len(semaine_plt_15)), semaine_plt_15, width=0.7,align='center',color='blue', label = 'Nombre de passages' )
plt.plot(semaine_var_plt_15, label = 'Variance/200')

plt.title("Nombre moyen de passages selon le jour de la semaine 2015")
plt.xlabel(" ")
plt.ylabel("Nombre")

plt.xticks(range(7),['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim'])

plt.legend(loc=1)


plt.show()


# In[64]:

#
#(1.séparer l'ensemble d'apprantissage et validations  par demi annee->Il n'est pas correct
train_week_15 = sum_p_day_15.copy()
validation_week_15_15 = sum_p_day_15.copy()


# In[65]:

#nombre moyen de passages de chaque jour de la semaine dans l'ensemble d'apprentissage
lun_train_15 = train_week_15[train_week_15['Weekday'] == 0].mean().Sum
mar_train_15 = train_week_15[train_week_15['Weekday'] == 1].mean().Sum
mer_train_15 = train_week_15[train_week_15['Weekday'] == 2].mean().Sum
jeu_train_15 = train_week_15[train_week_15['Weekday'] == 3].mean().Sum
ven_train_15 = train_week_15[train_week_15['Weekday'] == 4].mean().Sum
sam_train_15 = train_week_15[train_week_15['Weekday'] == 5].mean().Sum
dim_train_15 = train_week_15[train_week_15['Weekday'] == 6].mean().Sum


# In[66]:

#Ajouter un column de predicion a validation_week_15
validation_week_15['prediction'] = 0
validation_week_15['prediction'][validation_week_15['Weekday']== 0] = lun
validation_week_15['prediction'][validation_week_15['Weekday']== 1] = mar
validation_week_15['prediction'][validation_week_15['Weekday']== 2] = mer
validation_week_15['prediction'][validation_week_15['Weekday']== 3] = jeu
validation_week_15['prediction'][validation_week_15['Weekday']== 4] = ven
validation_week_15['prediction'][validation_week_15['Weekday']== 5] = sam
validation_week_15['prediction'][validation_week_15['Weekday']== 6] = dim


# In[ ]:

pre_validation_week_15 = validation_week_15['prediction']
res_validation_week_15 = abs(pre_validation_week_15 - validation_week_15.Sum)
fig, ax = plt.subplots()
Validation_plt_week_15 = plt.plot(validation_week_15.Sum, color = 'blue', label = 'Validation')
Prediction_plt_week_15 = plt.plot(pre_validation_week_15, color = 'green', label = 'Prediction')
Résidu_plt_week_15 = plt.plot(res_validation_week_15, color = 'red', label = 'Résidu')
plt.title('Prédiction de nombre moyen de passages hebdomadaire et les Résidus 2015')
plt.legend(loc=0)


# In[ ]:

#erreur de la validation (supprimer la derniere point)
Erreur_validation_week_halfdata_15 = (res_validation_week_15)
fig, ax = plt.subplots()
Erreur_Validation_plt_week_halfdata_15 = plt.plot(Erreur_validation_week_halfdata_15[:-1], color = 'blue')
plt.title("Résidus absolus 2015")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[ ]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation_week_halfdata_15, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2015")


# In[ ]:

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res_validation_week_15.reset_index()
grande_erreur_week_15 = validation_week_15[res_validation_week_15 > 1800].dropna()
print(display(pd.concat([grande_erreur_week_15.Date, res_validation_week_15[res_validation_week_15 > 1800]],axis = 1)))


# ### Prévoir nombre moyen de passages par heure 2014

# In[ ]:

#
#(On prend deux demis donnees comme l'apprentissage et la validation! Mais pas la premiere demi annee!->Il n'est pas correct)
train_hour = df0.copy()
validation_hour = df0.copy()


# In[ ]:

#(On prend deux demis donnees comme l'apprentissage et la validation! Mais pas la premiere demi annee!->Il n'est pas correct)
#Train:hourly sum of patients /Évolution du nombre moyen horaire de passages quotidiens 
##on peux juste utiliser TimeGrouper('H')!!!!!T^T Grand difference entre Date.hour(sum par heure total) et TimeGrouper(chaque heure par jour)!!! TimeGrouper est plus pratique
from IPython.display import display, HTML
ts_train=df0.copy().set_index('Date')
train_day=ts_train.groupby(ts_train.index.hour).sum()
#display(train_day)


# In[ ]:

print(train_day)


# In[ ]:

#La moyenne par heure 
mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9, mean_10, mean_11, mean_12, mean_13, mean_14, mean_15, mean_16, mean_17, mean_18, mean_19, mean_20, mean_21, mean_22, mean_23, mean_24  = train_day.Sum/365


# In[ ]:

#la nuit profonde minuit a 7h
sum_nuit_pro = train_day[0:8].sum()
print(sum_nuit_pro)


# In[ ]:

#matinee 8h-11h
sum_matinee = train_day[8:12].sum()
print(sum_matinee)


# In[ ]:

#debut d'apres midi 12h-15h
sum_d_am = train_day[12:16].sum()
print(sum_d_am)


# In[ ]:

#fin d'apres midi 16h-19h
sum_f_am = train_day[16:20].sum()
print(sum_f_am)


# In[ ]:

#nuit 20h-23h
sum_nuit = train_day[20:23].sum()
print(sum_nuit)


# In[ ]:

#la nuit profonde minuit a 7h
mean_nuit_pro = train_day[0:8].sum()/8/365
print(mean_nuit_pro)
#matinee 8h-11h
mean_matinee = train_day[8:12].sum()/4/365
print(mean_matinee)
#debut d'apres midi 12h-15h
mean_d_am = train_day[12:16].sum()/4/365
print(mean_d_am)
#fin d'apres midi 16h-19h
mean_f_am = train_day[16:20].sum()/4/365
print(mean_f_am)
#nuit 20h-23h
mean_nuit = train_day[20:23].sum()/4/365

print(mean_nuit)


# In[ ]:

#Validation on peux juste utiliser TimeGrouper('H')!!!!!T^T Grand difference entre Date.hour et TimeGrouper!!! TimeGrouper est plus pratique
Date = pd.to_datetime(validation_hour['Date'].tolist())
type(Date)
ts_validation=validation_hour.set_index('Date')


# In[ ]:

validation_day=ts_validation.groupby(TimeGrouper('H')).sum()[1:-1].reset_index()
#display(validation_day)


# In[ ]:

#Ajouter un column de l'heure
validation_day['hour'] = validation_day.Date.apply(lambda x: x.hour)
#print(validation_day.hour)


# In[ ]:

#la nuit profonde minuit a 7h
sum_nuit_pro_validation = (validation_day['Sum'][(validation_day['hour'] >= 0) & (validation_day['hour'] <= 7)]).sum()
print(sum_nuit_pro_validation)


# In[ ]:

#matinee 8h-11h
sum_matinee_validation = (validation_day['Sum'][(validation_day['hour'] >= 8) & (validation_day['hour'] <= 11)]).sum()
print(sum_matinee_validation)


# In[ ]:

#debut d'apres midi 12h-15h
sum_d_am_validation = (validation_day['Sum'][(validation_day['hour'] >= 12) & (validation_day['hour'] <= 15)]).sum()
print(sum_d_am_validation)


# In[ ]:

#fin d'apres midi 16h-19h
sum_f_am_validation = (validation_day['Sum'][(validation_day['hour'] >= 16) & (validation_day['hour'] <= 19)]).sum()
print(sum_f_am_validation)


# In[ ]:

#nuit 20h-23h
sum_nuit_validation = (validation_day['Sum'][(validation_day['hour'] >= 20) & (validation_day['hour'] <= 23)]).sum()
print(sum_nuit_validation)


# In[ ]:

train_day['day_mean'] = 0
train_day['day_mean'][(validation_day['hour'] >= 0) & (validation_day['hour'] <= 7)] = mean_nuit_pro.values
train_day['day_mean'][(validation_day['hour'] >= 8) & (validation_day['hour'] <= 11)] = mean_matinee.values
train_day['day_mean'][(validation_day['hour'] >= 12) & (validation_day['hour'] <= 15)] = mean_d_am.values
train_day['day_mean'][(validation_day['hour'] >= 16) & (validation_day['hour'] <= 19)] = mean_f_am.values
train_day['day_mean'][(validation_day['hour'] >= 20) & (validation_day['hour'] <= 23)] = mean_nuit.values


# In[ ]:

#Comparer la validation et la moyenne de chaque echelle
validation_day['day_mean'] = 0
validation_day['day_mean'][validation_day['hour'] == 0] = mean_1
validation_day['day_mean'][validation_day['hour'] == 1] = mean_2   
validation_day['day_mean'][validation_day['hour'] == 2] = mean_3
validation_day['day_mean'][validation_day['hour'] == 3] = mean_4
validation_day['day_mean'][validation_day['hour'] == 4] = mean_5
validation_day['day_mean'][validation_day['hour'] == 5] = mean_6
validation_day['day_mean'][validation_day['hour'] == 6] = mean_7
validation_day['day_mean'][validation_day['hour'] == 7] = mean_8
validation_day['day_mean'][validation_day['hour'] == 8] = mean_9
validation_day['day_mean'][validation_day['hour'] == 9] = mean_10
validation_day['day_mean'][validation_day['hour'] == 10] = mean_11
validation_day['day_mean'][validation_day['hour'] == 11] = mean_12   
validation_day['day_mean'][validation_day['hour'] == 12] = mean_13
validation_day['day_mean'][validation_day['hour'] == 13] = mean_14
validation_day['day_mean'][validation_day['hour'] == 14] = mean_15
validation_day['day_mean'][validation_day['hour'] == 15] = mean_16
validation_day['day_mean'][validation_day['hour'] == 16] = mean_17
validation_day['day_mean'][validation_day['hour'] == 17] = mean_18
validation_day['day_mean'][validation_day['hour'] == 18] = mean_19
validation_day['day_mean'][validation_day['hour'] == 19] = mean_20                               
validation_day['day_mean'][validation_day['hour'] == 20] = mean_21
validation_day['day_mean'][validation_day['hour'] == 21] = mean_22
validation_day['day_mean'][validation_day['hour'] == 22] = mean_23
validation_day['day_mean'][validation_day['hour'] == 23] = mean_24                                  


# In[ ]:

pre_validation_day = validation_day['day_mean']
res_validation_day = abs(pre_validation_day - validation_day.Sum)
fig, ax = plt.subplots()
Validation_plt_hour = plt.plot(validation_day.Sum, color = 'blue', label = 'Validation')
Prediction_plt_hour = plt.plot(pre_validation_day, color = 'green', label = 'Prediction')
Résidu_plt_hour = plt.plot(res_validation_day, color = 'red', label = 'Résidu')
plt.title("Prédiction de nombre moyen de passages par la tranche d'heure et les Résidus")
plt.legend(loc=0)


# In[ ]:

fig, ax = plt.subplots()
Validation_plt_hour = plt.plot(validation_day.Sum[:168], color = 'blue', label = 'Validation')
Prediction_plt_hour = plt.plot(pre_validation_day[:168], color = 'green', label = 'Prediction')
Résidu_plt_hour = plt.plot(res_validation_day[:168], color = 'red', label = 'Résidu')
plt.title("Prédiction de nombre moyen de passages par la tranche d'heure et les Résidus sur la première semaine 2014")
plt.legend(loc=0)


# In[ ]:

#erreur de la validation
Erreur_validation_day = abs(pre_validation_day - validation_day.Sum)
fig, ax = plt.subplots()
Erreur_Validation_plt_hour = plt.plot(Erreur_validation_day, color = 'blue')
plt.title("Résidus absolus 2014")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[ ]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation_day.dropna(), 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2014")


# In[ ]:

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res_validation_week_15.reset_index()
grande_erreur_hour = validation_day[Erreur_validation_day > 275].dropna()
print(display(pd.concat([grande_erreur_hour.Date, Erreur_validation_day[Erreur_validation_day > 275]], axis = 1)))


# ### Prévoir nombre moyen de passages par heure 2015

# In[ ]:

#
#(On prend deux demis donnees comme l'apprentissage et la validation! Mais pas la premiere demi annee!->Il n'est pas correct)
train_hour_15 = df0_15.copy()
validation_hour_15 = df0_15.copy()


# In[ ]:

#(On prend deux demis donnees comme l'apprentissage et la validation! Mais pas la premiere demi annee!->Il n'est pas correct)
#Train:hourly sum of patients /Évolution du nombre moyen horaire de passages quotidiens 
##on peux juste utiliser TimeGrouper('H')!!!!!T^T Grand difference entre Date.hour(sum par heure total) et TimeGrouper(chaque heure par jour)!!! TimeGrouper est plus pratique
from IPython.display import display, HTML
ts_train_15=df0_15.copy().set_index('Date')
train_day_15=ts_train_15.groupby(ts_train_15.index.hour).sum()
#display(train_day)


# In[ ]:

#La moyenne par heure 
mean_1_15, mean_2_15, mean_3_15, mean_4_15, mean_5_15, mean_6_15, mean_7_15, mean_8_15, mean_9_15, mean_10_15, mean_11_15, mean_12_15, mean_13_15, mean_14_15, mean_15_15, mean_16_15, mean_17_15, mean_18_15, mean_19_15, mean_20_15, mean_21_15, mean_22_15, mean_23_15, mean_24_15  = train_day_15.Sum/365


# In[ ]:

#la nuit profonde minuit a 7h
mean_nuit_pro_15 = train_day_15[0:8].sum()/8/365
print(mean_nuit_pro_15)
#matinee 8h-11h
mean_matinee_15 = train_day_15[8:12].sum()/4/365
print(mean_matinee_15)
#debut d'apres midi 12h-15h
mean_d_am_15 = train_day_15[12:16].sum()/4/365
print(mean_d_am_15)
#fin d'apres midi 16h-19h
mean_f_am_15 = train_day_15[16:20].sum()/4/365
print(mean_f_am_15)
#nuit 20h-23h
mean_nuit_15 = train_day_15[20:23].sum()/4/365

print(mean_nuit_15)


# In[ ]:

#Validation on peux juste utiliser TimeGrouper('H')!!!!!T^T Grand difference entre Date.hour et TimeGrouper!!! TimeGrouper est plus pratique
Date_15 = pd.to_datetime(validation_hour_15['Date'].tolist())
type(Date_15)
ts_validation_15=validation_hour_15.set_index('Date')


# In[ ]:

validation_day_15=ts_validation_15.groupby(TimeGrouper('H')).sum()[1:-1].reset_index()
#display(validation_day)#nuit 20h-23h
#Ajouter un column de l'heure
validation_day_15['hour'] = validation_day_15.Date.apply(lambda x: x.hour)
#print(validation_day.hour)
sum_nuit_validation_15 = (validation_day_15['Sum'][(validation_day_15['hour'] >= 20) & (validation_day_15['hour'] <= 23)]).sum()
print(sum_nuit_validation_15)


# In[ ]:

#Ajouter un column de l'heure
validation_day_15['hour'] = validation_day_15.Date.apply(lambda x: x.hour)
#print(validation_day.hour)


# In[ ]:

#la nuit profonde minuit a 7h
sum_nuit_pro_validation_15 = (validation_day_15['Sum'][(validation_day_15['hour'] >= 0) & (validation_day_15['hour'] <= 7)]).sum()
print(sum_nuit_pro_validation_15)
#matinee 8h-11h
sum_matinee_validation_15 = (validation_day_15['Sum'][(validation_day_15['hour'] >= 8) & (validation_day_15['hour'] <= 11)]).sum()
print(sum_matinee_validation_15)
#debut d'apres midi 12h-15h
sum_d_am_validation_15 = (validation_day_15['Sum'][(validation_day_15['hour'] >= 12) & (validation_day_15['hour'] <= 15)]).sum()
print(sum_d_am_validation_15)
#fin d'apres midi 16h-19h
sum_f_am_validation_15 = (validation_day_15['Sum'][(validation_day_15['hour'] >= 16) & (validation_day_15['hour'] <= 19)]).sum()
print(sum_f_am_validation_15)


# In[ ]:

train_day_15['day_mean'] = 0
train_day_15['day_mean'][(validation_day_15['hour'] >= 0) & (validation_day_15['hour'] <= 7)] = mean_nuit_pro_15.values
train_day_15['day_mean'][(validation_day_15['hour'] >= 8) & (validation_day_15['hour'] <= 11)] = mean_matinee_15.values
train_day_15['day_mean'][(validation_day_15['hour'] >= 12) & (validation_day_15['hour'] <= 15)] = mean_d_am_15.values
train_day_15['day_mean'][(validation_day_15['hour'] >= 16) & (validation_day_15['hour'] <= 19)] = mean_f_am_15.values
train_day_15['day_mean'][(validation_day_15['hour'] >= 20) & (validation_day_15['hour'] <= 23)] = mean_nuit_15.values


# In[ ]:

#Comparer la validation et la moyenne de chaque tranche
validation_day_15['day_mean'] = 0
validation_day_15['day_mean'][validation_day_15['hour'] == 0] = mean_1_15
validation_day_15['day_mean'][validation_day_15['hour'] == 1] = mean_2_15  
validation_day_15['day_mean'][validation_day_15['hour'] == 2] = mean_3_15
validation_day_15['day_mean'][validation_day_15['hour'] == 3] = mean_4_15
validation_day_15['day_mean'][validation_day_15['hour'] == 4] = mean_5_15
validation_day_15['day_mean'][validation_day_15['hour'] == 5] = mean_6_15
validation_day_15['day_mean'][validation_day_15['hour'] == 6] = mean_7_15
validation_day_15['day_mean'][validation_day_15['hour'] == 7] = mean_8_15
validation_day_15['day_mean'][validation_day_15['hour'] == 8] = mean_9_15
validation_day_15['day_mean'][validation_day_15['hour'] == 9] = mean_10_15
validation_day_15['day_mean'][validation_day_15['hour'] == 10] = mean_11_15
validation_day_15['day_mean'][validation_day_15['hour'] == 11] = mean_12_15
validation_day_15['day_mean'][validation_day_15['hour'] == 12] = mean_13_15
validation_day_15['day_mean'][validation_day_15['hour'] == 13] = mean_14_15
validation_day_15['day_mean'][validation_day_15['hour'] == 14] = mean_15_15
validation_day_15['day_mean'][validation_day_15['hour'] == 15] = mean_16_15
validation_day_15['day_mean'][validation_day_15['hour'] == 16] = mean_17_15
validation_day_15['day_mean'][validation_day_15['hour'] == 17] = mean_18_15
validation_day_15['day_mean'][validation_day_15['hour'] == 18] = mean_19_15
validation_day_15['day_mean'][validation_day_15['hour'] == 19] = mean_20_15                            
validation_day_15['day_mean'][validation_day_15['hour'] == 20] = mean_21_15
validation_day_15['day_mean'][validation_day_15['hour'] == 21] = mean_22_15
validation_day_15['day_mean'][validation_day_15['hour'] == 22] = mean_23_15
validation_day_15['day_mean'][validation_day_15['hour'] == 23] = mean_24_15                               



# In[ ]:

pre_validation_day_15 = validation_day_15['day_mean']
res_validation_day_15 = abs(pre_validation_day_15 - validation_day_15.Sum)
fig, ax = plt.subplots()
Validation_plt_hour_15 = plt.plot(validation_day_15.Sum, color = 'blue', label = 'Validation')
Prediction_plt_hour_15 = plt.plot(pre_validation_day_15, color = 'green', label = 'Prediction')
Résidu_plt_hour_15 = plt.plot(res_validation_day_15, color = 'red', label = 'Résidu')
plt.title("Prédiction de nombre moyen de passages par la tranche d'heure et les Résidus 2015")
plt.legend(loc=0)


# In[ ]:

fig, ax = plt.subplots()
Validation_plt_hour_15 = plt.plot(validation_day_15.Sum[:168], color = 'blue', label = 'Validation')
Prediction_plt_hour_15 = plt.plot(pre_validation_day_15[:168], color = 'green', label = 'Prediction')
Résidu_plt_hour_15 = plt.plot(res_validation_day_15[:168], color = 'red', label = 'Résidu')
plt.title("Prédiction de nombre moyen de passages par la tranche d'heure et les Résidus sur la première semaine 2015")
plt.legend(loc=0)


# In[ ]:

#erreur de la validation
Erreur_validation_day_15 = abs(pre_validation_day_15 - validation_day_15.Sum)
fig, ax = plt.subplots()
Erreur_Validation_plt_hour_15 = plt.plot(Erreur_validation_day_15, color = 'blue')
plt.title("Résidus absolus 2015")

#la moyenne de la premiere demie annee ne correspondante pas bien surtout l'hiver


# In[ ]:

import matplotlib.mlab as mlab
# the histogram of the data
n, bins, patches = plt.hist(Erreur_validation_day_15.dropna(), 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus 2015")


# In[ ]:

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
res_validation_week_15.reset_index()
grande_erreur_hour_15 = validation_day_15[Erreur_validation_day_15 > 275].dropna()
print(display(pd.concat([grande_erreur_hour_15.Date, Erreur_validation_day_15[Erreur_validation_day_15 > 275]], axis = 1)))


# ### Supprimer les sommets de lundi

# In[ ]:

sum_p_day_pas_lun = sum_p_day[~(sum_p_day['Weekday'] == 0)]


# In[ ]:

print(np.std(sum_p_day))
print(np.std(sum_p_day_pas_lun))


# ### Analyser les autres sommets
# #### On veut trouver les autres sommets sauf lundi, donc on va faire une figure sans lundi selon les fériés.

# In[ ]:

#Les jours de fériés (inclutant dimanche)
typ_jour=dataframe[(dataframe.TYP_JOU == 3)]#\|(dataframe.TYP_JOU == 3)]


# In[ ]:

typ_jour_ent=typ_jour['DHM_ENT'].order()
typ_jour_ent=pd.to_datetime(typ_jour_ent)

df1 = pd.DataFrame({'Date':typ_jour_ent, 'Sum':1})
#print(df.head())


# In[ ]:

#groupby chaque jour: sum_p_day_ferie
df_day_ferie = df1.sort('Date')[8:-6].set_index('Date').groupby(TimeGrouper('D')).sum()
sum_p_day_ferie = df_day_ferie.reset_index()
sum_p_day_ferie['Weekday'] = sum_p_day_ferie.Date.apply(lambda x: x.weekday())


# In[ ]:

#On veut trouver les autres sommets sauf lundi, donc on va faire une figure sans lundi selon les fériés
sum_p_day_pas_lun = sum_p_day[~(sum_p_day['Weekday'] == 0)]


# In[ ]:

#les fériés qui ne sont pas lundi
sum_p_day_ferie_pas_lun = sum_p_day_ferie[~(sum_p_day_ferie['Weekday'] == 0)]
sum_p_day_ferie_pas_lun = sum_p_day_ferie_pas_lun[sum_p_day_ferie_pas_lun['Sum'] >= 50]
print(sum_p_day_ferie_pas_lun[~(sum_p_day_ferie_pas_lun['Weekday'] == 6)])


# In[ ]:

day = sum_p_day_pas_lun.reset_index() 
day['position'] = range(0,313)
jour_ferie_pas_lun = day[day['Date'].isin(sum_p_day_ferie_pas_lun['Date'])].position


# In[ ]:

fig, ax = plt.subplots()
    
ax.plot(sum_p_day_pas_lun.Sum.reset_index())
plt.xlabel('Jour')
plt.ylabel('Nombre de passages')
plt.title('Nombre journalier de passages selon les fériés ')
plt.axis([1,330,5500,10000])
jour_ferie_pas_lun.apply(lambda x: ax.axvline(x, color='red', linestyle='--',linewidth=2))
fig.set_figwidth(35)


# In[ ]:

#nombre de passages dans les jours de fériés sont plus petit 
sum_p_day_ferie_pas_lun.mean()-sum_p_day_pas_lun.mean()


# ### how to find out other peaks?

# In[ ]:

"""Detect peaks in data based on their amplitude and other features."""

from __future__ import division, print_function
import numpy as np

__author__ = "Marcos Duarte, https://github.com/demotu/BMC"
__version__ = "1.0.4"
__license__ = "MIT"


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
from detect_peaks import detect_peaks
x = np.random.randn(100)
x[60:81] = np.nan
# detect all peaks and plot data
ind = detect_peaks(x, show=True)
print(ind)

x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
# set minimum peak height = 0 and minimum peak distance = 20
detect_peaks(x, mph=0, mpd=20, show=True)

x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
# set minimum peak distance = 2
detect_peaks(x, mpd=2, show=True)

x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
# detection of valleys instead of peaks
detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

x = [0, 1, 1, 0, 1, 1, 0]
# detect both edges
detect_peaks(x, edge='both', show=True)

x = [-2, 1, -2, 2, 1, 1, 3, 0]
# set threshold = 2
detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)                     & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        #ax.set_title("Deuxième détection")
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()


# In[ ]:

import sys
sys.path.insert(1, r'./../functions')  # add to pythonpath


x = np.random.randn(100)
x[60:81] = np.nan
# detect all peaks and plot data
ind = detect_peaks(x, show=True)
print(ind)

x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
# set minimum peak height = 0 and minimum peak distance = 20
detect_peaks(x, mph=0, mpd=20, show=True)

x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
# set minimum peak distance = 2
detect_peaks(x, mpd=2, show=True)

x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
# detection of valleys instead of peaks
detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

x = [0, 1, 1, 0, 1, 1, 0]
# detect both edges
detect_peaks(x, edge='both', show=True)

x = [-2, 1, -2, 2, 1, 1, 3, 0]
# set threshold = 2
detect_peaks(x, threshold = 2, show=True)


# In[ ]:

len(sum_p_day_pas_lun[sum_p_day_pas_lun.isnull()])


# In[ ]:

#plot Nombre moyen de passage selon le jour de la semaine
#bar plot

semaine_plt = [lun, mar, mer, jeu, ven, sam, dim]
print(semaine_plt)
ax = plt.subplot(111)
ax.grid()
barlist = ax.bar(range(len(semaine_plt)), semaine_plt, width=0.7,align='center',color='blue' )
                
plt.title("Nombre moyen de passages selon le jour de la semaine")
plt.xlabel(" ")
plt.ylabel("Nombre")

plt.xticks(range(7),['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim'])

plt.legend(loc=1)


plt.show()


# In[ ]:

ind_lun = detect_peaks(sum_p_day.Sum, mph=0, mpd=5 ,edge='both',show=True)
print(ind_lun)


# In[ ]:

#Peak: lundis, lendemain de Lundi de Paque(22/4), lendemain de la fete nationale(15/7), lendemain de Neol(26/12)
print(sum_p_day.ix[ind_lun])


# In[ ]:

print(sum_p_day['Weekday'].ix[ind_lun].value_counts(normalize = True))


# In[ ]:

#ind_lun = ind_lun.tolist()drop(df.index[[1,3]])!! delete rows using index
sum_p_day_supprim_peak1 = sum_p_day.drop(sum_p_day.index[ind_lun]).reset_index()


# In[ ]:

ind_supprim_peak1 = detect_peaks(sum_p_day_supprim_peak1.Sum, mph=0, mpd=5 ,edge='both',show=True)
print(ind_supprim_peak1)


# In[ ]:

print(sum_p_day)


# In[ ]:

print(sum_p_day_supprim_peak1.ix[ind_supprim_peak1])


# In[ ]:

print(sum_p_day_supprim_peak1['Weekday'].ix[ind_supprim_peak1].value_counts(normalize = True))


# In[ ]:

sum_p_day_supprim_peak2 = sum_p_day_supprim_peak1.ix[ind_supprim_peak1].drop(sum_p_day_supprim_peak1.index[[104,110,128,137,1,165,191,261,270,311,95]]) 
print(sum_p_day_supprim_peak2)


# In[ ]:

display(sum_p_day_supprim_peak2['Weekday'].ix[ind_supprim_peak1].value_counts(normalize = True))


# ### how to find out other peaks? 2015

# In[ ]:

sum_p_day_15.head(20)


# In[ ]:

ind_lun_15 = detect_peaks(sum_p_day_15.Sum, mph=0, mpd=5 ,edge='both',show=True)
print(ind_lun_15)


# In[ ]:

#Peak: lundis, lendemain de Lundi de Paque(22/4), lendemain de la fete nationale(15/7), lendemain de Neol(26/12)
print(sum_p_day_15.ix[ind_lun_15])
print()
print(sum_p_day_15.ix[ind_lun_15].Weekday.value_counts(normalize = True))


# In[ ]:

#ind_lun = ind_lun.tolist()drop(df.index[[1,3]])!! delete rows using index
sum_p_day_supprim_peak1_15 = sum_p_day_15.drop(sum_p_day_15.index[ind_lun_15]).reset_index()


# In[ ]:

ind_supprim_peak1_15 = detect_peaks(sum_p_day_supprim_peak1_15.Sum, mph=0, mpd=5 ,edge='both',show=True)
print(ind_supprim_peak1_15)


# In[ ]:

print(sum_p_day_supprim_peak1_15.ix[ind_supprim_peak1_15])


# In[ ]:


print(sum_p_day_supprim_peak1_15['Weekday'].ix[ind_supprim_peak1_15].value_counts(normalize = True))

