
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pandas.tseries.resample import TimeGrouper
import math
dataframe = pd.read_csv("/data/data/Urgences/data/PREDURG_rpu2014-degrade_20160428.csv", sep=";")


# In[2]:

dataframe.columns


# In[3]:

#Date et heure d’entrée
ENT=dataframe['DHM_ENT'].tolist()


# In[4]:

dataframe_or_ent=pd.to_datetime(ENT)


# In[5]:

df = pd.DataFrame({'Patient':dataframe.ID_RPU,'Date':dataframe_or_ent, 'Sum':1})


# In[6]:

df_day_hour = df.set_index('Date').groupby(TimeGrouper('H')).sum()[1:-1].reset_index()


# In[7]:

df_day_6hours = df.set_index('Date').groupby(TimeGrouper('6H')).sum()[1:-1].reset_index()


# In[8]:

print(df_day_6hours.head(24))


# ### Détection de value manquante

# In[9]:

#En utilisant moyenne de vosins de la value manquante
indexnan = df_day_hour.index[df_day_hour['Sum'].apply(np.isnan)]
df_day_hour.fillna(np.mean([df_day_hour.Sum.ix[indexnan-1].values,df_day_hour.Sum.ix[indexnan+1].values]), inplace = True)


# In[10]:

print(df_day_hour.ix[2114])


# ### Des données d'un jour avant

# 1.Par heure

# In[11]:

#df_day_24avant = df_day_hour.Sum va changer Sum!!!Utilise .copy et deep = True!!!!!
df_day_24avant = df_day_hour.Sum.copy(deep = True)
#print(df_day_24avant.head(48))
df_day_24avant[24:len(df_day_24avant)] = df_day_24avant[0:len(df_day_24avant)-24]
#print(df_day_hour.head(48))


# In[12]:

df_day_hour['24avant'] = df_day_24avant


# 2.Par 6 hours

# In[13]:

#df_day_24avant = df_day_hour.Sum va changer Sum!!!Utilise .copy et deep = True!!!!!
df_day_6hours_24avant = df_day_6hours.Sum.copy(deep = True)
#print(df_day_24avant.head(48))
df_day_6hours_24avant[4:len(df_day_6hours_24avant)] = df_day_6hours_24avant[0:len(df_day_6hours_24avant)-4]
#print(df_day_6hours_24avant.head(48))


# In[14]:

df_day_6hours['24avant'] = df_day_6hours_24avant


# ### Des données d'une semaine avant

# 1.Par heure

# In[15]:

df_day_724avant = df_day_hour.Sum.copy(deep = True)
df_day_724avant[168:len(df_day_hour)] = df_day_hour.Sum[0:len(df_day_hour)-168]


# In[16]:

df_day_724avant.fillna(0, inplace = True)


# In[17]:

df_day_hour['724avant'] = df_day_724avant


# 2.Par 6 heures

# In[18]:

df_day_6hours_724avant = df_day_6hours.Sum.copy(deep = True)
df_day_6hours_724avant[28:len(df_day_6hours)] = df_day_6hours.Sum[0:len(df_day_6hours)-28]


# In[19]:

df_day_6hours_724avant.fillna(0, inplace = True)


# In[20]:

df_day_6hours['724avant'] = df_day_6hours_724avant


# ### Construire dataframe par les variables de temps

# 1.Par heure

# In[21]:

df_day_hour['month'] = df_day_hour.Date.apply(lambda x: x.month)


# In[22]:

df_day_hour['day'] = df_day_hour.Date.apply(lambda x: x.day)


# In[23]:

df_day_hour['weekday'] = df_day_hour.Date.apply(lambda x: x.weekday())


# In[24]:

df_day_hour['hour'] = df_day_hour.Date.apply(lambda x: x.hour)


# In[25]:

df_day_hour['Sum'] = df_day_hour.Sum.fillna(0)


# 2.Par 6 heures

# In[26]:

df_day_6hours['month'] = df_day_6hours.Date.apply(lambda x: x.month)


# In[27]:

df_day_6hours['day'] = df_day_6hours.Date.apply(lambda x: x.day)


# In[28]:

df_day_6hours['weekday'] = df_day_6hours.Date.apply(lambda x: x.weekday())


# In[29]:

df_day_6hours['hour'] = df_day_6hours.Date.apply(lambda x: x.hour)


# In[30]:

df_day_6hours['Sum'] = df_day_6hours.Sum.fillna(0)


# ### Données de météo

# In[31]:

#La temperature de Paris
data = pd.read_csv("tem_preci_meteo.csv")


# In[32]:

print(data.head())


# 1.Par heure

# In[33]:

df_day_hour['temperature'] = data.temperature
df_day_hour['precipIntensity'] = data.precipIntensity
df_day_hour['precipProbability'] = data.precipProbability


# In[34]:

print(max(data.temperature),'\n',min(data.temperature))
print(np.var(data.temperature))
print(np.mean(data.temperature))


# In[35]:

print(df_day_hour.head(10))


# In[36]:

print(len(data.precipIntensity[data.precipIntensity == 0])/len(data.precipIntensity))
print(len(data.precipProbability[data.precipProbability == 0])/len(data.precipProbability))


# 2.Par 6 heures

# In[37]:

#Données de météos sont les moyennes de 6 heures
date_6hours = data['Unnamed: 0'].tolist()
date_6hours = pd.to_datetime(date_6hours)
data_6hours = pd.DataFrame({'Date':date_6hours,'precipIntensity':data.precipIntensity, 'precipProbability':data.precipProbability, 'temperature': data.temperature})


# In[38]:

data_6hours = data_6hours.set_index('Date').groupby(TimeGrouper('6H')).mean().reset_index()


# In[39]:

print(data_6hours)


# In[40]:

df_day_6hours['temperature'] = data_6hours.temperature
df_day_6hours['precipIntensity'] = data_6hours.precipIntensity
df_day_6hours['precipProbability'] = data_6hours.precipProbability


# In[41]:

print(df_day_6hours.head)


# ### Variable:  Toy Localisation d'une heure dans l'année

# 1.Par heure

# In[42]:


#def toy(dataframe):
toy = []
#for t in range(1,len(dataframe)):
#    toy.append(t/len(dataframe))
        
#toy_1 = toy#(df_day_hour)   
#print(toy_1)
toy = [t/len(df_day_hour) for t in range(0,len(df_day_hour))]


# In[43]:

len(toy)


# In[44]:

df_day_hour['Toy'] = toy


# 2.Par 6 heures

# In[45]:

toy_6hours = []
toy_6hours = [t/len(df_day_6hours) for t in range(0,len(df_day_6hours))]


# In[46]:

df_day_6hours['Toy'] = toy_6hours


# ###  Figure de relation entre les variables et le réponse

# 1.Par heure

# In[47]:

#Partager le même yaxis
plt.subplots()#,sharey = True)
plt.plot(df_day_hour.Sum)
plt.plot(df_day_hour.temperature*10)
plt.title('Figure de nombre moyen de passages par heure et la température')


# In[48]:

df_day = df_day_hour.set_index('Date').groupby(TimeGrouper('D')).sum()[1:-1].reset_index()
df_day_mean = pd.DataFrame({'Date': df_day.Date, 'Sum': df_day.Sum/24, 'temperature': df_day.temperature/24}, columns  = ['Date','Sum','temperature'])


# In[49]:

print(df_day_mean)


# In[50]:

import matplotlib.mlab as mlab
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9
#Partager le même yaxis
plt.subplots()#,sharey = True)
plt.plot(df_day_mean.Date,df_day_mean.Sum, label = 'Nombre moyen de passages par heure')
plt.plot(df_day_mean.Date,df_day_mean.temperature*10, label = 'Température')
plt.title('Figure de nombre moyen de passages par jour et la température')
plt.legend(loc = 0)


# 2.Par 6 heures

# In[51]:

#Partager le même yaxis
plt.subplots()#,sharey = True)
plt.plot(df_day_6hours.Sum)
plt.plot(df_day_6hours.temperature*100)
plt.title('Figure de nombre moyen de passages par 6 heures et la température')


# In[52]:

df_day_6hours_mean = df_day_6hours.set_index('Date').groupby(TimeGrouper('D')).sum().reset_index()
df_day_mean_6hours = pd.DataFrame({'Date': df_day_6hours_mean.Date, 'Sum': df_day_6hours_mean.Sum/24, 'temperature': df_day_6hours_mean.temperature/24}, columns  = ['Date','Sum','temperature'])


# In[53]:

#Partager le même yaxis
plt.subplots()#,sharey = True)
plt.plot(df_day_mean_6hours.Date,df_day_mean_6hours.Sum, label = 'Nombre moyen de passages par heure')
plt.plot(df_day_mean_6hours.Date,df_day_mean_6hours.temperature*50, label = 'Température')
plt.title('Figure de nombre moyen de passages par jour et la température')
plt.legend(loc = 0)


# # Multiple linear regression
# ### Prévoir nombre moyen de passages de chaque heure

# In[54]:

from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[55]:

#resampling de donnees
df_day_hour_shuffle = df_day_hour.iloc[np.random.permutation(len(df_day_hour))].reset_index(drop = True)
df_day_6hours_shuffle = df_day_6hours.iloc[np.random.permutation(len(df_day_6hours))].reset_index(drop = True)

#print(df_day_hour_shuffle.head(10))


# In[56]:

#Pour mieux repartir training et test (y comprenant la tendance et la saisonnalite)
#! Trop moins de donnees vont provoquer les tres grands variances
#On prend 80% de nombre de passages de chaque mois comme training
#On prend 20% de nombre de passages de chaque mois comme test
def cross_mois_train_test(df_shuffle):
    train_day_hour = pd.DataFrame()
    test_day_hour = pd.DataFrame()
    total_mois = [df_shuffle[df_shuffle.month.isin([i])] for i in range(1,13)]
    for i in range(0,12): 
        train_day_hour_i = total_mois[i][0:math.floor(0.8*len(total_mois[i]))]
        test_day_hour_i = total_mois[i][math.floor(0.8*len(total_mois[i])): len(total_mois[i])]
        train_day_hour = pd.concat([train_day_hour,train_day_hour_i])
        test_day_hour = pd.concat([test_day_hour,test_day_hour_i]) 
    return (train_day_hour, test_day_hour)   


# In[57]:

train_day_hour, test_day_hour = cross_mois_train_test(df_day_hour_shuffle)


# In[58]:

y_train = train_day_hour['Sum']
x_train = train_day_hour.drop(['Date','Sum'],axis = 1)


# In[59]:

y_test = test_day_hour['Sum']
x_test = test_day_hour.drop(['Date','Sum'],axis = 1)


# In[60]:

y_train.shape


# In[61]:

#np.isnan(x_train).any()
#np.isnan(y_train).any()


# In[62]:

reg_train = linear_model.LinearRegression()
#print(len(x_train.notnull()))
reg_train.fit(x_train, y_train)


# In[63]:

#Coefficients
print("Coefficient:\n", reg_train.coef_)


# In[64]:

#Mean Square Error of training
print('Residual sum of square of training: %.2f'%np.mean((y_train-reg_train.predict(x_train))**2))


# In[65]:

#Mean Square Error of test
print('Residual sum of square of test: %.2f'%np.mean((y_test-reg_train.predict(x_test))**2))


# In[66]:

#score R^2
reg_train.score(x_test, y_test)


# In[67]:

#Adjusted R^2
print(1-sum((y_test-reg_train.predict(x_test))**2)/(len(y_test)-8-1)/sum((y_test-np.mean(y_test))**2)*(len(y_test)-1))


# In[68]:

#MAPE mean absolute percentage error
print('mean absolute percentage error: %.2f'%np.mean(abs(y_train-reg_train.predict(x_train))/abs(y_train)))


# In[69]:

sum(abs(y_train-reg_train.predict(x_train))/abs(y_train))/len(y_train)


# ### Tester les modèles

# 1.Avec les données anciennes (24avant, 724avant)

# In[70]:

df_day_hour_tester = pd.DataFrame(df_day_hour.Sum, columns = ['Sum'])
#print(df_day_hour_tester)


# In[71]:

#dataframe est le matrice de features, y est le reponse
#Dans ce cas la, on prend dataframe = df_day_hour_shuffle.drop(['Date','Sum'],axis = 1), y = df_day_hour_shuffle.Sum
def test_linear_regression(dataframe, y):
    coef = {}
    RMSE_train = []
    RMSE_test = []
    R2 = []
    adjustedR2 = []
    MAPE = []
    dataframe_tester = pd.DataFrame()
    #line = [i for i in range(0,len(dataframe.columns))]
    #line.sort(reverse = True)
    for i in range(0,len(dataframe.columns)):
        dataframe_tester[dataframe.columns[i]] = dataframe.iloc[:,i].copy()
        #Couper les donnees
        x_train = dataframe_tester[0:math.floor(0.8*len(dataframe_tester))]
        x_test = dataframe_tester[math.floor(0.8*len(dataframe_tester)): len(dataframe_tester)]
        
        y_train = y[0:math.floor(0.8*len(y))]
        y_test = y[math.floor(0.8*len(y)): len(y)]
        #Faire la regression linear
        reg_train = linear_model.LinearRegression()
        reg_train.fit(x_train, y_train)
        #Mean Square Error of training and test
        RMSE_train.append(math.sqrt(np.mean((y_train-reg_train.predict(x_train))**2)))
        RMSE_test.append(math.sqrt(np.mean((y_test-reg_train.predict(x_test))**2)))
        #R^2
        R2.append(reg_train.score(x_test, y_test))
        #Adjusted R^2
        adjustedR2.append(1-sum((y_test-reg_train.predict(x_test))**2)/(len(y_test)-(i+1)-1)/sum((y_test-np.mean(y_test))**2)*(len(y_test)-1))
        #MAPE
        MAPE.append(np.mean(abs(y_train-reg_train.predict(x_train))/abs(y_train))) 
        #reg_train.predict(x_train)
        #reg_train.predict(x_test)
        reg_train_predict = reg_train.predict(x_train)
        reg_test_predict = reg_train.predict(x_test)
    #Coefficients
    coef = reg_train.coef_
    return(coef, RMSE_train, RMSE_test, R2, adjustedR2, MAPE, x_train, x_test, y_train, y_test, reg_train_predict, reg_train.predict(x_test))


# In[72]:

#En utilisant la fonction
coef, RMSE_train, RMSE_test, R2, adjustedR2, MAPE, x_train, x_test, y_train, y_test, reg_train_predict, reg_test_predict = test_linear_regression(df_day_hour_shuffle.drop(['Date','Sum'], axis = 1),df_day_hour_shuffle.Sum)


# In[73]:

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 4.5
plt.figure(1)
plt.subplot(121)
plt.plot(RMSE_train, color = 'red')
plt.title("RMSE Training")
x = range(0,10)
my_xticks = df_day_hour_shuffle.drop(['Date','Sum'],axis = 1).columns
plt.xticks(x, my_xticks, rotation = 70)
plt.subplot(122)
plt.plot(RMSE_test, color = 'blue')
plt.title("RMSE test")
x = range(0,10)
my_xticks = df_day_hour_shuffle.drop(['Date','Sum'],axis = 1).columns
plt.xticks(x, my_xticks, rotation = 70)


# In[74]:

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 8, 4.5
plt.figure(1)
plt.subplot(111)
plt.plot(adjustedR2, color = 'red')
plt.title("Adjusted R2")
x = range(0,10)
my_xticks = df_day_hour_shuffle.drop(['Date','Sum'],axis = 1).columns
plt.xticks(x, my_xticks, rotation = 45)


# In[75]:

#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


# In[76]:

print ("Linear model:", pretty_print_linear(coef))


# 2.Recursive feature elimination (RFE)->backward (Il n'est pas explicatif)

# In[77]:

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE

estimator = linear_model.LinearRegression()
selector = RFE(estimator, 5, step=1)
selector = selector.fit(x_train, y_train)
print(selector.support_) 
print(selector.ranking_)
print(x_train.columns)


# 3.Avec pénalisation L1 (Pour choisir les features)

# In[78]:

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

def Lasso_CV_regression(X, y):
    #En utilisant lasso et la validation crosée pour sélecter le modèle optimal
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold = 10e-10)
    sfm.fit(X, y)
    #Les features choisit
    n_features = sfm.transform(X).shape[1]
    
    while n_features > 5:
        sfm.threshold +=1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]
    return sfm.transform(X);
    
    


# In[79]:

X_transform = Lasso_CV_regression(df_day_hour_shuffle.drop(['Date','Sum'],axis = 1), df_day_hour_shuffle.Sum)


# In[80]:

print(X_transform.shape[1])


# In[81]:

clf = LassoCV()
sfm = SelectFromModel(clf)
sfm.fit(df_day_hour_shuffle.drop(['Date','Sum'],axis = 1), df_day_hour_shuffle.Sum)
#Les features choisit
n_features = sfm.transform(x_train).shape[1]
print(X_transform)    
print(sfm.get_params())    


# 4.Best subset selection

# 4.1Selection

# In[82]:

#Validation Croissée par mois
#En utilisant Best subset selection pour trouver un modèle pertinent
import itertools
import time
import statsmodels.api as sm
#Une fonction pour select les features lorsque le nombre de features dans le modele sont fixe
#Le regle est RSS selon Least Square
#1er étape
def subset(y, X, feature_set):
    """
    df_shuffle: dataframe, y compris le réponse et les features
    n_folds: n_folds validation croissée
    feature_set: la combinaison de features
    """
    reg = []
    RSS = []
    #En utilisant toutes les données à construire un modèle
    model = sm.OLS(y, X[list(feature_set)])
    reg = model.fit()
    RSS = ((reg.predict(X[list(feature_set)])-y) ** 2).sum()
    return{"model":reg, "RSS":RSS}


# In[83]:

#2em étape
def best_subset_selection(y, X, k):
    """
    df_shuffle: dataframe, y compris le réponse et les features
    n_folds: n_folds validation croissée
    k: nombre de features dans la combinaison
    """
    tic = time.time()
    result =[]
    for com in itertools.combinations(X.columns, k):
        result.append(subset(y, X, com))
        #reserver des modeles dans un dataframe
    models = pd.DataFrame(result)
    #Choisir le meilleure modele au point de vue du plus petit RSS
    best_model = models.loc[models['RSS'].argmin()]
    toc = time.time()
    #print(models.shape[0], "modèles avec", k, "prédicteurs en", (toc-tic), "s.")
    #print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")
    return best_model


# In[84]:

import statistics as st#mean
def best_subset_selection_CV(df_shuffle, n_folds, features):
    '''
    features: en ce cas là, on prends les features dans le meilleure modèle selon le nombre de features
    '''
    tic = time.time()
    models = pd.DataFrame(columns = ["RSS", "model"])
    for i in range(1,10):
        models.loc[i] = best_subset_selection(df_shuffle.Sum, df_shuffle.drop(['Sum', 'Date'], axis = 1), i)
#===#Initiation
    RMSE = {}
    RMSE_mean = []
    train_day_hour = pd.DataFrame()
    test_day_hour = pd.DataFrame()
    total_mois = [df_shuffle[df_shuffle.month.isin([i])] for i in range(1,13)]
    #Pour réserver les RMSE
    for p in range(1,10):
        RMSE['%s'%p] = []
#===#Validation croisée   
    for j in range(0, n_folds):
    #Pour un fold
        for i in range(0,12): 
            #Plus lent 
            #pour un mois, faire une validation croisée, comprendre les trainings et les tests dans train_day_hour et test_day_hour
            test_day_hour_i = total_mois[i][j*math.floor((1/n_folds)*len(total_mois[i])):(j+1)*math.floor((1/n_folds)*len(total_mois[i]))]
            train_day_hour_i = total_mois[i].drop(test_day_hour_i.index)
            train_day_hour = pd.concat([train_day_hour,train_day_hour_i])
            test_day_hour = pd.concat([test_day_hour,test_day_hour_i]) 
            
        x_train_day_hour = train_day_hour.drop(['Sum','Date'], axis = 1)
        y_train_day_hour = train_day_hour.Sum.copy()
        x_test_day_hour = test_day_hour.drop(['Sum','Date'], axis = 1)
        y_test_day_hour = test_day_hour.Sum.copy()
        
#=======#Pour chaque combinaison de features, on fait (un répétition de) validation croisée
        for k in range(1,10):
            feature_set = features[: k]
            #En utilisant Least Squared
            model_j = sm.OLS(y_train_day_hour, x_train_day_hour[list(feature_set)])
            reg_j = model_j.fit()
            #Calculer RMSE de chaque fold
            RMSE['%s'%k].append(math.sqrt((((reg_j.predict(x_test_day_hour[list(feature_set)])-y_test_day_hour) ** 2).sum()/len(x_test_day_hour))))
    for v in range(1,10):    
        RMSE_mean.append(st.mean(RMSE['%s'%v]))
    best_model = models.loc[RMSE_mean.index(min(RMSE_mean))]
#===#Figure de AIC, BIC, R^2    
    #3em étape: choisir le meilleure modèle parmi 10 combinaisons de features par AIC, BIC, R^2, CV
    plt.figure(figsize = (20,10))
    plt.rcParams.update({'font.size':18, 'lines.markersize': 10})
    #Figure de RSS
    plt.subplot(2,2,1)
    plt.plot(models["RSS"])
    plt.xlabel('Nombre de prédicteur')
    plt.ylabel('RSS')
    #Figure de adjusted R-squared
    rsquared = models.apply(lambda x: x[1].rsquared, axis = 1)
    plt.subplot(2,2,2)
    plt.plot(rsquared)
    plt.plot(rsquared.argmax(), rsquared.max(), "or")
    plt.xlabel('Nombre de prédicteur')
    plt.ylabel('adjusted rsquared')
    #Figure de AIC
    aic = models.apply(lambda x: x[1].aic, axis = 1)
    plt.subplot(2,2,3)
    plt.plot(aic)
    plt.plot(aic.argmin(), aic.min(), "or")
    plt.xlabel('Nombre de prédicteur')
    plt.ylabel('AIC')
    #Figure de BIC
    bic = models.apply(lambda x: x[1].bic, axis = 1)
    plt.subplot(2,2,4)
    plt.plot(bic)
    plt.plot(bic.argmin(), aic.min(), "or")
    plt.xlabel('Nombre de prédicteur')
    plt.ylabel('BIC')
    
    toc = time.time()
    print("Temps total", (toc-tic), "s.")
    print(best_model['model'].summary())
    return best_model


# 4.11Par heure

# In[85]:

features = ['724avant', '24avant', 'temperature', 'precipProbability', 'weekday', 'hour', 'day', 'month', 'precipIntensity']
bestmodel = best_subset_selection_CV(df_day_hour_shuffle, 10, features)


# In[86]:

#essaiyer le modèle de 4 features
best_4 = best_subset_selection(df_day_hour_shuffle.Sum, df_day_hour_shuffle.drop(['Date','Sum'],axis = 1), 4)
print(best_4['model'].summary())


# 4.12Par 6 heures

# In[87]:

bestmodel_6hours = best_subset_selection_CV(df_day_6hours_shuffle, 10, features)


# 7.2Détection de multicollinearity

# In[88]:

#Coefficient de corrélation
#cov = np.corrcoef(X)


# In[89]:

#cov


# In[90]:

#Variance inflation factor: pour détecter multicollinearity parmi les prédicteurs
#Interpretation:The square root of the variance inflation factor tells you how much larger the standard error is,
#compared with what it would be if that variable were uncorrelated with the other predictor variables in the model.
def VIF(X, X_index):
    """
    k_vars: nombre de columns
    x_i: column choisit
    mask: index de columns qui ne sont pas le column choisit
    x_noti: columns qui ne sont pas le column choisit
    r_squared_i: R-squared de column choisit
    vif: variance inflation factor
    """
    k_vars = X.shape[1]
    x_i = X.iloc[:, X_index]
    mask = np.arange(k_vars) != X_index
    x_noti = X.iloc[:, mask]
    r_squared_i = sm.OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


# In[91]:

test = df_day_hour_shuffle[0:math.floor(0.1*len(df_day_hour_shuffle))]

index_train = df_day_hour_shuffle.drop(test.index)


# In[92]:

def detect_multicollinearity(X):
    i = 0
    for feature in X.columns:
        vif_i = VIF(X, i)
        i += 1
        print("%s"%feature, "a un variance inflation factor", vif_i)
    return


# In[93]:

#vif est trop élevé s'il supérieur que 5
#print('Pour tous les features:')
#detect_multicollinearity(X)
#print('\n','Le modèle choisit:')
#detect_multicollinearity(X.drop(['month','day', 'weekday', 'hour', 'precipIntensity', 'Toy'],axis = 1))


# In[94]:

'''
Why should we consider multicollinearity?
severe multicollinearity is a problem because it can increase the variance of the coefficient estimates and 
make the estimates very sensitive to minor changes in the model. 
The result is that the coefficient estimates are unstable and difficult to interpret.
Multicollinearity saps the statistical power of the analysis, can cause the coefficients to switch signs, 
and makes it more difficult to specify the correct model.
'''


# 7.3 Prédiction

# 7.31 Par heure

# In[97]:

pre_best = bestmodel['model'].predict(df_day_hour_shuffle.drop(['Sum', 'Date', 'precipIntensity', 'month', 'Toy', 'day','weekday'], axis = 1))


# In[98]:

pre_best_4 = best_4['model'].predict(df_day_hour_shuffle.drop(['Sum', 'Date', 'Toy', 'weekday', 'hour', 'day', 'month', 'precipIntensity'], axis = 1))


# 7.32 Par 6 heures

# In[100]:

pre_best_4_6hours = bestmodel_6hours['model'].predict(df_day_6hours_shuffle.drop(['Sum', 'Date', 'Toy', 'weekday', 'hour', 'day', 'month', 'precipIntensity'], axis = 1))


# ### Distributions de résidus absolus

# 1.Avec les données anciennes (24avant, 724avant)

# In[101]:

res_train = abs((y_train-reg_train_predict))


# In[102]:

import matplotlib.mlab as mlab
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9
# the histogram of the data
n, bins, patches = plt.hist(res_train, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus de modèle 1")


# In[103]:

grande_erreur = df_day_hour_shuffle[0:math.floor(0.8*len(df_day_hour_shuffle))][res_train > 200]
grande_erreur['residu'] = res_train[res_train > 200]
print(display(grande_erreur[['Date','residu']].sort_values(by = 'residu')))


# 2.Best subset selection modèle

# In[104]:

res_best = abs(df_day_hour_shuffle.Sum - pre_best)


# In[105]:

import matplotlib.mlab as mlab
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9
# the histogram of the data
n, bins, patches = plt.hist(res_best, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus de modèle 2 (Best subset selection 7 features)")


# In[106]:

from IPython.display import display, HTML
grande_erreur = df_day_hour_shuffle[res_best > 200]
grande_erreur['residu'] = res_best[res_best > 200]
print(display(grande_erreur[['Date','residu']].sort_values(by = 'residu')))


# 3.Best subset selection modèle (4 features)

# 3.1Par heure

# In[ ]:

res_best_4 = abs(df_day_hour_shuffle.Sum - pre_best_4)


# In[ ]:

import matplotlib.mlab as mlab
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9
# the histogram of the data
n, bins, patches = plt.hist(res_best_4, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus de modèle 3 (Best subset selection 4 features)")


# In[ ]:

from IPython.display import display, HTML
grande_erreur_best_4 = df_day_hour_shuffle[res_best_4 > 200]
grande_erreur_best_4['residu'] = res_best_4[res_best_4 > 200]
print(display(grande_erreur_best_4[['Date','residu']].sort_values(by = 'residu')))


# 3.2Par 6 heures

# In[107]:

res_best_4_6hours = abs(df_day_6hours_shuffle.Sum - pre_best_4_6hours)


# In[110]:

import matplotlib.mlab as mlab
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 9
# the histogram of the data
n, bins, patches = plt.hist(res_best_4_6hours, 300, normed=1, facecolor='green', alpha=0.75)
plt.title("Histogram de résidus absolus de modèle (Best subset selection 3 features)")


# In[111]:

from IPython.display import display, HTML
grande_erreur_best_4_6hours = df_day_6hours_shuffle[res_best_4_6hours > 550]
grande_erreur_best_4_6hours['residu'] = res_best_4_6hours[res_best_4_6hours > 550]
print(display(grande_erreur_best_4_6hours[['Date','residu']].sort_values(by = 'residu')))


# In[ ]:



