# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:58:31 2017

@author: hankk
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import scoreCardOutput as cc
import logisticRegression as lg
import modelEvaluate as me
import woeInfoValue as wi
import populationStabilityIndex as psi
import optimalBining as ob
import ScoreCalculater as sc
import dataVisualization as dv
import seaborn as sns

sns.set(style="ticks", color_codes=True)

"""
read and copy example data
"""
os.chdir("D:/dowload/Chrome_download/default-of-credit-card-clients-dataset")
data = pd.read_csv("creditCard_UCI.csv")
df = data.copy()

# drop the unuse variable, and define X & y 
df = df.drop("Unnamed: 0", axis=1)
X = df.drop('target', axis=1)
y = df['target']


"""
describe of data
"""
# define  var types
# user base information
continues = ['LIMIT_BAL', 'AGE', 
             'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
             'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

for var in continues:
    print("describe of var: %s" % var)
    dv.drawHistogram(X[var])
    print('--'*20)

# plot scatter matrix 
g = sns.pairplot(X[continues])
# decribe for every vars in continues
X[continues].describe()
# corr_matrix
def corr_heatmap(X):
    """
    corr heatmap
    """
    corr_matrix = X.corr()
    plt.figure(figsize=(8, 8))
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr_matrix, 
                     mask=mask, 
                     vmax=1., 
                     square=True,
                     )
    
    
corr_matrix = X[continues].corr()
plt.figure(figsize=(8, 8))
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr_matrix, 
                     mask=mask, 
                     vmax=1., 
                     square=True,
                     )
# classify vars
classify = ['SEX', 'EDUCATION', 'MARRIAGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'target']

for var in classify:
    print("distribution of var: %s" % var)
    s = df[var]
    dv.drawBar(s, s.value_counts().index)
    print('--'*20)


"""
data optimal bining
"""
# define vars need to bining
continues_opt = continues
classify_opt = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# continues bining
cont_var_bin_map = dict()

df_bined = df.copy()
for var in continues_opt:
    cont_var_bin_map[var] = ob.binContVar(df[var], y, method=4)
    df_bined[var] = ob._applyBinMap(df[var], cont_var_bin_map[var])


class_var_bin_map = dict()

for var in classify_opt:
    class_var_bin_map[var] = ob.reduceCats(df[var], y, method=4)
    df_bined[var] = ob.applyMapCats(df[var], class_var_bin_map[var])


"""
woe and info value
"""

woe_maps = dict()
info_vales = dict()

# plot bar plot for woe, and calculate info value
for var in df_bined.columns:
    if var != 'target':
        woe_maps[var], info_vales[var] = wi.woe_single_x(df_bined[var], y)
        print('--'*20)
        print('woe bar plot of %s' %var)
        wi.plot_woe(woe_maps[var])

# select vars who got weak info value         
iv_series = pd.Series(info_vales)
weak_iv_vars = iv_series[iv_series < 0.02].index

# woe transform
df_woe_trans = df_bined.drop(weak_iv_vars, axis=1).copy()
df_woe_trans = df_woe_trans.drop('EDUCATION', axis=1)
for var in df_woe_trans.columns:
    if var != 'target':
        df_woe_trans[var] = df_woe_trans[var].map(woe_maps[var])

"""
logistic model
"""
# split trainsets and testsets, ratio is 1:3
X_model = df_woe_trans.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_model, y, 
                                                    test_size=0.33, 
                                                    random_state=42)


# build model, var select == forward
forward_vars = lg._forward_selected_logit(X_train, y_train)

model = sm.Logit(y_train, sm.add_constant(X_train[forward_vars]))
model_res = model.fit()
print(model_res.summary2())

# model evaluate
prob_y = model_res.predict(sm.add_constant(X_test[forward_vars]))
pred_y = np.where(prob_y > 0.5, 1, 0)

# comfuse matrix
me.plot_confusion_matrix(y_test, pred_y, labels=[0, 1])
# ROC curve
me.plot_roc_curve(prob_y, y_test)
# ks_value and ks_cruve
ks_df, _ = me.ks_stats(prob_y, y_test)
# lift chart and lorenz curve
me.lift_lorenz(prob_y, y_test)

"""
scoresCard output

scoresCard type
    basescore:yes
    weights:no
"""

parameters = {"basepoints":600, "odds":60, "PDO":20,
              "has_basescore":True, "has_weights":False}


paramsEst = model_res.params

scoresCard = cc.creditCards(paramsEst, X_train, parameters,
                            woe_maps=woe_maps, 
                            bin_maps=cont_var_bin_map,
                            red_maps=class_var_bin_map)


"""
scores calculate
"""

df_X_train = df.loc[X_train.index, :]
df_X_test = df.loc[X_test.index, :]

scores_train = sc.calculateCreditScore(df_X_train, scoresCard)
scores_test = sc.calculateCreditScore(df_X_test, scoresCard)

"""
PSI for scorescard
"""

parameters_psi_k = {'k':20}
psi_table, PSI_ind = psi.populationStabilityIndex(scores_train,
                                                  scores_test,
                                                  y_train,
                                                  y_test,
                                                  parameters_psi_k)












