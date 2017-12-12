# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:30:23 2017

@author: Administrator
@mission: PSI 

"""

import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target



def _check_target_binary(y):
    """
    check if the target variable is binary
    ------------------------------
    Param
    y:exog variable,pandas Series contains binary variable
    ------------------------------
    Return
    if y is not binary, raise a error   
    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')



def _EqualWidthBinMap(x, Mbins, adjust=np.inf):
    """
    Data bining function, 
    middle procession functions for binContVar
    method: equal width
    Mind: Generate bining width and interval by Mbins
    --------------------------------------------
    Params
    x: pandas Series, data need to bining
    Acc: float less than 1, partition ratio for equal width bining
    adjust: float or np.inf, bining adjust for limitation
    --------------------------------------------
    Return
    bin_map: pandas dataframe, Equal width bin map
    """
    varMax = x.max()
    varMin = x.min()
    # generate range by Acc
    minMaxSize = (varMax - varMin)/Mbins
    # get upper_limit and loewe_limit
    ind = range(1, Mbins+1)
    Upper = pd.Series(index=ind, name='upper')
    Lower = pd.Series(index=ind, name='lower')
    for i in ind:
        Upper[i] = varMin + i*minMaxSize
        Lower[i] = varMin + (i-1)*minMaxSize
    
    # adjust the min_bin's lower and max_bin's upper     
    Upper[Mbins] = Upper[Mbins]+adjust
    Lower[1] = Lower[1]-adjust
    bin_map = pd.concat([Lower, Upper], axis=1)
    bin_map.index.name = 'bin'
    return bin_map    


def temp_func(a, lower, upper, i):
    
    if a > lower and a <= upper:
        return i
    else:
        return a


def _applyBinMap(x, bin_map):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    #bin_res = np.array([0] * x.shape[-1], dtype=int)
    #x2 = x2.copy()
    res = x.copy()
    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]

        loc = np.where((x > lower) & (x <= upper))[0]
 
        ind = x.iloc[loc].index
        res.loc[ind] = i
        
        
    res.name = res.name + "_BIN"
    
    return res



def ratioFix(table, k, var):
    
    l = [0] * table.shape[1]
    for i in range(1, k+1):
        try:
            table.loc[i,:]
        except KeyError:
            table.loc[i, :] = l
         
    for i in table.index:
        if any(table.loc[i, [0,1]] == 0):
            table.loc[i, var] = 0
    
    table = table.sort_index()
    return table



def populationStabilityIndex(score_train, 
                             score_test,
                             y_train, 
                             y_test,
                             parameters
                             ):
    """
    Func
    Output of PSI and PSI table
    other: index of points and targets must equal
    --------------------------------------------
    Params
    p_train: 训练集信用评分
    p_test: 测试集信用评分
    y_train: 训练集的y
    y_test: 测试集的y
    parameters:dict,只有1个key：k,为结果表分段数量
                parameters={
                            'k':20}
    
    ---------------------------------------------
    Return
    PSI: float
    res: pandas dataframe
    """
    # 检查测试目标和训练目标是否二元属性
    _check_target_binary(y_test)
    _check_target_binary(y_train)
    # 合并训练集和测试集的评分结果和实际好坏客户结果
    data1 = pd.concat([score_train, y_train], axis=1)
    
    data1.columns = ['scores_train', 'y_train']
    data2 = pd.concat([score_test, y_test], axis=1)
    
    data2.columns = ['scores_test', 'y_test']
    # 等宽分箱
    bin_map = _EqualWidthBinMap(data1['scores_train'], Mbins=parameters['k'])
    
    data1['bins'] = _applyBinMap(data1['scores_train'], bin_map)
    data2['bins'] = _applyBinMap(data2['scores_test'], bin_map)
    
    ctab1 = pd.crosstab(index=data1['bins'], columns=data1['y_train'], dropna=False)
    ctab2 = pd.crosstab(index=data2['bins'], columns=data2['y_test'], dropna=False)
    
    
    ctab1['counts'] = ctab1.apply(func=np.sum, axis=1)
    ctab2['counts'] = ctab2.apply(func=np.sum, axis=1)
    
    ctab1['positive_ratio_expect'] = ctab1[1]/ctab1['counts']
    ctab2['positive_ratio_actual'] = ctab2[1]/ctab2['counts']
    
    ctab1 = ratioFix(ctab1, parameters['k'], 'positive_ratio_expect')
    ctab2 = ratioFix(ctab2, parameters['k'], 'positive_ratio_actual')
    
    res = pd.concat([ctab1['positive_ratio_expect'],
                     ctab2['positive_ratio_actual']], axis=1)
    
    A_sub_E = res.apply(lambda x: x[1] - x[0], axis=1)
    log_A_div_E = res.apply(lambda x: np.log(x[1] / x[0]), axis=1)
    res['range'] = bin_map.apply(lambda x: '['+str(x[0])+',  '+ str(x[1])+')', axis=1)
    
    
    res['index'] = A_sub_E * log_A_div_E
    
    # res['log(A/E)'] = res['log(A/E)'].fillna(0)
    res['index'] = res['index'].fillna(0)
    # res = res.drop(['A-E', 'log(A/E)'], axis=1)
    indx = np.where(res['index']==np.inf, 0, res['index'])
    PSI = indx.sum()
    res = res[['range', 'positive_ratio_actual', 'positive_ratio_expect', 'index']]
    res['positive_ratio_actual'] = res['positive_ratio_actual'].map(lambda x: round(x, 3))
    res['positive_ratio_expect'] = res['positive_ratio_expect'].map(lambda x: round(x, 3))
    return res, PSI

"""
测试脚本
--------------------------------
# 读取数据
os.chdir("E:/anaylsis_engine/creditCardArtcle/codes")
data = pd.read_excel("datasets.xlsx")

# 划分训练和测试集合
from sklearn.model_selection import train_test_split


X = data['credit_score']
y = data['target']

score_train, score_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

populationStabilityIndex(score_train, 
                         score_test,
                         y_train, 
                         y_test,
                         targets=1, k=20, adjust=0.00001)


"""










