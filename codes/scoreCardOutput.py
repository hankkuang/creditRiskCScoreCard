# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:48:35 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:49:17 2017

@author: Hank Kuang
@title: 评分卡生成
"""

import numpy as np
import statsmodels.api as sm
#import re
import pandas as pd

def creditCards(paramsEst,
                X_train,
                parameters,
                woe_maps={},               
                bin_maps={},
                red_maps={}
                ):
    """
    output credit card for each var in model
    --------------------------------------------
    ParamsEst: pandas Series, 参数估计结果, index为变量名称，value为估计结果
    bin_maps: dict, key 为变量名称，包含在ParamsEst的index内， 值为dataframe，是分箱映射结果
    red_maps: dict, key 为变量名称, 包含在ParamsEst的index内，值为dataframe，是降基映射结果
    woe_maps: dict, key 为变量名称, 要求与ParamsEst的index一一对应，value 同为字典，键值对为变量分段数值--woe值
    parameters = {
       has_basescore:bool,是否包含一个默认的基础分
       has_weights:bool,是否包含权重
    	basepoints: int, 基准分,默认600
    	odds: 概率比基数odds, 默认60
    	PDO: 翻倍系数，默认20
    	}
    parameters = {"basepoints":600, "odds":60, "PDO":20,
                  "has_basescroe":False, "has_weights":True}
    -------------------------------------------
    Return
    creditCard: pandas dataframe
    """
    # 计算A&B
    A, B = _score_cal(parameters["basepoints"], parameters["odds"], parameters["PDO"])
    # 计算基础分
    
    # 如果有权重，则将参数估计转化为权重
    if parameters['has_weights'] == True:
        paramsEst2 = paramsEst.map(lambda x: abs(x))
        weights = [float(x)/float(paramsEst2.sum()) for x in paramsEst2]
        weights = pd.Series(weights, index=paramsEst.index)
        if parameters['has_basescore'] == False and 'const' in paramsEst.index:
            p = weights.const/(paramsEst.shape[0]-1)
            weights = weights[1:] + p
        weights = weights.map(lambda x: round(x, 2))
    # 评分卡计算    
    Scores = pd.DataFrame()
    
    for k in paramsEst.index:
    #for k in woe_maps.keys():
        # 如果k不在X_train的变量名列表中，则跳过k，继续循环
        if k not in X_train.columns:
            continue
        # 如果变量进行了woe转换，则调取woe转换的变量,否则调取bin或者red后的分段数
        # 如果变量既没有woe，也没有bin和red，则调取训练集unique后的数值
        if k in woe_maps.keys():
            d = pd.DataFrame(woe_maps[k], index=[k]).T
        elif k in bin_maps.keys():
            d = pd.DataFrame(bin_maps[k].loc[:,'bin'])
            d = d.rename(columns={'bin':k})
        elif k in red_maps.keys():
            d = pd.DataFrame(red_maps[k].loc[:,'bin'].unique())
            d = d.rename(columns={'bin':k})
        else:
            d = pd.DataFrame(np.unique(X_train[k]), columns=[k])
        # 将索引转换成整数
        d.index = [int(float(i)) for i in d.index]
        
        # 如果权重为真，则分数的计算不包括变量参数
        # 如果初始的参数估计值为负数
        if parameters['has_weights'] == True:
            if paramsEst[k] > 0: 
                d['score'] = round(-B*d.loc[:,k])
            else:
                d['score'] = round(B*d.loc[:,k])
        else:
            d['score'] = round(-B*d.loc[:,k]*paramsEst[k])       
        #else:
            #print(k_2)
            #pass
        if k in bin_maps.keys():
            bin_map = bin_maps[k]
            bin_map.index = bin_map['bin']
            bin_map = bin_map.drop(['total', 'bin'], axis=1)
            bin_map['range'] = bin_map.apply(lambda x:str(x[0]) + '--' + str(x[1]), axis=1)
            bin_map = bin_map.drop(['lower', 'upper'], axis=1)
            d = pd.merge(d, bin_map, left_index=True, right_index=True)
        
        elif k in red_maps.keys():
            red_map = red_maps[k]
            s = tableTranslate(red_map)
            s = pd.DataFrame(s.T, columns=['range'])            
            d = pd.merge(d, s, left_index=True, right_index=True)
                    
        else:
            d['range'] = d.index
        
       
        n = len(d)
        ind_0 = []
        i = 0
        single_weight = [1] * n
        while i < n:
            ind_0.append(k)
            if parameters['has_weights'] == True:
                single_weight[i] = weights[k]
                
            i += 1
        
        d.index = [ind_0, single_weight, list(d.index)]
        d = d.drop(k, axis=1)
        Scores = pd.concat([Scores, d], axis=0)
    
    Scores.index.names = ["varname", "weight", "binCode"]
    
    # 输出评分卡
    # 如果包含权重，则将参数估计结果转化为权重                
    try:    
        baseScore = round(A - B * paramsEst['const'])
    except KeyError:
        raise("模型缺少截距项，无法计算截距得分")
    if parameters['has_basescore'] == True:
        
        baseScore = pd.DataFrame([[baseScore, '-']], 
                            index=[['baseScore'], [weights[0]], ['-']], 
                            columns=['score', 'range'])
        
        credit_card = pd.concat([baseScore, Scores], axis=0)
    else:
        Scores['score'] = round(Scores['score'] + (baseScore/(paramsEst.shape[0]-1)))
        credit_card = Scores.copy()
            
    return credit_card

def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin']==i].index
        value = list(value.map(lambda x:str(x)+';'))
        value = "".join(value)
        res[i] = value
    return res

def _score_cal(basepoints, odds, PDO):
    """
    cal alpha&beta for score formula, 
    score = alpha + beta * log(odds)
    ---------------------------------------
    Params
    basepoints: expect base points
    odds: cal by logit model
    PDO: points of double odds  
    ---------------------------------------
    Return
    alpha, beta 
    """
    beta = PDO/np.log(2)
    #odds = 1./odds  
    alpha = basepoints - beta*np.log(odds)
    return alpha, beta






"""
test script:
    
paramsEst = pd.Series([2.3, 1.1, -0.9, -0.6], index=['const', 'age', 'salary', 'house'])

X_train = pd.DataFrame(columns=['age', 'house', 'salary']) 
X_train['house'] = [0,0,1,0,1,0,1]
 

woe_maps = {"age":{'1':1.13, '2':0.9, '3':0.16}}

bin_maps = dict()
bin_maps['age'] = pd.DataFrame(columns=['bin', 'lower', 'upper', 'total'], index=[0,1,2])
bin_maps['age']['bin'] = [1,2,3]
bin_maps['age']['lower'] = [18,25,35]
bin_maps['age']['upper'] = [25,35, 50]
bin_maps['age']['total'] = [100, 120, 90]

bin_maps['salary'] = pd.DataFrame(columns=['bin', 'lower', 'upper', 'total'], index=[0,1,2])
bin_maps['salary']['bin'] = [1,2,3]
bin_maps['salary']['lower'] = [1000,2500,5000]
bin_maps['salary']['upper'] = [2500,5000, 10000]
bin_maps['salary']['total'] = [100, 80, 40]



"""
















