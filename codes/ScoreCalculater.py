# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:18:09 2017

@author: Hank Kuang
@mission: Credit Risk Score Calculate
"""


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


class CreditScoreCalculater:
    
    
    def __init__(self, credit_card, datasets):
        """
        Params
        ------------------------------------------
        credit_card: pandas dataframe
        datasets: pandas dataframe
        """
        self.credit_card = credit_card
        self.datasets = datasets
    
    
    def ScoresOutput(self):
        """
        Func:
        output the results
        -------------------------
        Return
        pandas Series, index is dataset's index, values is scores
        """
        scores_list = []
        
        # 解析所有评分卡，将评分卡转为可计算的表格形式
        # 连续型变量包括：scores, lower&upper
        # 字符型变量包括：scores，valuelist
        scoresMap = dict()
        weights = dict()
        
        not_in_credit_cards = []
        for v_name in self.datasets.columns:
            if v_name in self.credit_card.index.levels[0]:
                
                weights[v_name], scoresMap[v_name] = self.ParseCreditCard(v_name)
            else:
                not_in_credit_cards.append(v_name)
        # parse
        for i in self.datasets.index:
            
            p = self.datasets.loc[i,:]
            # 计算个人每个变量下对应值得分数
            scores = [self.calScore(i, v, weights[i], scoresMap[i]) 
                     for i, v in zip(p.index, p.values) 
                     if i not in not_in_credit_cards]
            # 加总评分
            scoreSum = sum(scores)
            scores_list.append(scoreSum)
            
        score_series = Series(scores_list, index=self.datasets.index)
        score_series.name = 'credit_score'
        # 如果评分卡包含基准分，则加入基准分
        if 'baseScores' in self.credit_card.index.levels[0]:
            baseScores = self.credit_card.loc['baseScores']['score'][0]
            w = self.credit_card.loc['baseScores', :].index.levels[0][0]
            baseScore = float(baseScores) * float(w)
            score_series = score_series + baseScore
        
        return score_series
    
    
    def ParseCreditCard(self, varname):
        """
        Func:
        Parse CreditCard to table, 
        contains scores, rule, and weights  
        ----------------------------------------
        Params
        varname: variable name
        ----------------------------------------
        Return
        3 kinds of table, single variable's weight
        """
        # 根据变量名提取子评分卡，并将子评分卡索引进行转换，提取权重
        sub_card = self.credit_card.loc[varname, :]
        weight = sub_card.index.levels[0][0]
        sub_card.index = range(1, len(sub_card)+1)
        
        try:
            if '--' in sub_card['range'].iloc[0]:
            
                table = self.rangeParse(sub_card, method=1)
            
            elif ';' in sub_card['range'].iloc[0]:
                table = self.rangeParse(sub_card, method=2)
            
            else:
                table = self.rangeParse(sub_card, method=3)
        
        except TypeError:
            table = self.rangeParse(sub_card, method=3)
        return weight, table
    
       
    def calScore(self, varname, value, w, scoreMap):
        """
        Func
        calculate score of single variable
        -------------------------------------------
        Params
        varname:
        value:
        w:
        scoreMap:
        -------------------------------------------
        score 
        """
        
        if 'lower' in scoreMap.columns:
            for i in scoreMap.index:
                if value >= scoreMap.loc[i, 'lower'] and value < scoreMap.loc[i, 'upper']:
                    score = scoreMap.loc[i, 'score']
                else:
                    pass
        
        else:
            for i in scoreMap.index:
                value = str(value)
                try:
                    
                    if value in scoreMap.loc[i, 'range']:
                        score = scoreMap.loc[i, 'score']
                except TypeError:
                    
                    if value == scoreMap.loc[i, 'range']:
                        score = scoreMap.loc[i, 'score']
        
        score = score * w
        
        return score 
    
    def rangeParse(self, sub_card, method):
        """
        Func
        Parse range to cal rule
        -------------------------------------
        sub_card:
        method:
        -------------------------------------
        Return
        new sub_card
        """
        if method == 1:
            lowerList = []
            upperList = []
            for item in sub_card['range']:
                lower, upper = self.Parse(item)
                lowerList.append(lower)
                upperList.append(upper)
            sub_card['lower'] = lowerList
            sub_card['upper'] = upperList
            sub_card['lower'] = sub_card['lower'].astype(np.float_)
            sub_card['upper'] = sub_card['upper'].astype(np.float_)
        elif method == 2:
            lst = []
            for item in sub_card['range']:
                l = item.split(';')
                lst.append(l)
                lst = [str(l) for l in lst]
        
            sub_card.loc[:, 'range'] = lst
        
        else:
            try:
                sub_card.loc[:, 'range'] = [str(s) for s in sub_card.loc[:, 'range']]
            except Exception:
                print("something different in sub_card")
                print(sub_card)
        return sub_card
            
    
    def Parse(self, ranges_):
        """
        Func
        parse lower&upper
        ---------------------------
        Params
        ranges_: ranges of every rows
        -----------------------------
        Return
        upper & lower
        """
        lst = ranges_.split('--')
        return lst[0], lst[1]
        





def calculateCreditScore(datasets, credit_card):
    score_cal = CreditScoreCalculater(credit_card, datasets)
    res = score_cal.ScoresOutput()
    return res
    
"""
example:

import os
import pandas as pd

os.chdir("E:/anaylsis_engine/creditCardArtcle/codes")
credit_card = pd.read_excel("credit_card.xlsx")

credit_card['varname'] = credit_card['varname'].fillna(method='ffill')
credit_card['weights'] = credit_card['weights'].fillna(method='ffill')
credit_card['binCode'] = credit_card['binCode'].fillna(method='ffill')

credit_card.index = credit_card[['varname', 'weights', 'binCode']]
credit_card = pd.DataFrame(credit_card[['score', 'range']], 
                           index=[credit_card['varname'], credit_card['weights'], credit_card['binCode']])
    

os.chdir("D:/dowload/Chrome_download/default-of-credit-card-clients-dataset")
data = pd.read_csv("creditCard_UCI.csv")

score_cal = CreditScoreCalculater(credit_card, data)
res = score_cal.ScoresOutput()

"""
