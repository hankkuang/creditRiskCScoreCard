# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 18:55:07 2017

@author: Hank Kuang
@title: Logistic 回归模型
"""

#import numpy as np
import statsmodels.api as sm
from pandas import Series, DataFrame
from datetime import datetime
import pandas as pd
import numpy as np

class Logistic(object):
    
    def __init__(self, exog, endog):
        self.exog = exog
        self.endog = endog
        
    def modelFit(self, constant=True):
        """
        model fit
        -------------------------------
        Params
        constant:bool，True means contain intercept in model
        -------------------------------
        Return
        model: sm model obj
        results: fit result with x and intercept 
        results_0: fit result with only with intercept 
        """
        X = self.exog
        y = self.endog
        if constant:        
            X = sm.add_constant(X)
             
        model = sm.Logit(y, X, missing='drop')
        model_0 = sm.Logit(y, X.const, missing='drop')
        results = model.fit()
        results_0 = model_0.fit()
        return model, results, results_0

   
    def modelDescript(self, model, results):#通用方法
        """
        return information of model
        ----------------------------------
        Params
        model: sm model
        results: fit result with x and intercept 
        ----------------------------------
        Return
        pandas series
        """
        
        rlt = {
               "模型":"二元logistic模型", 
               "使用的观测个数":results.nobs,
               "含缺失值观测个数":self.exog.shape[0] - results.nobs,
               "总观测个数":self.exog.shape[0],
               "自变量":list(self.exog.columns),
               "因变量":self.endog.name,
               "方法":"最大似然估计",
               "日期时间":datetime.now()
              }#,
               #"Warnings":results.cov_kwds}
        return Series(rlt)



def ParamEST(results):
    """
    return params estimate from model fit results
    ------------------------------------
    Params
    results: model fit result 
    ------------------------------------
    Return
    pandas dataframe
    """
        
    rlt = pd.concat([
               results.params,
               results.bse,
               results.tvalues,
               (results.params/results.bse)**2,
               results.pvalues,
               results.conf_int()
               ], axis=1)
    rlt.columns = [u'参数估计', u'标准误', u'z值', u'wald卡方', u'p值', u'置信下界', u'置信上界']
    return rlt
    
def fitEval(results_1, results_0):
    """
    return metrics of model evaluation
    ------------------------------------
    Params
    results_1:  fit result with x and intercept 
    results_0: fit result with only with intercept 
    ------------------------------------
    Return
    rlt: pandas dataframe
    Series(rsq): 
    """
    indx = ['aic', 'bic', '-2*logL']
    S0 = Series([results_0.aic, results_0.bic, -2*results_0.llf], index=indx)
    S0.name = '仅含截距'
    S1 = Series([results_1.aic, results_1.bic, -2*results_1.llf], index=indx)
    S1.name = '包含截距和协变量'
    rlt = pd.concat([S0, S1], axis=1)
    rsq = {"mcfadden R^2":results_1.prsquared}
    return rlt, Series(rsq)
    
def modelQuality(results):
    """
    return metrics of model Quality
    ------------------------------------
    Params
    results: model fitting results
    ------------------------------------
    Return
    Series(rsq): pandas series
    """
    rlt = {
               "似然比":results.llr,
               "自由度":results.df_model,
               "似然比p值":results.llr_pvalue
              }
    return Series(rlt)

def confMatrix(results):
    """
    create confuse matrix
    ----------------------------------
    Params
    results: model fitting results
    ----------------------------------
    Return
    confu_mat: confuse matrix
    """
    confu_mat = DataFrame(results.pred_table())
    confu_mat.index.name = '实际结果'
    confu_mat.columns.name = '预测结果'
    return confu_mat

"""
other middle results
"""

def cov_matrix(results, normalized=False):
    """
    cov_matrix
    --------------------------------------
    
    """
    if normalized:
        rlt = results.normalized_cov_params
    else:
        rlt = results.cov_params()
    return rlt

def prediction(results, exog=None):
    """
    model prediction
    ---------------------------------
    """
    pred = results.predict(exog)
    pred = Series(pred)
    return pred



def Residual(results):
    return results.resid_generalized

def standardResidual(results):
    return results.resid_pearson

def devResidual(results):
    return results.resid_response


def _forward_selected_logit(X, y, sle=0.05):
    """
    Linear model designed by forward selection.

    Parameters:
    -----------
    X : pandas DataFrame with all possible predictors and response

    y: pandas series

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
    """
    import statsmodels.formula.api as smf
    #合并x,y
    data = pd.concat([X, y], axis=1)
    #提取相应变量名和自变量名称
    try:
        response = y.name
    except Exception:
        response = y.columns[0]
    
    #对每个变量遍历回归，并排序选出wald卡方最大的变量
    d = pd.DataFrame(index=X.columns, columns=['wald_chisq', 'p_value'])
    for var in X.columns:
        #print(var)
        formula = '{} ~ {} + 1'.format(response, var)
        mod = smf.logit(formula, data).fit()
        p = mod.pvalues[1]
        
        wald_chisq = (mod.params[1]/mod.bse[1])**2
        
        d.loc[var,:] = [wald_chisq, p] 
        
    d = d.sort_values(by='p_value', ascending=True)
    candidate = d.index
    # candidate = list(d.index)
    selected = []
    for cand_var in candidate:
        formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [cand_var]))
        try:
            mod = smf.logit(formula, data).fit()
            
        except Exception:
            continue
        if mod.pvalues[-1] <= sle:
            selected.append(cand_var)
        else:
            pass

    return selected
	
    
def _backward_selected_logit(X, y, sls=0.05):
    """
    Linear model designed by backward selection.

    Parameters:
    -----------
    X: pandas DataFrame with all possible predictors
    y: pandas Series with response
    sls: measure for drop variable
        
    Return:
    --------
    var_list
    """
    import statsmodels.formula.api as smf#导入相应模块
    data = pd.concat([X, y], axis=1)#合并数据
    #提取X，y变量名
    var_list = X.columns
    try:
        response = y.name
    except AttributeError:
        response = y.columns[0]
    #首先对所有变量进行模型拟合
    while True:
        formula = "{} ~ {} + 1".format(response, ' + '.join(var_list))
        mod = smf.logit(formula, data).fit()
        p_list = mod.pvalues.sort_values()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        # 判断最后一个
        candidate = p_list.index[-1]
        if candidate == 'Intercept':
            c = p_list[-2]
            ind_num = -2
        else:
            c = p_list[-1]
            ind_num = -1
        if c > sls:
            #提取p_list中最后一个index
            var = p_list.index[ind_num]
            #var_list中删除
            
            var_list = var_list.drop(var)           
        else:
            break
    return var_list


def logistic_reg(X, y, constant=True, stepwise=None, sls=0.05):
    """
    model fit
    -----------------------------------------
    Params
    X: pandas dataframe, endogenous variable
    y: pandas series, endogenous variable
    constant：bool, True means add constant
    stepwise: str, variable select,"BS" is backward, "FS" is forward
    sls: float, threshold for variable select metric
    -----------------------------------------
    Return
    logit_instance: instance of logit model
    logit_model: sm model object of logit model
    logit_result: fit results of logit model
    logit_result_0: fit results of logit model(only with constant)
    """
    if stepwise == "FS" and X.shape[1] > 1:
        varlist = _forward_selected_logit(X, y)
        X = X.ix[:,varlist]
    elif stepwise == "BS" and X.shape[1] > 1:
        varlist = _backward_selected_logit(X, y, sls=sls)
        X = X.ix[:,varlist]
    logit_instance = Logistic(X, y)
    logit_model, logit_result, logit_result_0 = logit_instance.modelFit(constant=constant)
    return logit_instance, logit_model, logit_result, logit_result_0

def logit_output(logit_instance, logit_model, logit_result, logit_result_0):
    """
    generate logistic model output
    -------------------------------------------------------
    Params
    logit_instance: instance of logit model
    logit_model: sm model object of logit model
    logit_result: fit results of logit model
    logit_result_0: fit results of logit model(only with constant)
    -------------------------------------------------------
    Return
    desc: describe of model
    params: estimated results
    evaluate: evaluate for model
    quality: model quality metric
    """
    desc = logit_instance.modelDescript(logit_model, logit_result)
    params = ParamEST(logit_result)
    evaluate = fitEval(logit_result, logit_result_0)
    quality = modelQuality(logit_result)
    return desc, params, evaluate, quality




    
