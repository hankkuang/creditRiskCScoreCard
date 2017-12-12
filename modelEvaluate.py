# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:05:45 2017

@author: Hank Kuang
@title: 模型评价(classifer model)
"""

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd
from sklearn.utils.multiclass import type_of_target
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score


"""
ROC曲线绘制脚本
"""
def plot_roc_curve(prob_y, y):
    """
    plot roc curve
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    ----------------------------------
    plt object
    """
    fpr, tpr, _ = roc_curve(y, prob_y)
    c_stats = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label="ROC curve")
    s = "AUC = %.4f" % c_stats
    plt.text(0.6, 0.2, s, bbox=dict(facecolor='red', alpha=0.5))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')#ROC 曲线
    plt.legend(loc='best')
    plt.show()

    
"""
KS曲线以及KS统计表
"""
def ks_stats(prob_y, y, k=20):
    """
    plot K-S curve and output ks table
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number 
    ----------------------------------
    ks_results: pandas dataframe 
    ks_ax: plt object, k-s curcve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(prob_y, name='prob_y', index=y.index)], axis=1)
    datasets.columns = ["y", "prob_y"]
    datasets = datasets.sort_values(by="prob_y", axis=0, ascending=True)
    # 计算正负案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    N = Nrows - P
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    ks_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
            "Ntot":0}
    ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['y'])
        Ntot = n-Ptot
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                "Ntot":Ntot}
        ks_df = ks_df.append(pd.Series(rlt), ignore_index=True)
    # 计算各子集中的正负例比例,以及累积比例
    ks_df['PerP'] = ks_df['Ptot']/P
    ks_df['PerN'] = ks_df['Ntot']/N
    ks_df['PerP_cum'] = ks_df['PerP'].cumsum()
    ks_df['PerN_cum'] = ks_df['PerN'].cumsum()
    # 计算ks曲线以及ks值
    ks_df['ks'] = ks_df['PerN_cum'] - ks_df['PerP_cum']
    ks_value = ks_df['ks'].max()
    s = "KS value is %.4f" % ks_value
    # 整理得出ks统计表
    ks_results = ks_df.ix[1:,:]
    ks_results = ks_results[['tile', 'Ntot', 'Ptot', 'PerN', 'PerP', 'PerN_cum', 'PerP_cum', 'ks']]
    ks_results.columns = ['子集','负例数','正例数','负例比例','正例比例','累积负例比例','累积正例比例', 'ks']
    # 获取ks值所在的数据点
    ks_point = ks_results.ix[:,['子集','ks']]
    ks_point = ks_point.ix[ks_point['ks']==ks_point['ks'].max(),:]
    # 绘制KS曲线
    ks_ax = _ks_plot(ks_df=ks_df, ks_label='ks', good_label='PerN_cum', bad_label='PerP_cum', 
                    k=k, point=ks_point, s=s)
    return ks_results, ks_ax


def _ks_plot(ks_df, ks_label, good_label, bad_label, k, point, s):
    """
    middle function for ks_stats, plot k-s curve
    """
    ks_df['tile'] = ks_df['tile'].astype(np.int32)
    plt.plot(ks_df['tile'], ks_df[ks_label], "r-.", label="ks_curve", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[good_label], "g-.", label="good", lw=1.2)
    plt.plot(ks_df['tile'], ks_df[bad_label], "m-.", label="bad", lw=1.2)
    #plt.plot(point[0], point[1], 'o', markerfacecolor="red",
             #markeredgecolor='k', markersize=6)
    plt.legend(loc=0)
    plt.plot([0, k], [0, 1], linestyle='--', lw=0.8, color='k', label='Luck')
    plt.xlabel("decilis")#等份子集
    plt.title(s)#KS曲线图
    plt.show()    

"""
混淆矩阵
"""
def plot_confusion_matrix(y, 
                          pred, 
                          labels,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    plot confusion_matrix with colorbar
    ------------------------------------------
    Params
    y：real data labels
    pred: predict results
    labels: labels
    normalize: bool, True means trans results to percent
    cmap: color index
    ------------------------------------------
    Return
    plt object
    """
    cm = confusion_matrix(y, pred, labels=labels)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)#在指定的轴上展示图像
    
    plt.colorbar()#增加色柱
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)#设置坐标轴标签
    plt.yticks(tick_marks, labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("标准化混淆矩阵")
    else:
        #print('混淆矩阵')
        pass
    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title("confusion matrix")
    plt.show()

"""
提升图和洛伦茨曲线
"""
def lift_lorenz(prob_y, y, k=10):
    """
    plot lift_lorenz curve 
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number 
    ----------------------------------
    lift_ax: lift chart
    lorenz_ax: lorenz curve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(prob_y, name='prob_y', index=y.index)], axis=1)
    datasets.columns = ["y", "prob_y"]
    datasets = datasets.sort_values(by="prob_y", axis=0, ascending=False)
    # 计算正案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    lift_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
          }
    lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['y'])
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    # 计算正例比例&累积正例比例
    lift_df['PerP'] = lift_df['Ptot']/P
    lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
    # 计算随机正例数、正例率以及累积随机正例率
    lift_df['randP'] = float(P)/k
    lift_df['PerRandP'] = lift_df['randP']/P
    lift_df.ix[0,:]=0
    lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
    lift_ax = lift_Chart(lift_df, k)
    lorenz_ax = lorenz_cruve(lift_df)
    return lift_ax, lorenz_ax


def lift_Chart(df, k):
    """
    middle function for lift_lorenz, plot lift Chart
    """
    #绘图变量
    PerP = df['PerP'][1:]
    PerRandP = df['PerRandP'][1:]
    #绘图参数
    fig, ax = plt.subplots()
    index = np.arange(k+1)[1:]
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, PerP, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Per_p')#正例比例
    rects2 = plt.bar(index + bar_width, PerRandP, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='random_P')#随机比例
    plt.xlabel('Group')
    plt.ylabel('Percent')
    plt.title('lift_Chart')
    plt.xticks(index + bar_width / 2, tuple(index))
    plt.legend()
    plt.tight_layout()
    plt.show()

def lorenz_cruve(df):
    """
    middle function for lift_lorenz, plot lorenz cruve
    """
    #准备绘图所需变量
    PerP_cum = df['PerP_cum']
    PerRandP_cum = df['PerRandP_cum']
    decilies = df['tile']
    #绘制洛伦茨曲线
    plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')#lorenz曲线
    plt.plot(decilies, PerRandP_cum, 'k-.', label='random')#随机
    plt.legend()
    plt.xlabel("decilis")#等份子集
    plt.title("lorenz_cruve", fontsize=10)#洛伦茨曲线
    plt.show()  