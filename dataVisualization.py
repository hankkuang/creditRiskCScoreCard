# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:42:46 2017

@author: Hank Kuang
@title: 统计图形
"""

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import numpy as np


def drawPie(s, labels=None, dropna=True):
    """
    Pie Plot for s
    -------------------------------------
    Params
    s: pandas Series
    lalels:labels of each unique value in s
    dropna:bool obj
    -------------------------------------
    Return
    show the plt object
    """
    counts = s.value_counts(dropna=dropna)
    if labels is None:
        labels = counts.index
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def drawBar(s, x_ticks=None, pct=False, dropna=False, horizontal=False):
    """
    bar plot for s
    -------------------------------------------
    Params
    s: pandas Series
    x_ticks: list, ticks in X axis
    pct: bool, True means trans data to odds
    dropna: bool obj,True means drop nan
    horizontal: bool, True means draw horizontal plot
    -------------------------------------------
    Return
    show the plt object
    """
    counts = s.value_counts(dropna=dropna)
    if pct == True:
        counts = counts/s.shape[0]
    ind = np.arange(counts.shape[0])
    if x_ticks is None:
        x_ticks = counts.index
    
    if horizontal == False:
        p = plt.bar(ind, counts)
        plt.ylabel('frequecy')
        plt.xticks(ind, tuple(counts.index))
    else:
        p = plt.barh(ind, counts)
        plt.xlabel('frequecy')
        plt.yticks(ind, tuple(counts.index))
    plt.title('Bar plot for %s' % s.name)
    
    plt.show()


def drawHistogram(s, num_bins=20, save=False, filename='myHist'):
    """
    plot histogram for s
    ---------------------------------------------
    Params
    s: pandas series
    num_bins: number of bins
    save: bool, is save? 
    filename: png name
    ---------------------------------------------
    Return
    show the plt object
    """
    fig, ax = plt.subplots()
    mu = s.mean()
    sigma = s.std()
    # the histogram of the data
    n, bins, patches = ax.hist(s, num_bins, normed=1)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel(s.name)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (s.name, mu, sigma))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    if save:
        plt.savefig(filename+'.png')
    plt.show()
    


    
    

