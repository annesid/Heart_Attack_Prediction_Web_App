#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:33:02 2022

@author: anne
"""
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import numpy as np

class EDA:
    def displot_graph(self,cont,df):
        for i in cont: #using for loop
            plt.figure()
            sns.distplot(df[i])
            plt.show()
            
    def countplot_graph(self,cat,df):
        for i in cat: #using for loop
            plt.figure()
            sns.countplot(df[i])
            plt.show()
    
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher, 
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

