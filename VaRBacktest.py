# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:18:10 2018

@author: zhli6157
"""

import pandas as pd
import numpy as np
#==============================================================================
  # Kupiec Uncondition Coverage Backtesting, Proportion of Failures(POF)
  # Defined as UCoverage
  # UCoverage(Returns, Value at Risk, Confidence Level of VaR)
#==============================================================================

def UCoverage(Returns,VaR,P):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    Number_of_Fail=len(Compare.ix[Compare.T.iloc[0]<Compare.T.iloc[1]])
    N = Number_of_Fail
    T = len(Compare)
    t = (1-N/T)**(T-N)*(N/T)**N
    c = ((P)**(T-N))*((1-P)**N)
    Likelihood_Ratio = 2*np.log(t)-2*np.log(c)
    return Likelihood_Ratio
#==============================================================================
def FailRate(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    Number_of_Fail=len(Compare.ix[Compare.T.iloc[0]<Compare.T.iloc[1]])
    N = Number_of_Fail
    T = len(Compare)
    FailRate = N/T
    return FailRate 
#==============================================================================
  # Christoffersen's Interval Forecast Tests, Conditional Coverage Backtesting
  # Defined as LRCCI
  # LRCCI(Returns, Value at Risk, Confidence Level of VaR)
#==============================================================================
def LRCCI(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    LRCC = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    TF=LRCC.T.iloc[0]>LRCC.T.iloc[1]
    n00=0
    n10=0
    n01=0
    n11=0
    for i in range(len(TF)-1):
        if TF[i] == True and TF[i+1] == True:
            n00 = n00+1
    for m in range(len(TF)-1):
        if TF[m] == False and TF[m+1] == True:
            n10 = n10+1
    for q in range(len(TF)-1):
        if TF[q] == True and TF[q+1] == False:
            n01 = n01+1
    for f in range(len(TF)-1):
        if TF[f] == False and TF[f+1] == False:
            n11 = n11+1
    
    pi0= n01/(n00+n01)
    pi1 = n11/(n10+n11)
    pi = (n01+n11)/(n00+n01+n10+n11)
    Numeritor = ((1-pi)**(n00+n10))*(pi**(n01+n11))
    Denominator = ((1-pi0)**(n00))*(pi0**n01)*((1-pi1)**(n10))*(pi1**n11)
    LRCCI = -2*np.log(Numeritor/Denominator)
    return LRCCI

#==============================================================================
  # Regulator's Loss Function Family
  # Mathmatical Reference: The role of the loss function in value-at-risk comparisons
  # The score for the complete sample is the sum of each individual point
#==============================================================================
  # Lopez's quadratic (RQL)
  # Defined as RQL
  # RQL(Returns, Value at Risk)
#==============================================================================  
def RQL(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RQL = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = 1+(Compare.T.iloc[1][i]-Compare.T.iloc[0][i])**2
        else:
            quadratic = 0
        RQL.append(quadratic)
        RQL_Score = np.sum(RQL)
    return RQL_Score
#==============================================================================
  # Linear (RL)
  # Defined as RL
  # RL(Returns, Value at Risk)
#==============================================================================
def RL(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RL = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = (Compare.T.iloc[1][i]-Compare.T.iloc[0][i])
        else:
            quadratic = 0
        RL.append(quadratic)
        RL_Score = np.sum(RL)
    return RL_Score        
#==============================================================================
  # Quadratic (RQ)
  # Defined as RQ
  # RQ(Returns, Value at Risk)
#==============================================================================
def RQ(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RQ = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = (Compare.T.iloc[1][i]-Compare.T.iloc[0][i])**2
        else:
            quadratic = 0
        RQ.append(quadratic)
        RQ_Score = np.sum(RQ)
    return RQ_Score
#==============================================================================
  # Caporin_1 (RC_1)
  # Defined as RC_1
  # RC_1(Returns, Value at Risk)
#==============================================================================
def RC_1(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RC_1 = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = np.abs(1-np.abs(Compare.T.iloc[0][i]/Compare.T.iloc[1][i]))
        else:
            quadratic = 0
        RC_1.append(quadratic)
        RC_1_Score = np.sum(RC_1)
    return RC_1_Score
#==============================================================================
  # Caporin_2 (RC_2)
  # Defined as RC_2
  # RC_2(Returns, Value at Risk)
#==============================================================================
def RC_2(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RC_2 = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = (np.abs(Compare.T.iloc[0][i])-np.abs(Compare.T.iloc[1][i]))**2/(np.abs(Compare.T.iloc[1][i]))
        else:
            quadratic = 0
        RC_2.append(quadratic)
        RC_2_Score = np.sum(RC_2)
    return RC_2_Score
#==============================================================================
  # Caporin_3 (RC_3)
  # Defined as RC_3
  # RC_3(Returns, Value at Risk)
#==============================================================================
def RC_3(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    RC_3 = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            quadratic = (np.abs(Compare.T.iloc[1][i]-Compare.T.iloc[0][i]))
        else:
            quadratic = 0
        RC_3.append(quadratic)
        RC_3_Score = np.sum(RC_3)
    return RC_3_Score
#==============================================================================
  # Firm's Loss Function Family
  # Mathmatical Reference: The role of the loss function in value-at-risk comparisons
#==============================================================================
#==============================================================================
  # Caporin_1 (FC_1)
  # Defined as FC_1
  # FC_1(Returns, Value at Risk)
#==============================================================================
def FC_1(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    FC_1 = []
    for i in range(len(VaR)):
        quadratic = np.abs(1-np.abs(Compare.T.iloc[0][i]/Compare.T.iloc[1][i]))                    
        FC_1.append(quadratic)
        FC_1_Score = np.sum(FC_1)
    return FC_1_Score
#==============================================================================
  # Caporin_2 (FC_2)
  # Defined as FC_2
  # FC_2(Returns, Value at Risk)
#==============================================================================
def FC_2(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    FC_2 = []
    for i in range(len(VaR)):
        quadratic = (np.abs(Compare.T.iloc[0][i])-np.abs(Compare.T.iloc[1][i])**2)/np.abs(Compare.T.iloc[1][i])                  
        FC_2.append(quadratic)
        FC_2_Score = np.sum(FC_2)
    return FC_2_Score
#==============================================================================
  # Caporin_3 (FC_3)
  # Defined as FC_3
  # FC_3(Returns, Value at Risk)
#==============================================================================
def FC_3(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    FC_3 = []
    for i in range(len(VaR)):
        quadratic = np.abs(Compare.T.iloc[1][i]-Compare.T.iloc[0][i])                
        FC_3.append(quadratic)
        FC_3_Score = np.sum(FC_3)
    return FC_3_Score

#==============================================================================
  # Quantile Loss Function
  # Reference: The Use of GARCH Models in VaR Estimation.
  # Defined as QL
  # Ql(Returns, Value at Risk, Condidence Level of VaR)
#==============================================================================
def QL(Returns,VaR,ConfidenceLevel):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    Compare = pd.concat([Returns[First_Windows:],-VaR],axis=1)
    QL = []
    for i in range(len(VaR)):
        if Compare.T.iloc[0][i] < Compare.T.iloc[1][i]:
            QuantileLoss = (Compare.T.iloc[0][i]-Compare.T.iloc[1][i])**2
        else:
            QuantileLoss = (Compare.T.iloc[0][-i-1:].quantile(1-ConfidenceLevel)-Compare.T.iloc[1][i])**2
        QL.append(QuantileLoss)
        QL_Score = np.sum(QL)
    return QL_Score  