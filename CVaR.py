# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:38:59 2018

@author: Administrator
"""
import pandas as pd
#==============================================================================
  #Expected Shortfall(CVaR)
  #The 1-Head CVaR calculation based on VaR provided
  #Use rolling windows by setting the n observations as 1st window
  #Defined as CVaR
#==============================================================================

def CVaR(Returns,VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    CVaR = pd.Series(index=Returns.index, name = 'CVaR')
    for i in range(0,Time-First_Windows):
        VaR_Return = pd.concat([Returns[First_Windows:],VaR],axis=1)
        Expected_Shortfall = (VaR_Return[:i+1].ix[VaR_Return[:i+1].T.iloc[0]<VaR_Return[:i+1].T.iloc[1]]).T.iloc[0].mean()
        CVaR[First_Windows+i] = Expected_Shortfall
    return pd.DataFrame(CVaR[First_Windows:])
