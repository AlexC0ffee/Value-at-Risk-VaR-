#==============================================================================
  #Normal Distribution Method for Value at Risk
  #The 1-Head VaR calculation based on the Equal Weigted standard deviation
  #Use rolling windows by setting the first 1000th observations as 1st window
  #Defined as NormalVaR
#==============================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm

def NormalVaR(Returns,Confidence_Level,First_Windows):
    Time = len(Returns)
    VaR = pd.Series(index=Returns.index, name = 'NormalVaR')
    for i in range(0,Time-First_Windows):
        Data = Returns[:First_Windows+i]
        stdev = np.std(Data)
        VaR[First_Windows+i] = stdev*norm.ppf(Confidence_Level)
    return pd.DataFrame(VaR[First_Windows:])

#==============================================================================
  #Exponential Weighted Moving Average Method for Value at Risk
  #The 1-Head VaR calculation based on the EWMA standard deviation
  #Use rolling windows by setting the first 1000th observations as 1st window
  #Defined as EWMAlVaR
#==============================================================================
def EWMAVaR(Returns,Confidence_Level, First_Windows, Decay_Factors):
    Time = len(Returns)
    VaR = pd.Series(index=Returns.index, name = 'EWMAVaR')
    for i in range(0,Time-First_Windows):
        Data = Returns[:First_Windows+i]
        EWMAstdev = pd.ewma(Data,com=1/Decay_Factors-1).std()
        VaR[First_Windows+i] = EWMAstdev*norm.ppf(Confidence_Level)
    return pd.DataFrame(VaR[First_Windows:])


#==============================================================================
  #Historical Simulation Method for Value at Risk
  #The 1-Head VaR calculation based on the Historical Simulation Method
  #Use rolling windows by setting the first 1000th observations as 1st window
  #Defined as HSVaR
#==============================================================================
def HSVaR(Returns,Confidence_Level,First_Windows):
    Time = len(Returns)
    HSVaR = pd.Series(index=Returns.index, name = 'HSVaR')
    for i in range(0,Time-First_Windows):
        Data = Returns[:First_Windows+i]
        HSVaR[First_Windows+i] = -Data.quantile(1-Confidence_Level)
    return pd.DataFrame(HSVaR[First_Windows:])