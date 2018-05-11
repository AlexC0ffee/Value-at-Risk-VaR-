# Value-at-Risk
### Some code related to the VaR methodology. Welecome to criticize if anything can be better.
The VaR_RollingWIndows.py include the Normal Distribution Method, Exponential Weighted Moving Average Method and Historical Simulation Method for calculate the Value at Risk with rolling windows.
The function can be import and use directly.

NormalVaR(Returns,Confidence_Level,First_Windows)\
EWMAVaR(Returns,Confidence_Level,First_Windows,Decay_Factors)\
HSVaR(Returns,Confidence_Level,First_Windows)
### Expected Shortfall (Conditional Value at Risk)
CVaR(Returns,VaR)
#### Notes: The Returns should be a n*1 vector or DataFrame, it will be much convenience for later to set the index as timestame. 
