# Value-at-Risk
### Some codes related to the VaR methodology. Welecome to criticize if anything can be better.
### VaR Calculation Method
The VaR_RollingWIndows.py include the Normal Distribution Method, Exponential Weighted Moving Average Method and Historical Simulation Method for calculate the Value at Risk with rolling windows.
The function can be import and use directly.

NormalVaR(Returns,Confidence_Level,First_Windows)\
EWMAVaR(Returns,Confidence_Level,First_Windows,Decay_Factors)\
HSVaR(Returns,Confidence_Level,First_Windows)
### Expected Shortfall (Conditional Value at Risk)
CVaR(Returns,VaR)
#### Notes: The Returns should be a vector or DataFrame, it will be much more convenience if users can set the index as timestame.
#### Also: The VaR inputed to CVaR function should be the positive number.
