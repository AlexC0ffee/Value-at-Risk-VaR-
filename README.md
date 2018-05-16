# Value-at-Risk
### Some codes related to the VaR methodology. Welecome to criticize if anything can be better.
### VaR Calculation Method
The VaR_RollingWIndows.py include the Normal Distribution Method, Exponential Weighted Moving Average Method and Historical Simulation Method for calculate the Value at Risk with expanding windows.
The function can be import and use directly.

NormalVaR(Returns,Confidence_Level,First_Windows)\
EWMAVaR(Returns,Confidence_Level,First_Windows,Decay_Factors)\
HSVaR(Returns,Confidence_Level,First_Windows)
### Value at Risk(VaR) Backtesting
#### VaRBacktest.py
The Backtest methods include Unconditional Coverage, Conditional Coverage, Loss Function and Quantile Loss Function.
##### Kupiec's Unconditional Coverage, Proportion of Failures(POF)
UCoverage(Returns,VaR,expected signifiance level P)
FailRate(Returns,Value at Risk)
##### Christiffersen's Conditional Coverage
LRCCI(Returns,Value at Risk)
##### Loss Function
###### Regulator's Loss Function Family include Lopez's Quadratic(RQL), Linear(RL), Quadratic(RQ), Caporin_1(RC_1), Caporin_2(RC_2), Caporin_3(RC_3).
RQL(Returns,VaR)
RL(Returns, VaR)
RQ(Returns,VaR)
RC_1(Returns,VaR)
RC_2(Returns,VaR)
RC_3(Returns,VaR)
###### Firm's Loss Function Family include Caporin_1(FC_1), Caporin_2(FC_2), Caporin_3(FC_3).
FC_1(Returns,VaR)
FC_2(Returns,VaR)
FC_3(Returns,VaR)
##### Quantile Loss Function
QL(Returns, VaR, ConfidenceLevel)
### Expected Shortfall (Conditional Value at Risk)
#### CVaR.py
CVaR(Returns,VaR)
#### Notes: The Returns should be a vector or DataFrame, it will be much more convenience if users can set the index as timestame.
#### Also: The VaR inputed to CVaR function should be the positive number.
