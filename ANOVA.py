import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp # plotting 
import statsmodels.formula.api as sm 
import statsmodels.stats as sms #ANOVA model
import statsmodels.api as sma #api entry 

# Creating the dataframe for the houseing data
Housedb = pd.read_csv('train.csv')
#looking at the first few rows
Housedb.head()

#variable selection 
Deeds = Housedb.loc[Housedb['SaleType'].isin(['New','WD']) & Housedb['KitchenQual'].isin(['Ex','Gd'])]
print(Deeds.shape)
Deeds['SaleType'].value_counts()
Deeds['KitchenQual'].value_counts()
print(Deeds)

House_lm = sm.ols(formula="SalePrice ~ C(SaleType)+C(KitchenQual)+C(SaleType):C(KitchenQual)",data=Deeds).fit()
table = sma.stats.anova_lm(House_lm,typ=1)
print(table)