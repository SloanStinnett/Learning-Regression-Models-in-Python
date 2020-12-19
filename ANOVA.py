import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as mp # plotting 
import statsmodels.formula.api as sm 
import statsmodels.stats as sms #ANOVA model
import statsmodels.api as sma #api entry 
import statsmodels.stats.multicomp import pairwise_tukeysd

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

#runnning the factorial ANOVA 
House_lm = sm.ols(formula="SalePrice ~ C(SaleType)+C(KitchenQual)+C(SaleType):C(KitchenQual)",data=Deeds).fit()
table = sma.stats.anova_lm(House_lm,typ=2)
print(table)

# running a tukey test on the data to measure the main effects of kitchen quality and sale type
SaleTypeEta = table['sum_sq'][0]/table['sum_sq'][3]
print(SaleTypeEta)

KitQualEta = table['sum_sq'][1]/table['sum_sq'][3]
print(KitQualEta)