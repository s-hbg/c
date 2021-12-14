#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from IPython.core.interactiveshell import InteractiveShell
import warnings
warnings.filterwarnings('ignore')


# In[2]:


test = pd.read_csv('D:/house-prices-advanced-regression-techniques/test.csv')
test


# In[3]:


train = pd.read_csv('D:/house-prices-advanced-regression-techniques/train.csv')
train


# In[4]:


# 给出样本数据的相关信息概览 ：行数，列数，列索引，列非空值个数，列类型
train.info()


# In[5]:


# 直接给出样本数据的一些基本的统计量
train.describe()


# In[6]:


# 设置显示最大行数
pd.set_option('display.max_rows', 500)
# 设置显示最大列数
pd.set_option('display.max_columns', 500)
# pandas设置显示宽度
pd.set_option('display.width', 1000)
# 筛选出训练集所有含有缺失值的列
train.isnull().any()


# In[7]:


# 计算训练集中每列有多少缺失值
na = train.isnull().sum().sort_values(ascending=False).head(19)


# In[8]:


# 绘制柱状图
na.plot.bar()


# In[9]:


# 删除大量数据缺失的列
drop_cols = ['LotFrontage', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
train.drop(columns = drop_cols, axis=1, inplace=True)
test.drop(columns=drop_cols, axis=1, inplace=True)


# In[10]:


# 处理空值和缺失值

# 空值在GarageType, GarageFinish, GarageQual 以及 GarageCond这些属性中意味着没有停车场
# 故可以用'No Garage'来代替'NA'
train['GarageType'].fillna('No Garage', inplace=True)
train['GarageFinish'].fillna('No Garage', inplace=True)
train['GarageQual'].fillna('No Garage', inplace=True)
train['GarageCond'].fillna('No Garage', inplace=True)         
# 空值在BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual 意味着没有地下室 
# 故可以用''No Basement'来代替'NA'
train['BsmtFinType2'].fillna('No Basement', inplace=True)
train['BsmtExposure'].fillna('No Basement', inplace=True)         
train['BsmtFinType1'].fillna('No Basement', inplace=True)         
train['BsmtCond'].fillna('No Basement', inplace=True)         
train['BsmtQual'].fillna('No Basement', inplace=True)
train['MasVnrType'].fillna('None', inplace=True)
train['MasVnrArea'].fillna(0, inplace=True)
# 将年份转换为存在年数
train['GarageYrBlt'] = 2021 - train['GarageYrBlt']
train['YearBuilt'] = 2021 - train['YearBuilt']
train['YearRemodAdd'] = 2021 - train['YearRemodAdd']
train['YrSold'] = 2021 - train['YrSold']
# 向GarageYrBlt的缺失值 填写-1，代表房子没有停车场
train['GarageYrBlt'].fillna(-1, inplace=True)
# 由于Electrical中主要为SBrkr，所以填写SBrkr
train['Electrical'].fillna('SBrkr', inplace=True) 

test['GarageType'].fillna('No Garage', inplace=True)
test['GarageFinish'].fillna('No Garage', inplace=True)
test['GarageQual'].fillna('No Garage', inplace=True)
test['GarageCond'].fillna('No Garage', inplace=True)
test['BsmtFinType2'].fillna('No Basement', inplace=True)
test['BsmtExposure'].fillna('No Basement', inplace=True)         
test['BsmtFinType1'].fillna('No Basement', inplace=True)         
test['BsmtCond'].fillna('No Basement', inplace=True)         
test['BsmtQual'].fillna('No Basement', inplace=True)
test['MasVnrType'].fillna('None', inplace=True)
test['MasVnrArea'].fillna(0, inplace=True)

test['GarageYrBlt'] = 2021 - train['GarageYrBlt']
test['YearBuilt'] = 2021 - train['YearBuilt']
test['YearRemodAdd'] = 2021 - train['YearRemodAdd']
test['YrSold'] = 2021 - train['YrSold']

test['GarageYrBlt'].fillna(-1, inplace=True)


# In[11]:


train.shape


# In[12]:


test.shape


# In[13]:


# 检查训练集是否还有缺失值
train.isnull().any()


# In[14]:


# 筛选出测试集所有含有缺失值的列
test.isnull().any()


# In[15]:


# 计算测试集中每列有多少缺失值
test.isnull().sum().sort_values(ascending=False).head(15)


# In[16]:


# 合并训练集和测试集数据，同时处理数据
feature_train=train.drop('SalePrice',axis=1)
feature_test=test
all_data=pd.concat([feature_train,feature_test])
all_data.reset_index(drop=True,inplace=True)


# In[17]:


# 用众数填补缺失值
test['MSZoning'].fillna(all_data['MSZoning'].mode()[0], inplace=True)
test['Functional'].fillna(all_data['Functional'].mode()[0], inplace=True)
test['BsmtHalfBath'].fillna(all_data['BsmtHalfBath'].mode()[0], inplace=True)
test['Utilities'].fillna(all_data['Utilities'].mode()[0], inplace=True)
test['BsmtFullBath'].fillna(all_data['BsmtFullBath'].mode()[0], inplace=True)
test['GarageCars'].fillna(all_data['GarageCars'].mode()[0], inplace=True)
test['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0], inplace=True)
test['SaleType'].fillna(all_data['SaleType'].mode()[0], inplace=True)
test['BsmtFinSF2'].fillna(all_data['BsmtFinSF2'].mode()[0], inplace=True)
test['BsmtUnfSF'].fillna(all_data['BsmtUnfSF'].mode()[0], inplace=True)
test['TotalBsmtSF'].fillna(all_data['TotalBsmtSF'].mode()[0], inplace=True)
test['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0], inplace=True)
test['BsmtFinSF1'].fillna(all_data['BsmtFinSF1'].mode()[0], inplace=True)
test['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0], inplace=True)


# In[18]:


# 删除存在缺失值的列
train.dropna(axis=0, inplace=True)
# 重复值的处理
train.drop_duplicates()
train.shape


# In[19]:


test.shape


# In[20]:


# 异常值的处理
num_col = list(train.dtypes[train.dtypes != 'object'].index)
def drop_outliers(x):
    list = []
    for col in num_col:
        Q1 = x[col].quantile(.25)
        Q3 = x[col].quantile(.99)
        IQR = Q3 - Q1
        x = x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))]
    return x
train = drop_outliers(train)


# In[21]:


train.describe()


# In[22]:


# 删除PoolArea列，因为删除异常值后所有值均为0
train.drop(columns=['PoolArea'], inplace=True)
test.drop(columns=['PoolArea'], inplace=True)


# In[23]:


# 移除ID列
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train.shape


# In[24]:


test.shape


# In[25]:


# 查看函数是否符合正态分布
plt.title('SalePrice')
sns.distplot(train['SalePrice'], bins=10)
plt.show()


# In[26]:


# 取对数
train['SalePrice'] = np.log1p(train['SalePrice'])
plt.title('SalePrice')
sns.distplot(train['SalePrice'])
plt.show()


# In[27]:


# 获取值为数值信息的列
num_vars = list(train.dtypes[train.dtypes != 'object'].index)
num_vars


# In[28]:


plt.figure(figsize=(10,5))
sns.pairplot(train, x_vars=['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt']
             , y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF'],
             y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea'], 
             y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr'],
             y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch'],
             y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold','YrSold'],
             y_vars=['SalePrice'], kind='scatter')
sns.pairplot(train, x_vars=['KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt'],
             y_vars=['SalePrice'], kind='scatter')
plt.show()


# In[29]:


# Heatmap
plt.figure(figsize = (20, 20))
sns.heatmap(train[num_vars].corr(), annot=True)
plt.show()


# In[30]:


# 根据散点图删除一些属性并控制一些属性的范围
train.drop(columns=['MSSubClass', 'LowQualFinSF', 'MiscVal', '3SsnPorch', 'KitchenAbvGr','BsmtFullBath','BsmtHalfBath',], inplace=True)
train['LotArea'].clip(0,60000)
train['GrLivArea'].clip(0,3000)
train['TotalBsmtSF'].clip(0,3000)
train['BsmtFinSF1'].clip(0,2500)
train['1stFlrSF'].clip(0,3000)
train['GarageArea'].clip(0,1200)
train['OpenPorchSF'].clip(0,400)
train.shape


# In[31]:


test.drop(columns=['MSSubClass', 'LowQualFinSF', 'MiscVal', '3SsnPorch', 'KitchenAbvGr','BsmtFullBath','BsmtHalfBath'], inplace=True)
test['LotArea'].clip(0,60000)
test['GrLivArea'].clip(0,3000)
test['TotalBsmtSF'].clip(0,3000)
test['BsmtFinSF1'].clip(0,2500)
test['1stFlrSF'].clip(0,3000)
test['GarageArea'].clip(0,1200)
test['OpenPorchSF'].clip(0,400)
test.shape


# In[32]:


# 根据热力图删除一些列
train.drop(columns=['GarageArea','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','FullBath','GarageYrBlt', 'YearRemodAdd'], inplace=True)
test.drop(columns=['GarageArea','TotRmsAbvGrd','1stFlrSF','2ndFlrSF','FullBath','GarageYrBlt', 'YearRemodAdd'], inplace=True)


# In[33]:


# 取对数
train['BsmtFinSF2'] = np.log1p(train['BsmtFinSF2'])
train['ScreenPorch'] = np.log1p(train['ScreenPorch'])
train['LotArea'] = np.log1p(train['LotArea'])
train['EnclosedPorch'] = np.log1p(train['EnclosedPorch'])
train['MasVnrArea'] = np.log1p(train['MasVnrArea'])
train['OpenPorchSF'] = np.log1p(train['OpenPorchSF'])
train['WoodDeckSF'] = np.log1p(train['WoodDeckSF'])
train['GrLivArea'] = np.log1p(train['GrLivArea'])
train['BsmtUnfSF'] = np.log1p(train['BsmtUnfSF'])
train.shape


# In[34]:


test['BsmtFinSF2'] = np.log1p(test['BsmtFinSF2'])
test['ScreenPorch'] = np.log1p(test['ScreenPorch'])
test['LotArea'] = np.log1p(test['LotArea'])
test['EnclosedPorch'] = np.log1p(test['EnclosedPorch'])
test['MasVnrArea'] = np.log1p(test['MasVnrArea'])
test['OpenPorchSF'] = np.log1p(test['OpenPorchSF'])
test['WoodDeckSF'] = np.log1p(test['WoodDeckSF'])
test['GrLivArea'] = np.log1p(test['GrLivArea'])
test['BsmtUnfSF'] = np.log1p(test['BsmtUnfSF'])
test['BsmtFinSF1'] = np.log1p(test['BsmtFinSF1'])
test.shape


# In[35]:


# 获取值为object信息的列
obj_vars = list(train.dtypes[train.dtypes == 'object'].index)
obj_vars


# In[36]:


plt.figure(figsize = (15,20)) 
plt.subplot(4,4,1)
sns.boxplot(x='MSZoning', y='SalePrice', data=train)
plt.subplot(4,4,2)
sns.boxplot(x='Street', y='SalePrice', data=train)
plt.subplot(4,4,3)
sns.boxplot(x='LotShape', y='SalePrice', data=train)
plt.subplot(4,4,4)
sns.boxplot(x='LandContour', y='SalePrice',  data=train)
plt.subplot(4,4,5)
sns.boxplot(x='Utilities', y='SalePrice', data=train)
plt.subplot(4,4,6)
sns.boxplot(x='LotConfig', y='SalePrice', data=train)
plt.subplot(4,4,7)
sns.boxplot(x='LandSlope', y='SalePrice', data=train)
plt.subplot(4,4,8)
sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
plt.subplot(4,4,9)
sns.boxplot(x='Condition1', y='SalePrice', data=train)
plt.subplot(4,4,10)
sns.boxplot(x='Condition2', y='SalePrice', data=train)
plt.subplot(4,4,11)
sns.boxplot(x='BldgType', y='SalePrice', data=train)
plt.subplot(4,4,12)
sns.boxplot(x='HouseStyle', y='SalePrice',  data=train)
plt.subplot(4,4,13)
sns.boxplot(x='RoofStyle', y='SalePrice', data=train)
plt.subplot(4,4,14)
sns.boxplot(x='RoofMatl', y='SalePrice', data=train)
plt.subplot(4,4,15)
sns.boxplot(x='Exterior1st', y='SalePrice', data=train)
plt.subplot(4,4,16)
sns.boxplot(x='Exterior2nd', y='SalePrice', data=train)
plt.show()


# In[37]:


plt.figure(figsize = (15,20)) 
plt.subplot(3,4,1)
sns.boxplot(x='MasVnrType', y='SalePrice', data=train)
plt.subplot(3,4,2)
sns.boxplot(x='ExterQual', y='SalePrice', data=train)
plt.subplot(3,4,3)
sns.boxplot(x='ExterCond', y='SalePrice', data=train)
plt.subplot(3,4,4)
sns.boxplot(x='Foundation', y='SalePrice',  data=train)
plt.subplot(3,4,5)
sns.boxplot(x='BsmtQual', y='SalePrice', data=train)
plt.subplot(3,4,6)
sns.boxplot(x='BsmtCond', y='SalePrice', data=train)
plt.subplot(3,4,7)
sns.boxplot(x='BsmtExposure', y='SalePrice', data=train)
plt.subplot(3,4,8)
sns.boxplot(x='BsmtFinType1', y='SalePrice', data=train)
plt.subplot(3,4,9)
sns.boxplot(x='BsmtFinType2', y='SalePrice', data=train)
plt.subplot(3,4,10)
sns.boxplot(x='Heating', y='SalePrice', data=train)
plt.subplot(3,4,11)
sns.boxplot(x='HeatingQC', y='SalePrice', data=train)
plt.subplot(3,4,12)
sns.boxplot(x='CentralAir', y='SalePrice',  data=train)


# In[38]:


plt.figure(figsize = (15,15)) 
plt.subplot(3,3,1)
sns.boxplot(x='Electrical', y='SalePrice', data=train)
plt.subplot(3,3,2)
sns.boxplot(x='KitchenQual', y='SalePrice', data=train)
plt.subplot(3,3,3)
sns.boxplot(x='Functional', y='SalePrice', data=train)
plt.subplot(3,3,4)
sns.boxplot(x='GarageType', y='SalePrice', data=train)
plt.subplot(3,3,5)
sns.boxplot(x='GarageFinish', y='SalePrice', data=train)
plt.subplot(3,3,6)
sns.boxplot(x='GarageQual', y='SalePrice', data=train)
plt.subplot(3,3,7)
sns.boxplot(x='GarageCond', y='SalePrice', data=train)
plt.subplot(3,3,8)
sns.boxplot(x='PavedDrive', y='SalePrice', data=train)
plt.subplot(3,3,9)
sns.boxplot(x='SaleType', y='SalePrice', data=train)
plt.show()
plt.figure(figsize = (15,5)) 
plt.subplot(1,2,1)
sns.boxplot(x='SaleCondition', y='SalePrice', data=train)


# In[39]:


# 根据箱线图删除Utilities
train.drop(columns=['Utilities'], inplace=True) 
test.drop(columns=['Utilities'], inplace=True)


# In[40]:


# 将object转换为数值型
cvars = list(train.dtypes[train.dtypes == 'object'].index)
train[cvars].head()


# In[41]:


# object信息数值化
train['LandSlope'] = train.LandSlope.map({'Sev':0,'Mod':1,'Gtl':2})
train['ExterQual'] = train.ExterQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
train['BsmtQual'] = train.BsmtQual.map({'No Basement':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train['BsmtCond'] = train.BsmtCond.map({'No Basement':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train['BsmtExposure'] = train.BsmtExposure.map({'No Basement':0,'No':1,'Mn':2,'Av':3,'Gd':4})
train['BsmtFinType1'] = train.BsmtFinType1.map({'No Basement':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
train['BsmtFinType2'] = train.BsmtFinType2.map({'No Basement':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
train['HeatingQC'] = train.HeatingQC.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
train['CentralAir'] = train.CentralAir.map({'N':0,'Y':1})
train['KitchenQual'] = train.KitchenQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
train['GarageFinish'] = train.GarageFinish.map({'No Garage':0,'Unf':1,'RFn':2,'Fin':3})
train['GarageQual'] = train.GarageQual.map({'No Garage':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train['GarageCond'] = train.GarageCond.map({'No Garage':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
train['ExterCond'] = train.ExterCond.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
train['LotShape'] = train.LotShape.map({'IR1':0,'IR2':1,'IR3':2,'Reg':3})

test['LandSlope'] = test.LandSlope.map({'Sev':0,'Mod':1,'Gtl':2})
test['ExterQual'] = test.ExterQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
test['BsmtQual'] = test.BsmtQual.map({'No Basement':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test['BsmtCond'] = test.BsmtCond.map({'No Basement':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test['BsmtExposure'] = test.BsmtExposure.map({'No Basement':0,'No':1,'Mn':2,'Av':3,'Gd':4})
test['BsmtFinType1'] = test.BsmtFinType1.map({'No Basement':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
test['BsmtFinType2'] = test.BsmtFinType2.map({'No Basement':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
test['HeatingQC'] = test.HeatingQC.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
test['CentralAir'] = test.CentralAir.map({'N':0,'Y':1})
test['KitchenQual'] = test.KitchenQual.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
test['GarageFinish'] = test.GarageFinish.map({'No Garage':0,'Unf':1,'RFn':2,'Fin':3})
test['GarageQual'] = test.GarageQual.map({'No Garage':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test['GarageCond'] = test.GarageCond.map({'No Garage':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})
test['ExterCond'] = test.ExterCond.map({'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})
test['LotShape'] = test.LotShape.map({'IR1':0,'IR2':1,'IR3':2,'Reg':3})


# In[42]:


# 对单列属性进行数值标准化
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
for col in train:
   if train[col].dtype=="object":
       train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
train


# In[43]:


enc = OrdinalEncoder()
for col in test:
   if test[col].dtype=="object":
       test[col] = enc.fit_transform(test[col].astype(str).values.reshape(-1,1))
test


# In[44]:


# 确保测试集中列的顺序与训练组中列的顺序相同
train, test = train.align(test, axis=1)


# In[45]:


# 分割数据
X = train.drop(['SalePrice'], axis=1)
Y = train['SalePrice']
# 标准化
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# In[46]:


cols = test.columns
test = pd.DataFrame(scale(test))
test.columns = cols
test.columns


# In[47]:


# 拆分数据集
np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42)


# In[48]:


lm = LinearRegression()
lm.fit(X_train,Y_train)
# 保留30个最重要的属性
rfe = RFE(lm, 30)
lm_rfe = rfe.fit(X_train, Y_train)


# In[49]:


# 显示属性的排名，其中排名为1的保留
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[50]:


Y_pred_lm_rfe_train = lm_rfe.predict(X_train)
Y_pred_lm_rfe = lm_rfe.predict(X_test)


# In[51]:


# 计算 MAE（MAE评估的是真实值和预测值的偏离程度：小）
# 计算 MSE（误差平方的平均值：小）
# 计算 RMSE（均误差平方根：小）
print("MAE = ", mean_absolute_error(Y_test, Y_pred_lm_rfe))
print("MSE = ", mean_squared_error(Y_test, Y_pred_lm_rfe))
print("RMSE = ", np.sqrt(mean_squared_error(Y_test, Y_pred_lm_rfe)))
# 求平均绝对百分比误差
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))*100
print('MAPE = ', mape(Y_test, Y_pred_lm_rfe))


# In[52]:


# 绘制残差图
plt.scatter(Y_train, (Y_train - Y_pred_lm_rfe_train))
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.show()


# In[53]:


test = test.drop(['SalePrice'], axis=1)
# 填补缺失值
test = test.fillna(test.interpolate())
preds = rfe.predict(test)
# 之前做过对数处理，现在恢复
final_predictions = np.exp(preds)


# In[55]:


t = pd.read_csv('D:/house-prices-advanced-regression-techniques/test.csv')
t.index = t.index + 1461
submission = pd.DataFrame({'Id': t.index ,'SalePrice': final_predictions })
submission.to_csv("D:/house-prices-advanced-regression-techniques/submission6.csv",index=False)


# In[ ]:




