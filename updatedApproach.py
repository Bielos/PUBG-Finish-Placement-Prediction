# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:58:51 2018

@author: DANIEL MARTINEZ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import EDA_framework as EDA

df = pd.read_csv('train_V2.csv')
# PART 1: PRE-PROCESSING

#Missing Values
EDA.get_missing_data_table(df)
df = EDA.delete_null_observations(df,'winPlacePerc')

#Transform dtypes of features
df['matchType'] = df['matchType'].astype('category')

# BoxPlots
for col in df.columns:
    if df[col].dtype.name != 'object' and df[col].dtype.name != 'category':
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        sn.boxplot(data=df[col], ax= ax)
        ax.set(title=col+' Box Plot')
        plt.show()


# PART 2: Feature Engenieering

# matchType decomposition
standard_modes = ['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp']
isStandard = []
isfpp = []
isSolo = []
isDuo = []
isSquad = []

for match in df['matchType']:
    if match in standard_modes:
        isStandard.append(1)
    else:
        isStandard.append(0)
        
    if 'fpp' in match:
        isfpp.append(1)
    else:
        isfpp.append(0)
    
    if 'solo' in match:
        isSolo.append(1)
    else:
        isSolo.append(0)
    
    if 'duo' in match:
        isDuo.append(1)
    else:
        isDuo.append(0)
    
    if 'Squad' in match:
        isSquad.append(1)
    else:
        isSquad.append(0)
    
df['standardMatch'] = isStandard
df['fppMatch'] = isfpp
df['soloMatch'] = isSolo
df['duoMatch'] = isDuo
df['squadMatch'] = isSquad

df.drop(['matchType'], axis='columns', inplace=True) #delete original matchType

# Adding team features
df_team_dict = (df.groupby('groupId', as_index = True)
          .agg({'Id':'count', 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).to_dict()

teamKills = []
teamSize = []

for teamId in df['groupId']:
    teamKills.append(df_team_dict['teamKills'][teamId])
    teamSize.append(df_team_dict['teamSize'][teamId])

df['teamKills'] = teamKills
df['teamSize'] = teamSize

# Adding match features
df_team = (df.groupby('groupId', as_index = False)
          .agg({'Id':'count', 'matchId':lambda x: x.unique()[0], 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).reset_index()

df_match = (df_team.groupby('matchId', as_index = True)
           .agg({'teamSize':'sum', 'teamKills':'sum'})
           .rename(columns={'teamSize':'matchSize', 'teamKills':'matchKills'})).to_dict()
matchSize = []
matchKills = []

for matchId in df['matchId']:
    matchSize.append(df_match['matchSize'][matchId])
    matchKills.append(df_match['matchKills'][matchId])

df['matchSize'] = matchSize
df['matchKills'] = matchKills

# Delete Outliers according to matchDuration
h_spread = df['matchDuration'].quantile(.75) - df['matchDuration'].quantile(.25)
limit = df['matchDuration'].quantile(.25) - 2 * h_spread
df.drop(df[df['matchDuration'] < limit].index, inplace=True)

# PART 3: FEATURE SELECTION
#Drop insignificant variables
df.drop(['Id'], axis='columns', inplace=True)
df.drop(['groupId'], axis='columns', inplace=True)
df.drop(['matchId'], axis='columns', inplace=True)

# X and y split
y = df['winPlacePerc'].values
X = df.drop(['winPlacePerc'], axis='columns').values

# PART 4: MODELING
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#LightGBM
import lightgbm as lgb

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
        "objective" : "regression",
        "metric" : "mae",
        "n_estimators":15000,
        "early_stopping_rounds":100,
        "num_leaves" : 31, 
        "learning_rate" : 0.05, 
        "bagging_fraction" : 0.9,
        "bagging_seed" : 0, 
        "num_threads" : 4,
        "colsample_bytree" : 0.7
        }

model = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

# PART 5: EVALUATION
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
from sklearn.metrics import mean_absolute_error
print('MAE: {0}'.format(mean_absolute_error(y_test, y_pred)))

# PART 6: FINAL PREDICTION AND SUBMIT
import pandas as pd
import numpy as np

df_test = pd.read_csv('../input/test_V2.csv')
df_test['matchType'] = df_test['matchType'].astype('category')

standard_modes = ['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp']
isStandard_test = []
isfpp_test = []
isSolo_test = []
isDuo_test = []
isSquad_test = []

for match in df_test['matchType']:
    if match in standard_modes:
        isStandard_test.append(1)
    else:
        isStandard_test.append(0)
        
    if 'fpp' in match:
        isfpp_test.append(1)
    else:
        isfpp_test.append(0)
    
    if 'solo' in match:
        isSolo_test.append(1)
    else:
        isSolo_test.append(0)
    
    if 'duo' in match:
        isDuo_test.append(1)
    else:
        isDuo_test.append(0)
    
    if 'Squad' in match:
        isSquad_test.append(1)
    else:
        isSquad_test.append(0)
    
df_test['standardMatch'] = isStandard_test
df_test['fppMatch'] = isfpp_test
df_test['soloMatch'] = isSolo_test
df_test['duoMatch'] = isDuo_test
df_test['squadMatch'] = isSquad_test

df_test.drop(['matchType'], axis='columns', inplace=True) #delete original matchType in test data

df_test_team_dict = (df_test.groupby('groupId', as_index = True)
          .agg({'Id':'count', 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).to_dict()

teamKills_test = []
teamSize_test = []

for teamId in df_test['groupId']:
    teamKills_test.append(df_test_team_dict['teamKills'][teamId])
    teamSize_test.append(df_test_team_dict['teamSize'][teamId])

df_test['teamKills'] = teamKills_test
df_test['teamSize'] = teamSize_test

df_team_test = (df_test.groupby('groupId', as_index = False)
          .agg({'Id':'count', 'matchId':lambda x: x.unique()[0], 'kills':'sum'})
          .rename(columns={'Id':'teamSize', 'kills':'teamKills'})).reset_index()

df_match_test = (df_team_test.groupby('matchId', as_index = True)
           .agg({'teamSize':'sum', 'teamKills':'sum'})
           .rename(columns={'teamSize':'matchSize', 'teamKills':'matchKills'})).to_dict()
matchSize_test = []
matchKills_test = []

for matchId in df_test['matchId']:
    matchSize_test.append(df_match_test['matchSize'][matchId])
    matchKills_test.append(df_match_test['matchKills'][matchId])

df_test['matchSize'] = matchSize_test
df_test['matchKills'] = matchKills_test

X_testdata = df_test.drop(['Id','groupId','matchId'], axis='columns').values

df_test['winPlacePerc'] = model.predict(X_testdata, num_iteration=model.best_iteration)
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)