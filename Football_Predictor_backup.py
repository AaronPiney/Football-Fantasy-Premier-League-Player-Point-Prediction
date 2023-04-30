#!/usr/bin/env python
# coding: utf-8

# Welcome to my fantasy football player prediction system. In this code i will use machine learning to predict how many points a player should get in an upcoming gameweek. I will use a classfier which will produce high accruacy results based on the data which has been trained by a user. 

# Firslty we need to load the neccessary packages.

# In[9]:


import os
import csv 
import pandas as pd
import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[10]:


directory_path = 'PremierLeagueData/2022-23/understat/'

csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

dataframes = []

for file in csv_files:
    if 'understat_' not in file:
        df = pd.read_csv(file)
        player_name = os.path.basename(file).split('.')[0].replace('_',' ')
        position = df['position']
        
        h_team = df['h_team']
        a_team = df['a_team']
            
        df['player_name'] = player_name
        df['h_team'] = h_team
        df['a_team'] = a_team
        df['position'] = position
        dataframes.append(df)
        
players = pd.concat(dataframes, ignore_index=True)

players = players.groupby(['id', 'player_name', 'position', 'h_team', 'a_team'], as_index=False).agg('sum')

players.to_csv('players.csv', index=False)

print(players)


# I will now drop any player that started off as a sub because this will affect the accuracy score and if a player who normally starts but is rested for one week and is started as sub then this won't give an accurate representation.

# Data Preprocessing & Feature Engieneering
# 
# Data Preprocessing - This part will involve me cleaning, transforming and selecting the data which will be relevent for me. 
# 
# Feature Engineering- Creating any other bits of data to improve my prediction

# Firstly i'm going to separate the goalkeeper, these will need different features to predict the points off as the main focus for there points will be around keeping clean sheets. If a goal keeper keeps a clean sheet they acheive 5 points. 

# After separeting the goal keepers i'm now going to separate the defenders. I will be introducing goals and assists into this as some defenders reguarly get assists as they are very attacking minded.

# Now i'm separating the midfielders, this will included the same features as the defenders because if a midfielder keeps a clean sheet they earn an extra point. 

# Finally i'm seperating the forwards which won't include goals condeded as if a forwards team concedes they don't lose any points. The main feature for these will be goals scored and assists becuase that will be where the majority of their points come from. 

# Model Selection- Selecting my ML algorithm which is most appropriate and achieves the highest accuracy. 

# In[11]:


df = pd.read_csv("players.csv")

#dropping any player who's starting position is sub
df = df[df['position'] != 'Sub']

# Filter the data to get the last 10 games for each player
last_10_games = df.groupby("player_name").tail(10)

# Select the features for X
X = last_10_games[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]]

# Create the y matrix for goals only
y = last_10_games[["goals"]]


# In[12]:


# Encode the position column using LabelEncoder
le = LabelEncoder()
last_10_games["position_encoded"] = le.fit_transform(last_10_games["position"])


# In[13]:


# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X.values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)


# In[14]:


# Create an empty list to store predicted values for each player
preds_list = []

for player_id in df['player_name'].unique():
    player_data = df[df['player_name'] == player_id].tail(10)  # select last 10 games for each player
    X = scaler.transform(player_data[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]].values)
    
    y = player_data['goals']
    pred = regressor.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    # Append the predicted value to the preds_list
    preds_list.append({'player_name': player_id, 'predicted_goals': round(pred[0], 2)})

top_predicted = pd.DataFrame(preds_list).sort_values(by='predicted_goals', ascending=False)
print(top_predicted.head(10))


# In[15]:


# Create a dataframe from the preds_list
preds = pd.DataFrame(preds_list)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("Mean absolute error: ", mae)
print("R-squared value: ", r2)


# In[16]:


#GRID SEARCH

# param_grid = {
#     'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50), (100,100)],
#     'activation': ['logistic', 'tanh', 'relu'],
#     'solver': ['adam', 'sgd'],
#     'learning_rate': ['constant', 'adaptive'],
#     'alpha': [0.0001, 0.001, 0.01, 0.1],
# }


# # # Perform grid search
# grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train_scaled, y_train)

# # # Get the best parameters
# best_params = grid_search.best_params_
# print("Best parameters:", best_params)


# NEW MODEL USING MY OWN MLP CLASSIFIER

# In[17]:


df = pd.read_csv("players.csv")

#dropping any player who's starting position is sub
df = df[df['position'] != 'Sub']

# Filter the data to get the last 10 games for each player
last_10_games = df.groupby("player_name").tail(10)

# Select the features for X
X = last_10_games[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]]

# Create the y matrix for goals only
y = last_10_games[["goals"]]


# In[18]:


# Encode the position column using LabelEncoder
le = LabelEncoder()
last_10_games["position_encoded"] = le.fit_transform(last_10_games["position"])


# In[19]:


# Scale the features using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X.values)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the architecture of your MLP
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on your training data
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[20]:


# Create an empty list to store predicted values for each player
preds_list = []

for player_id in df['player_name'].unique():
    player_data = df[df['player_name'] == player_id].tail(10)  # select last 10 games for each player
    X = scaler.transform(player_data[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]].values)
    
    y = player_data['goals']
    pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    # Append the predicted value to the preds_list
    preds_list.append({'player_name': player_id, 'predicted_goals': round(pred[0][0], 2)})

top_predicted = pd.DataFrame(preds_list).sort_values(by='predicted_goals', ascending=False)
print(top_predicted.head(10))


# In[21]:


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("Mean absolute error: ", mae)
print("R-squared value: ", r2)


# Model Training- Train the model selected using different type of parameters which can improve it's performance.

# Model Evaluation- Evaluate the performance of the model by testing.  
