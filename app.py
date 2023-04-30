import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("players.csv")
scaler = StandardScaler()  # Instantiate the scaler

#dropping any player who's starting position is sub
df = df[df['position'] != 'Sub']

# Filter the data to get the last 10 games for each player
last_10_games = df.groupby("player_name").tail(10)

# Select the features for X
X = last_10_games[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]]

# Scale the features using StandardScaler
scaler.fit(X)  # Fit the scaler with the training data

def load_model():
    model_path = 'model_1/model_1.h5'
    model = tf.keras.models.load_model(model_path)
    return tf.keras.models.load_model(model_path)

model = load_model()  # Load the model

def predict_goals(player_name, model, scaler, df):
    player_data = df[df['player_name'] == player_name].tail(10)  # select last 10 games for the player
    X = scaler.transform(player_data[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]].values)
    pred = model.predict(X)
    predicted_goals = round(pred[0][0], 2)
    
    return predicted_goals

def calculate_points(predicted_goals):
    if 2.51 <= predicted_goals <= 3.5:
        return 17
    elif 1.8 <= predicted_goals <= 2.5:
        return 12
    elif 0.85 <= predicted_goals <= 1.79:
        return 7
    else:
        return 2

def main():
    st.title("Fantasy Football Premier League Player Predictor")
    
    st.header("How it works:")
    
    container = st.container()
    with container:
        st.text("Any Premier League football player's name can be typed into the search bar, and once done,\n the system will start predicting how many goals they are likely to score in their next match.\n This will involve predicting the players' goals using a variety of football metrics that have been gathered from their previous 10 games,\n taking into account the players' current form. The scoring algorithm will be used to determine the player's predicted number of \n fantasy football points after the model has finished predicting the player's predicted goals.")
    container.expander("Read more")
     
    player_name = st.text_input("Enter the player name:")

    if st.button("Predict"):
        predicted_goals = predict_goals(player_name, model, scaler, df)
        st.write(f"Predicted goals for {player_name}: {predicted_goals}")
        points = calculate_points(predicted_goals)
        st.write(f"Predicted points: {points}")
    
    if st.button("Top 30 Predicted Players"):
        preds_list = []
        for player_id in df['player_name'].unique():
            player_data = df[df['player_name'] == player_id].tail(10)  # select last 10 games for each player
            X = scaler.transform(player_data[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]].values)
            pred = model.predict(X)
            predicted_goals = round(pred[0][0], 2)
            points = calculate_points(predicted_goals)
            preds_list.append({'player_name': player_id, 'predicted_goals': predicted_goals, 'points': points})
        
        
        #VISUALISE TOP 30 PLAYERS
        top_predicted = pd.DataFrame(preds_list).sort_values(by='predicted_goals', ascending=False).head(30)
        st.subheader("Top 30 Players with Highest Predicted Goals")
        st.dataframe(top_predicted)
        
        # Scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(top_predicted['player_name'], top_predicted['predicted_goals'])
        plt.xlabel("Player")
        plt.ylabel("Predicted Goals")
        plt.title("Predicted Goals for Top 30 Players")
        plt.xticks(rotation=45)
        st.pyplot(plt)
    
    #TOP 10 WORST PLAYERS    
    if st.button("Worst 10 Players to Pick"):
        preds_list = []
        for player_id in df['player_name'].unique():
            player_data = df[df['player_name'] == player_id].tail(10)  # select last 10 games for each player
            X = scaler.transform(player_data[["goals", "shots", "xG", "time", "h_goals", "a_goals", "xA", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]].values)
            pred = model.predict(X)
            predicted_goals = round(pred[0][0], 2)
            points = calculate_points(predicted_goals)
            preds_list.append({'player_name': player_id, 'predicted_goals': predicted_goals, 'points': points})
        
        worst_predicted = pd.DataFrame(preds_list).sort_values(by='predicted_goals').head(10)
        st.subheader("Worst 10 Players to Pick")
        st.dataframe(worst_predicted)

if __name__ == '__main__':
    main()