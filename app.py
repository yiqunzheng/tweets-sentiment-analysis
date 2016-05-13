import pandas as pd
import load_model

new_tweets = pd.read_pickle("tweetdata.pkl")['text']
model = load_model()
predictions = model.predict(new_tweets)