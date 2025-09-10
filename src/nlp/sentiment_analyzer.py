from transformers import pipeline
import pandas as pd

# Load data
df = pd.read_csv("data/processed/clean_reviews.csv")

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis")

# Apply sentiment analysis
df["sentiment"] = df["review_text"].apply(lambda x: sentiment_model(x)[0]['label'])

# Save
df.to_csv("data/processed/reviews_with_sentiment.csv", index=False)
print(df[["review_text", "sentiment"]])