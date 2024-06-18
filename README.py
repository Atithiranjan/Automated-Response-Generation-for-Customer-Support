# Automated-Response-Generation-for-Customer-Support

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Step 1: Data Exploration and Preprocessing

# Load the dataset
df = pd.read_csv('customer_queries.csv')

# Display the first few rows of the dataset
print("Sample data:")
print(df.head())

# Check for any missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data Cleaning and Preprocessing
def clean_text(text):
    # Lowercase text
    text = text.lower()
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning function to both 'query' and 'response' columns
df['cleaned_query'] = df['query'].apply(clean_text)
df['cleaned_response'] = df['response'].apply(clean_text)

# Drop rows with empty cleaned_query or cleaned_response
df.dropna(subset=['cleaned_query', 'cleaned_response'], inplace=True)

# Tokenization (if needed)
# Example using simple whitespace tokenization
df['tokenized_query'] = df['cleaned_query'].apply(lambda x: x.split())
df['tokenized_response'] = df['cleaned_response'].apply(lambda x: x.split())

# Step 2: Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("\nTraining data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# Save preprocessed data (optional)
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# Further steps involve model selection, training, and evaluation, which can vary based on the chosen approach (Seq2Seq, transformer-based model like GPT-3).

# Example model training :
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example LSTM-based Seq2Seq model 
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, num_features)))  # Define your input shape
model.add(Dense(vocab_size, activation='softmax'))  # Adjust output based on vocabulary size

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=10, batch_size=32, validation_data=(val_X, val_Y))

# Evaluation example (basic example, not optimized)
loss, accuracy = model.evaluate(test_X, test_Y)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
