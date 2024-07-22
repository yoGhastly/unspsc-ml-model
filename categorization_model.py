import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import re
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
unspsc_large_df = pd.read_excel('unspsc_large_dataset.xlsx')
unspsc_df = pd.read_excel('unspsc_dataset.xlsx')
category_df = pd.read_excel('category_dataset.xlsx')

# Select only necessary columns from unspsc_df
unspsc_df = unspsc_df[['Supplier/Supplying Plant', 'Short Text', 'AMAZON Description', 'AMAZON UNSPSC', 'UNSPSC Description']]

# Merge datasets using 'AMAZON UNSPSC' from unspsc_df and 'UNSPSC' from unspsc_large_df
merged_df = pd.merge(unspsc_df, unspsc_large_df, left_on='AMAZON UNSPSC', right_on='UNSPSC', how='left')

# Remove rows with missing target values
merged_df = merged_df.dropna(subset=['UNSPSC'])

# Clean and preprocess text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(' +', ' ', text)
    # Tokenize words
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into string
    return ' '.join(tokens)

merged_df['Cleaned Short Text'] = merged_df['Short Text'].apply(preprocess_text)
merged_df['Cleaned Amazon Description'] = merged_df['AMAZON Description'].apply(preprocess_text)

# Combine the cleaned text columns for training
merged_df['Combined Text'] = merged_df['Cleaned Short Text'] + ' ' + merged_df['Cleaned Amazon Description']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(merged_df['Combined Text'])

# Define the target variable
y = merged_df['UNSPSC']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # pyright: ignore[reportArgumentType]
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1) # pyright: ignore[reportArgumentType]
f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

logging.info(f'Accuracy: {accuracy}')
logging.info(f'Precision: {precision}')
logging.info(f'Recall: {recall}')
logging.info(f'F1 Score: {f1}')

# Save the model and vectorizer
joblib.dump(model, 'unspsc_predictor_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

def predict_unspsc(description):
    cleaned_description = preprocess_text(description)
    vectorized_description = vectorizer.transform([cleaned_description])
    predicted_code = model.predict(vectorized_description)

    # Retrieve the corresponding UNSPSC description from the reliable dataset
    predicted_description_row = unspsc_large_df[unspsc_large_df['UNSPSC'] == predicted_code[0]]

    if not predicted_description_row.empty and 'UNSPSC Description' in predicted_description_row:
        predicted_description = predicted_description_row['UNSPSC Description'].values[0]
    else:
        # Check in unspsc_dataset if not found
        fallback_row = unspsc_df[unspsc_df['AMAZON UNSPSC'] == predicted_code[0]]
        if not fallback_row.empty and 'UNSPSC Description' in fallback_row:
            predicted_description = fallback_row['UNSPSC Description'].values[0]
        else:
            predicted_description = "Description not found"

    return predicted_code[0], predicted_description

def get_category_from_unspsc(unspsc_code):
    category_row = category_df[category_df['UNSPSC'] == unspsc_code]
    if not category_row.empty:
        category = category_row.iloc[0][['Category Level 1', 'Category Level 2', 'Category Level 3', 'Category Level 4']].values
        return ' > '.join(category)
    
    # Check in unspsc_dataset for category if not found
    fallback_row = unspsc_df[unspsc_df['AMAZON UNSPSC'] == unspsc_code]
    if not fallback_row.empty:
        fallback_code = fallback_row['UNSPSC'].iat[0]
        return get_category_from_unspsc(fallback_code)

    return "Category not found"

# Interactive prediction loop
while True:
    user_input = input("Enter a product description (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_code, predicted_description = predict_unspsc(user_input)
    
    if predicted_code:
        category = get_category_from_unspsc(predicted_code)
        logging.info(f'Description: {user_input}')
        logging.info(f'Predicted UNSPSC Code: {predicted_code}')
        logging.info(f'Predicted Description: {predicted_description}')
        logging.info(f'Category: {category}')
    else:
        logging.info('UNSPSC Code not found for the given description.')
