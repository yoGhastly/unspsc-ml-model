import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import logging
import re
import joblib
import coloredlogs

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load datasets with error handling
try:
    unspsc_large_df = pd.read_excel('datasets/unspsc_large_dataset.xlsx')
    unspsc_df = pd.read_excel('datasets/unspsc_dataset.xlsx')
    category_df = pd.read_excel('datasets/category_dataset.xlsx')
    category_structure_df = pd.read_excel('datasets/category_structure.xlsx')
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

# Select only necessary columns from unspsc_df
unspsc_df = unspsc_df[['Supplier/Supplying Plant', 'Short Text', 'AMAZON Description', 'AMAZON UNSPSC', 'UNSPSC Description']]

# Merge datasets
merged_df = pd.merge(unspsc_df, unspsc_large_df, left_on='AMAZON UNSPSC', right_on='UNSPSC', how='left')
merged_df = merged_df.dropna(subset=['UNSPSC'])

# Clean and preprocess text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(' +', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

merged_df['Cleaned Short Text'] = merged_df['Short Text'].apply(preprocess_text)
merged_df['Cleaned Amazon Description'] = merged_df['AMAZON Description'].apply(preprocess_text)
merged_df['Combined Text'] = merged_df['Cleaned Short Text'] + ' ' + merged_df['Cleaned Amazon Description']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(merged_df['Combined Text'])

# Define the target variable
y = merged_df['UNSPSC']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # pyright: ignore[reportArgumentType]
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1) # pyright: ignore[reportArgumentType]
f1 = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))

logger.info(f'Accuracy: {accuracy}')
logger.info(f'Precision: {precision}')
logger.info(f'Recall: {recall}')
logger.info(f'F1 Score: {f1}')

# Save the model and vectorizer
joblib.dump(model, 'unspsc_predictor_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

def predict_unspsc(description):
    cleaned_description = preprocess_text(description)
    # Search in unspsc_large_df first
    filtered_large_df = unspsc_large_df[unspsc_large_df['UNSPSC Description'].apply(lambda x: cleaned_description in str(x).lower())]
    if not filtered_large_df.empty:
        predicted_code = filtered_large_df.iloc[0]['UNSPSC']
        predicted_description = filtered_large_df.iloc[0]['UNSPSC Description']
        return predicted_code, predicted_description
    # If not found, use the model for prediction
    combined_input = vectorizer.transform([cleaned_description])
    predicted_code = model.predict(combined_input)[0]
    predicted_description = unspsc_large_df.loc[unspsc_large_df['UNSPSC'] == predicted_code, 'UNSPSC Description'].values[0]

    return predicted_code, predicted_description

def get_category_codes(predicted_code):
    category_codes = category_df.loc[category_df['UNSPSC'] == predicted_code, ['Category Level 1 Code ', 'Category Level 2 Code ', 'Category Level 3 Code ', 'Category Level 4 Code']].values
    if category_codes.size > 0:
        return category_codes[0]
    return [None, None, None, None]

def get_category_details_from_category_structure(level_1_code, level_2_code, level_3_code, level_4_code):
    levels = [
        level_4_code,
        level_3_code,
        level_2_code,
        level_1_code
    ]
    
    for level_code in levels:
        if level_code:
            category_code = category_structure_df.loc[category_structure_df['Category Level 1, 2, 3 & 4 Code '] == level_code, ['Category Level 1, 2, 3 & 4 Code ', 'Category Level 1, 2, 3 & 4 Description']]
            if not category_code.empty:
                return {
                    'code': category_code['Category Level 1, 2, 3 & 4 Code '].values[0],
                    'description': category_code['Category Level 1, 2, 3 & 4 Description'].values[0]
                }
    return None

def search_category_by_description(description):
    cleaned_description = preprocess_text(description)
    matched_rows = category_structure_df[category_structure_df['Category Level 1, 2, 3 & 4 Description'].str.contains(cleaned_description, case=False, na=False)]
    if not matched_rows.empty:
        return matched_rows
    return None

def main():
    while True:
        user_input = input("Enter a product description (or type 'q' to quit): ")
        if user_input == 'q':
            break
        try:
            predicted_code, predicted_description = predict_unspsc(user_input)
            level_1_code, level_2_code, level_3_code, level_4_code = get_category_codes(predicted_code)
            category_details = get_category_details_from_category_structure(level_1_code, level_2_code, level_3_code, level_4_code)
            logger.info(f'Predicted UNSPSC Code: {predicted_code}')
            logger.info(f'Predicted Description: {predicted_description}')
            if category_details:
                logger.info(f'Category Code: {category_details["code"]}')
                logger.info(f'Category Description: {category_details["description"]}')
            else:
                logger.warning('We could not find the category based on the predicted UNSPSC code, but here are some options:')
                matched_rows = search_category_by_description(user_input)
                if matched_rows is not None:
                    for _, row in matched_rows.iterrows():
                        logger.info(f'Found Category Code by Description: {row["Category Level 1, 2, 3 & 4 Code "]}')
                        logger.info(f'Found Category Description: {row["Category Level 1, 2, 3 & 4 Description"]}')
                else:
                    logger.warning('No matching category found based on description')
        except Exception as e:
            logger.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
