import pandas as pd
import numpy as np
import logging
import re
import joblib
import coloredlogs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import process
from rake_nltk import Rake, Metric


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
    unspsc_with_levels_df = pd.read_excel('datasets/unspsc_with_levels.xlsx')
except Exception as e:
    logger.error(f"Error loading datasets: {e}")
    raise

# Select only necessary columns from unspsc_df
unspsc_df = unspsc_df[['Supplier/Supplying Plant', 'Short Text', 'AMAZON Description', 'AMAZON UNSPSC', 'UNSPSC Description']]

# Merge datasets
merged_df = pd.merge(unspsc_df, unspsc_large_df, left_on='AMAZON UNSPSC', right_on='UNSPSC', how='left')
merged_df = merged_df.dropna(subset=['UNSPSC'])

# Combine 'AMAZON Description' and 'UNSPSC Description_x' columns for text processing
merged_df['Combined Text'] = merged_df['AMAZON Description'].fillna('') + ' ' + merged_df['UNSPSC Description_x'].fillna('')

# Convert all values in 'Combined Text' to string
merged_df['Combined Text'] = merged_df['Combined Text'].astype(str)

# Merge with `unspsc_with_levels` to add descriptions and definitions
merged_df = pd.merge(merged_df, unspsc_with_levels_df, left_on='UNSPSC', right_on='UNSPSC with N', how='left')
merged_df['Combined Text'] = merged_df['Combined Text'] + ' ' + merged_df['UNSPSC Definition'].fillna('')

# Clean and preprocess text data
stop_words = set(stopwords.words('english'))
# punctuations Optional[Set[str]] = None
punctuations = {'.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '<', '>', '@', '#', '$', '%', '^', '&', '*', '~', '`', '+'}
rnk = Rake(stopwords=stop_words, max_length=1, min_length=1, ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO, punctuations=punctuations)

def preprocess_text(text: str) -> str:
    """
    Preprocesses text data by removing special characters, extra spaces, and extracting keywords.
        - If sentence has more than 3 words, extract keywords using Rake.
        - Returns the first keyword if found, otherwise returns the original text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(' +', ' ', text)
    words = text.split()
    # HACK: This helps for large descriptions with multiple keywords to extract the most relevant one
    if len(words) > 3:
        rnk.extract_keywords_from_text(text)
        keyword_extracted = rnk.get_ranked_phrases_with_scores() # [(1.0, 'desktop')]
        text = keyword_extracted[0][1] if keyword_extracted else text
    return text

# Use TF-IDF Vectorizer instead of Word2Vec
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(merged_df['Combined Text'])

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

# Save the model
joblib.dump(model, 'unspsc_predictor_model.pkl')

def get_category_codes(predicted_code):
    try:
        category_codes = category_df.loc[category_df['UNSPSC'] == predicted_code, [
            'Category Level 1 Code ', 'Category Level 2 Code ', 'Category Level 3 Code ', 'Category Level 4 Code']].values
        if category_codes.size > 0:
            return category_codes[0]
        return [None, None, None, None]
    except Exception as e:
        logger.error(f"Error retrieving category codes: {e}")
        return [None, None, None, None]

def get_category_details_from_category_structure(level_1_code, level_2_code, level_3_code, level_4_code):
    try:
        levels = [
            level_4_code,
            level_3_code,
            level_2_code,
            level_1_code
        ]
        
        for level_code in levels:
            if level_code:
                category_code = category_structure_df.loc[
                    category_structure_df['Category Level 1, 2, 3 & 4 Code '] == level_code,
                    ['Category Level 1, 2, 3 & 4 Code ', 'Category Level 1, 2, 3 & 4 Description']]
                if not category_code.empty:
                    return {
                        'code': category_code['Category Level 1, 2, 3 & 4 Code '].values[0],
                        'description': category_code['Category Level 1, 2, 3 & 4 Description'].values[0]
                    }
        return None
    except Exception as e:
        logger.error(f"Error retrieving category details: {e}")
        return None

def search_category_by_description(description):
    try:
        cleaned_description = preprocess_text(description)
        matched_rows = category_structure_df[
            category_structure_df['Category Level 1, 2, 3 & 4 Description'].str.contains(
                cleaned_description, case=False, na=False
            )
        ]
        if not matched_rows.empty:
            return matched_rows
        return None
    except Exception as e:
        logger.error(f"Error searching category by description: {e}")
        return None

def predict_unspsc(description):
    try:
        # Preprocess the user description
        cleaned_description = preprocess_text(description)

        # Step 1: Check for exact matches in unspsc_large_df
        exact_match = unspsc_large_df[
            unspsc_large_df['UNSPSC Description'].str.contains(cleaned_description, case=False, na=False)
        ]
        if not exact_match.empty:
            unspsc_code = exact_match.iloc[0]['UNSPSC']
            unspsc_description = exact_match.iloc[0]['UNSPSC Description']
        else:
            # Step 2: Use model prediction
            vectorized_description = tfidf_vectorizer.transform([cleaned_description])
            unspsc_code = model.predict(vectorized_description)[0]
            unspsc_description = unspsc_large_df.loc[unspsc_large_df['UNSPSC'] == unspsc_code, 'UNSPSC Description'].values
            
            if len(unspsc_description) > 0:
                unspsc_description = unspsc_description[0]
            else:
                unspsc_description = "Description not found"

            # Step 3: Fuzzy Matching
            possible_descriptions = unspsc_large_df['UNSPSC Description'].values
            best_match_tuple = process.extractOne(description, possible_descriptions)

            threshold = 90
            if best_match_tuple:
                best_match, match_score = best_match_tuple # pyright: ignore[reportAssignmentType]
                if match_score > threshold:
                    unspsc_code = unspsc_large_df.loc[unspsc_large_df['UNSPSC Description'] == best_match, 'UNSPSC'].values[0]
                    unspsc_description = best_match

        # Retrieve category information
        level_1_code, level_2_code, level_3_code, level_4_code = get_category_codes(unspsc_code)
        category_details = get_category_details_from_category_structure(level_1_code, level_2_code, level_3_code, level_4_code)
        
        if not category_details:
            # Search by description if category is not found by code
            category_matches = search_category_by_description(cleaned_description)
            if category_matches is not None and not category_matches.empty:
                category_code = category_matches.iloc[0]['Category Level 1, 2, 3 & 4 Code ']
                category_description = category_matches.iloc[0]['Category Level 1, 2, 3 & 4 Description']
                category_details = {
                    'code': category_code,
                    'description': category_description
                }

        return {
            'UNSPSC Code': unspsc_code,
            'UNSPSC Description': unspsc_description,
            'Category Code': category_details['code'] if category_details else None,
            'Category Description': category_details['description'] if category_details else None
        }
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            'UNSPSC Code': None,
            'UNSPSC Description': None,
            'Category Code': None,
            'Category Description': None
        }

def main():
    while True:
        users_description = input("Enter a product description (or type 'q' to quit): ")
        if users_description == 'q':
            break
        prediction = predict_unspsc(users_description)
        print(f'UNSPSC Code: {prediction["UNSPSC Code"]}')
        print(f'UNSPSC Description: {prediction["UNSPSC Description"]}')
        print(f'Category Code: {prediction["Category Code"]}')
        print(f'Category Description: {prediction["Category Description"]}')

if __name__ == '__main__':
    main()
