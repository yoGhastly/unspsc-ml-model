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
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

try:
    dataset = pd.read_excel('datasets/new/Training.xlsx')
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

print("Columns in the dataset:", dataset.columns)

# Select necessary columns from the dataset
dataset = dataset[['Description', 'Supplier Name', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']]

dataset['Description'] = dataset['Description'].astype(str)
dataset['Supplier Name'] = dataset['Supplier Name'].astype(str)
dataset['UNSPSC Description'] = dataset['UNSPSC Description'].astype(str)

# Combine text fields for feature extraction
dataset['Combined Text'] = dataset['Description'] + ' ' + dataset['Supplier Name'] + ' ' + dataset['UNSPSC Description']

# Clean and preprocess text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(' +', ' ', text)
    
    # Tokenize text
    word_tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_words = [w for w in word_tokens if not w in stop_words]
    
    # Reconstruct the cleaned text
    cleaned_text = " ".join(filtered_words)
    
    return cleaned_text

# Apply preprocessing
dataset['Cleaned Text'] = dataset['Combined Text'].apply(preprocess_text)

# Vectorization for text features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# Encoding categorical features
label_encoder_code = LabelEncoder()
label_encoder_description = LabelEncoder()
dataset['UNSPSC Code Encoded'] = label_encoder_code.fit_transform(dataset['UNSPSC Code'])
dataset['UNSPSC Description Encoded'] = label_encoder_description.fit_transform(dataset['UNSPSC Description'])

# Prepare features and target variables
X = dataset[['Cleaned Text', 'Supplier Name']]
y_code = dataset['UNSPSC Code']
y_description = dataset['UNSPSC Description']

# Define the feature extraction and model pipeline
text_pipeline = Pipeline([
    ('vectorizer', tfidf_vectorizer),
    ('transformer', FunctionTransformer(lambda x: x, validate=False))  # Identity function as placeholder
])

def get_features(df):
    # Vectorize the text features
    text_features = text_pipeline.fit_transform(df['Cleaned Text'])
    # Encode the categorical features
    supplier_encoded = pd.get_dummies(df['Supplier Name'])
    # Ensure the same columns in supplier_encoded
    supplier_encoded = supplier_encoded.reindex(columns=dataset['Supplier Name'].unique(), fill_value=0)
    return np.hstack([text_features.toarray(), supplier_encoded])

X_features = get_features(dataset)

# Split the data
X_train, X_test, y_code_train, y_code_test, y_description_train, y_description_test = train_test_split(
    X_features, y_code, y_description, test_size=0.2, random_state=42
)

# Train Random Forest classifiers for both code and description
model_code = RandomForestClassifier(n_estimators=150, random_state=42)
model_code.fit(X_train, y_code_train)

model_description = RandomForestClassifier(n_estimators=150, random_state=42)
model_description.fit(X_train, y_description_train)

# Make predictions
y_code_pred = model_code.predict(X_test)
y_description_pred = model_description.predict(X_test)

# Evaluate the models
accuracy_code = accuracy_score(y_code_test, y_code_pred)
precision_code = precision_score(y_code_test, y_code_pred, average='weighted', zero_division=1)
recall_code = recall_score(y_code_test, y_code_pred, average='weighted', zero_division=1)
f1_code = f1_score(y_code_test, y_code_pred, average='weighted')

accuracy_description = accuracy_score(y_description_test, y_description_pred)
precision_description = precision_score(y_description_test, y_description_pred, average='weighted', zero_division=1)
recall_description = recall_score(y_description_test, y_description_pred, average='weighted', zero_division=1)
f1_description = f1_score(y_description_test, y_description_pred, average='weighted')

logger.info(f'UNSPSC Code - Accuracy: {accuracy_code}')
logger.info(f'UNSPSC Code - Precision: {precision_code}')
logger.info(f'UNSPSC Code - Recall: {recall_code}')
logger.info(f'UNSPSC Code - F1 Score: {f1_code}')

logger.info(f'UNSPSC Description - Accuracy: {accuracy_description}')
logger.info(f'UNSPSC Description - Precision: {precision_description}')
logger.info(f'UNSPSC Description - Recall: {recall_description}')
logger.info(f'UNSPSC Description - F1 Score: {f1_description}')

# Save the models
joblib.dump(model_code, 'unspsc_code_model.pkl')
joblib.dump(model_description, 'unspsc_description_model.pkl')


def preprocess_user_input(user_input):
    combined_text = preprocess_text(user_input)
    return combined_text

def predict_unspsc(user_input):
    try:
        # Preprocess user input
        preprocessed_input = preprocess_user_input(user_input)
        
        # Vectorize the user input
        vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
        
        # Create a DataFrame with the user input for categorical encoding
        user_df = pd.DataFrame({
            'Cleaned Text': [preprocessed_input],
            'Supplier Name': [user_input]
        })
        
        # Encode categorical features
        supplier_encoded = pd.get_dummies(user_df['Supplier Name'])
        supplier_encoded = supplier_encoded.reindex(columns=dataset['Supplier Name'].unique(), fill_value=0)
        
        # Combine features
        X_input = np.hstack([vectorized_input.toarray(), supplier_encoded])
        
        # Predict UNSPSC Code and Description
        predicted_code = model_code.predict(X_input)[0]
        predicted_description = model_description.predict(X_input)[0]

        # Get Category Code and Description based on the predicted UNSPSC
        category_code = dataset[dataset['UNSPSC Code'] == predicted_code]['Category Code'].values[0]
        category_description = dataset[dataset['UNSPSC Code'] == predicted_code]['Category Description'].values[0]

        return predicted_code, predicted_description, category_code, category_description

    except Exception as e:
        logger.error(f"Error predicting UNSPSC code and description: {e}")
        return None, None, None, None

def main():
    while True:
        try:
            user_input = input("Enter the product description or supplier name (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            unspsc_code, unspsc_description, category_code, category_description = predict_unspsc(user_input)
            
            if unspsc_code:
                print(f"Predicted UNSPSC Code: {unspsc_code}")
                print(f"Predicted UNSPSC Description: {unspsc_description}")
                print(f"Category Code: {category_code}")
                print(f"Category Description: {category_description}")
            else:
                print("UNSPSC Code or Description not found.")
        except Exception as e:
            logger.error(f"Error in main function: {e}")
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
