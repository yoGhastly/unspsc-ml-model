import pandas as pd
import numpy as np
import logging
import re
import joblib
import coloredlogs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

try:
    # Efficient data loading using openpyxl engine
    dataset = pd.read_excel('datasets/new/Training_reference.xlsx', engine='openpyxl')
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Select necessary columns from the dataset
dataset = dataset[['Description', 'Supplier Name', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']]

# Convert specific columns to strings
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
encoder = OneHotEncoder(sparse_output=True)
supplier_encoded = encoder.fit_transform(dataset[['Supplier Name']])

# Prepare features and target variables
X = dataset[['Cleaned Text']]
y_code = dataset['UNSPSC Code']
y_description = dataset['UNSPSC Description']

# Vectorize text features
text_features = tfidf_vectorizer.fit_transform(X['Cleaned Text'])

# Combine text features and supplier features
X_features = hstack([text_features, supplier_encoded])

# Split the data
X_train, X_test, y_code_train, y_code_test = train_test_split(X_features, y_code, test_size=0.2, random_state=42)
_, _, y_desc_train, y_desc_test = train_test_split(X_features, y_description, test_size=0.2, random_state=42)

# Use Random Forests for UNSPSC Code prediction
model_code = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)

# Train UNSPSC Code model
logger.info("Training UNSPSC Code model...")
model_code.fit(X_train, y_code_train)

# Predict UNSPSC Code
logger.info("Predicting UNSPSC Code...")
y_code_pred = model_code.predict(X_test)

# Calculate metrics
accuracy_code = accuracy_score(y_code_test, y_code_pred)
precision_code = precision_score(y_code_test, y_code_pred, average='weighted')
recall_code = recall_score(y_code_test, y_code_pred, average='weighted')
f1_code = f1_score(y_code_test, y_code_pred, average='weighted')

logger.info(f"UNSPSC Code Metrics - Accuracy: {accuracy_code:.4f}, Precision: {precision_code:.4f}, Recall: {recall_code:.4f}, F1 Score: {f1_code:.4f}")

# Use Linear Classifier for UNSPSC Description prediction
model_description = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train UNSPSC Description model
logger.info("Training UNSPSC Description model...")
model_description.partial_fit(X_train, y_desc_train, classes=np.unique(y_description))

# Predict UNSPSC Description
logger.info("Predicting UNSPSC Description...")
y_desc_pred = model_description.predict(X_test)

# Calculate metrics
accuracy_desc = accuracy_score(y_desc_test, y_desc_pred)
precision_desc = precision_score(y_desc_test, y_desc_pred, average='weighted')
recall_desc = recall_score(y_desc_test, y_desc_pred, average='weighted')
f1_desc = f1_score(y_desc_test, y_desc_pred, average='weighted')

logger.info(f"UNSPSC Description Metrics - Accuracy: {accuracy_desc:.4f}, Precision: {precision_desc:.4f}, Recall: {recall_desc:.4f}, F1 Score: {f1_desc:.4f}")

# Save models and vectorizer
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(encoder, 'models/encoder.joblib')
joblib.dump(model_code, 'models/random_forest_code.joblib')
joblib.dump(model_description, 'models/sgd_description.joblib')

# Interactive part
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
                print("Exiting the interactive mode.")
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
