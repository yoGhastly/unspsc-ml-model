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
from scipy.sparse import hstack
from sklearn.utils import parallel_backend
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')

try:
    dataset = pd.read_excel('datasets/new/Training_reference.xlsx', engine='openpyxl')
    # Columns: Description | Supplier Name
    inputs_df = pd.read_excel('datasets/new/input_descriptions_and_suppliers.xlsm', engine="openpyxl")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Select necessary columns from the dataset
dataset = dataset[['Description', 'Supplier Name', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']]
inputs_df = inputs_df[['Description', 'Supplier Name']]
logger.info(f'Columns in the dataset: {dataset.columns}')

# Check for missing values
missing_values = dataset.isna().sum()
logger.info(f"Checking for NaN values:\n{missing_values}")

# Impute missing UNSPSC Code with a placeholder
dataset['UNSPSC Code'].fillna('Unknown', inplace=True) # pyright: ignore[reportAttributeAccessIssue]

# Convert specific columns to strings
dataset['Description'] = dataset['Description'].astype(str)
dataset['Supplier Name'] = dataset['Supplier Name'].astype(str)
dataset['UNSPSC Description'] = dataset['UNSPSC Description'].astype(str)

# Combine text fields for feature extraction
dataset['Combined Text'] = dataset['Description'] + ' ' + dataset['Supplier Name'] + ' ' + dataset['UNSPSC Description']

# Preprocessing text data with stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    # Convert text to lowercase and remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

# Apply preprocessing
dataset['Cleaned Text'] = dataset['Combined Text'].apply(preprocess_text) # pyright: ignore[reportAttributeAccessIssue]

# Vectorization for text features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)

# Encoding categorical features
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
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
model_code = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)

logger.info("Training UNSPSC Code model...")
with parallel_backend('threading', n_jobs=-1):  # pyright: ignore[reportCallIssue] 
    model_code.fit(X_train, y_code_train)

logger.info("Predicting UNSPSC Code...")
y_code_pred = model_code.predict(X_test)

accuracy_code = accuracy_score(y_code_test, y_code_pred)
precision_code = precision_score(y_code_test, y_code_pred, average='weighted')
recall_code = recall_score(y_code_test, y_code_pred, average='weighted')
f1_code = f1_score(y_code_test, y_code_pred, average='weighted')

logger.info(f"UNSPSC Code Metrics - Accuracy: {accuracy_code:.4f}, Precision: {precision_code:.4f}, Recall: {recall_code:.4f}, F1 Score: {f1_code:.4f}")

model_description = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

logger.info("Training UNSPSC Description model...")
model_description.partial_fit(X_train, y_desc_train, classes=np.unique(y_description))

logger.info("Predicting UNSPSC Description...")
y_desc_pred = model_description.predict(X_test)

accuracy_desc = accuracy_score(y_desc_test, y_desc_pred)
precision_desc = precision_score(y_desc_test, y_desc_pred, average='weighted')
recall_desc = recall_score(y_desc_test, y_desc_pred, average='weighted')
f1_desc = f1_score(y_desc_test, y_desc_pred, average='weighted')

logger.info(f"UNSPSC Description Metrics - Accuracy: {accuracy_desc:.4f}, Precision: {precision_desc:.4f}, Recall: {recall_desc:.4f}, F1 Score: {f1_desc:.4f}")

joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(encoder, 'models/encoder.joblib')
joblib.dump(model_code, 'models/random_forest_code.joblib')
joblib.dump(model_description, 'models/sgd_description.joblib')

def lookup_details_by_description(predicted_description, dataset):
    matched_rows = dataset[dataset['UNSPSC Description'] == predicted_description]
    
    if not matched_rows.empty:
        row = matched_rows.iloc[0]
        category_code = row['Category Code'] if not pd.isna(row['Category Code']) else 'Category Code not found'
        category_description = row['Category Description'] if not pd.isna(row['Category Description']) else 'Category Description not found'
        return row['UNSPSC Code'], category_code, category_description
    else:
        return None, 'Not found', 'Not found'

def preprocess_user_input(user_input: str) -> str:
    combined_text = preprocess_text(user_input)
    return combined_text

def predict_unspsc(user_input: str):
    try:
        # Preprocess user input
        preprocessed_input = preprocess_user_input(user_input)
        
        # Vectorize the user input
        vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
        
        # Combine text features and supplier features
        supplier_name = user_input.split()[-1] 
        supplier_encoded_input = encoder.transform([[supplier_name]])
        user_features = hstack([vectorized_input, supplier_encoded_input])

        predicted_description = model_description.predict(user_features)[0]
        
        predicted_code, category_code, category_description = lookup_details_by_description(predicted_description, dataset)
        
        return predicted_code, predicted_description, category_code, category_description
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, None, None, None

def get_combined_description_supplier(inputs_df: pd.DataFrame | pd.Series) -> list:
    """
    Combine the Description and Supplier Name columns into a single string for each row.
    
    Args:
    inputs_df (pd.DataFrame): DataFrame containing 'Description' and 'Supplier Name' columns.

    Returns:
    list: List of combined strings of '<Description> <Supplier Name>'.
    """
    combined_list = []
    
    # Iterate over each row in the dataframe
    for _index, row in inputs_df.iterrows():
        description = str(row['Description'])
        supplier_name = str(row['Supplier Name'])
        
        # Combine Description and Supplier Name
        combined_text = f"{description} {supplier_name}"
        
        # Add to list
        combined_list.append(combined_text)

    logger.info("Successfully created combined description and supplier list.")
    return combined_list


def main():
    combined_descriptions_suppliers = get_combined_description_supplier(inputs_df)

    results = []
    for user_input in combined_descriptions_suppliers:
        predicted_code, predicted_description, category_code, category_description = predict_unspsc(user_input)
        results.append([user_input, predicted_code, predicted_description, category_code, category_description])

    results_df = pd.DataFrame(results, columns=['Input', 'Predicted UNSPSC Code', 'Predicted UNSPSC Description', 'Category Code', 'Category Description']) # pyright: ignore[reportArgumentType]
    results_df.to_excel('results/predicted_unspsc_results.xlsx', index=False)
    logger.info("Results saved to 'predicted_unspsc_results.xlsx'.")

if __name__ == '__main__':
    main()
