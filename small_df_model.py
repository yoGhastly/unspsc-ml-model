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
import os

# Initialize logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

try:
    # Load datasets
    dataset = pd.read_excel('datasets/new/Training_a.xlsx', engine='openpyxl')
    inputs_df = pd.read_excel('datasets/new/inputs_b.xlsx', engine="openpyxl")
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
dataset['UNSPSC Code'].fillna('Unknown', inplace=True)

# Convert specific columns to strings
dataset['Description'] = dataset['Description'].astype(str)
dataset['Supplier Name'] = dataset['Supplier Name'].astype(str)
dataset['UNSPSC Description'] = dataset['UNSPSC Description'].astype(str)

# Combine text fields for feature extraction
dataset['Combined Text'] = dataset['Description'] + ' ' + dataset['Supplier Name'] + ' ' + dataset['UNSPSC Description']

# Preprocessing text data with stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Preprocesses text by converting to lowercase, removing punctuation, tokenizing, and removing stopwords.
    
    Args:
    text (str): The text to preprocess.

    Returns:
    str: The cleaned and preprocessed text.
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

# Apply preprocessing
dataset['Cleaned Text'] = dataset['Combined Text'].apply(preprocess_text)

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
with parallel_backend('threading', n_jobs=-1): 
    model_code.fit(X_train, y_code_train)

logger.info("Predicting UNSPSC Code...")
y_code_pred = model_code.predict(X_test)

accuracy_code = accuracy_score(y_code_test, y_code_pred)
precision_code = precision_score(y_code_test, y_code_pred, average='weighted')
recall_code = recall_score(y_code_test, y_code_pred, average='weighted')
f1_code = f1_score(y_code_test, y_code_pred, average='weighted')

logger.info(f"UNSPSC Code Metrics - Accuracy: {accuracy_code:.4f}, Precision: {precision_code:.4f}, Recall: {recall_code:.4f}, F1 Score: {f1_code:.4f}")

# Use Stochastic Gradient Descent for UNSPSC Description prediction
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

# Save models and vectorizers
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.joblib')
joblib.dump(encoder, 'models/encoder.joblib')
joblib.dump(model_code, 'models/random_forest_code.joblib')
joblib.dump(model_description, 'models/sgd_description.joblib')

def lookup_details_by_description(predicted_description, dataset):
    """
    Lookup UNSPSC code and category details based on the predicted UNSPSC description.

    Args:
    predicted_description (str): The predicted UNSPSC description.
    dataset (pd.DataFrame): The dataset to search for the matching description.

    Returns:
    tuple: A tuple containing UNSPSC code, category code, and category description.
    """
    matched_rows = dataset[dataset['UNSPSC Description'] == predicted_description]
    
    if not matched_rows.empty:
        row = matched_rows.iloc[0]
        category_code = row['Category Code'] if not pd.isna(row['Category Code']) else 'Category Code not found'
        category_description = row['Category Description'] if not pd.isna(row['Category Description']) else 'Category Description not found'
        return row['UNSPSC Code'], category_code, category_description
    else:
        return None, 'Not found', 'Not found'

def preprocess_user_input(user_input: str) -> str:
    """
    Preprocess the user's input by combining and cleaning the text.

    Args:
    user_input (str): The user's input string.

    Returns:
    str: The preprocessed input text.
    """
    combined_text = preprocess_text(user_input)
    return combined_text

def load_feedback_file():
    """
    Load feedback file if it exists. Otherwise, return an empty DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame containing feedback if available.
    """
    feedback_file_path = 'outputs/feedback.csv'
    if os.path.exists(feedback_file_path):
        return pd.read_csv(feedback_file_path)
    else:
        return pd.DataFrame(columns=['Description', 'Supplier Name', 'Correct UNSPSC Code', 'Correct UNSPSC Description', 'Correct Category Code', 'Correct Category Description'])


def predict_unspsc(user_input: str, feedback_df: pd.DataFrame):
    """
    Predict UNSPSC code, description, and category details for a given user input.
    First checks feedback file for corrections.

    Args:
    user_input (str): The user's input string.
    feedback_df (pd.DataFrame): DataFrame containing feedback corrections.

    Returns:
    tuple: A tuple containing predicted UNSPSC code, UNSPSC description, category code, and category description.
    """
    try:
        # Preprocess user input
        preprocessed_input = preprocess_user_input(user_input)
        
        # Vectorize the user input
        vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
        
        # Combine text features and supplier features
        supplier_name = user_input.split()[-1] 
        supplier_encoded_input = encoder.transform([[supplier_name]])
        user_features = hstack([vectorized_input, supplier_encoded_input])

        # Predict UNSPSC description
        predicted_description = model_description.predict(user_features)[0]

        # Check if the prediction exists in the feedback file
        feedback_match = feedback_df[(feedback_df['Description'] + ' ' + feedback_df['Supplier Name']) == user_input]
        
        if not feedback_match.empty:
            feedback_row = feedback_match.iloc[0]
            return (feedback_row['Correct UNSPSC Code'],
                    feedback_row['Correct UNSPSC Description'],
                    feedback_row['Correct Category Code'],
                    feedback_row['Correct Category Description'])
        else:
            # Lookup UNSPSC code and category details
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
    list: A list of combined description and supplier name strings.
    """
    return (inputs_df['Description'] + ' ' + inputs_df['Supplier Name']).tolist()

def main_loop():
    """
    Main loop for continuously running the prediction script and gathering feedback.
    """
    input_list = get_combined_description_supplier(inputs_df)
    feedback_df = load_feedback_file()

    for idx, input_text in enumerate(input_list):
        # Print the original user input
        logger.info(f"\nOriginal Input ({idx+1}): {input_text}")
        
        # Predict and display results
        predicted_code, predicted_description, category_code, category_description = predict_unspsc(input_text, feedback_df)
        
        # Append predictions to the DataFrame
        inputs_df.loc[idx, 'Predicted UNSPSC Code'] = predicted_code
        inputs_df.loc[idx, 'Predicted UNSPSC Description'] = predicted_description
        inputs_df.loc[idx, 'Category Code'] = category_code
        inputs_df.loc[idx, 'Category Description'] = category_description

    # Save predictions to a file
    inputs_df.to_csv('outputs/predictions.csv', index=False)
    logger.info(f"Predictions saved to 'outputs/predictions.csv'.")

    # Collect user feedback
    feedback_needed = True
    while feedback_needed:
        feedback_response = input("\nDo you want to provide feedback on any prediction? (yes/no): ").strip().lower()

        if feedback_response == 'yes':
            # Collect feedback
            feedback_index = int(input(f"Enter the index (1 to {len(input_list)}) of the prediction you want to provide feedback on: ").strip()) - 1
            
            if 0 <= feedback_index < len(input_list):
                # Display current predictions for feedback
                current_prediction = inputs_df.loc[feedback_index]
                logger.info(f"Current Prediction for entry {feedback_index+1}:")
                logger.info(f"  Description: {current_prediction['Description']}")
                logger.info(f"  Supplier Name: {current_prediction['Supplier Name']}")
                logger.info(f"  Predicted UNSPSC Code: {current_prediction['Predicted UNSPSC Code']}")
                logger.info(f"  Predicted UNSPSC Description: {current_prediction['Predicted UNSPSC Description']}")
                logger.info(f"  Category Code: {current_prediction['Category Code']}")
                logger.info(f"  Category Description: {current_prediction['Category Description']}")

                feedback_description = input(f"Enter the correct UNSPSC Description for entry {feedback_index+1}: ").strip()
                feedback_code = input(f"Enter the correct UNSPSC Code for entry {feedback_index+1}: ").strip()
                feedback_category_code = input(f"Enter the correct Category Code for entry {feedback_index+1}: ").strip()
                feedback_category_description = input(f"Enter the correct Category Description for entry {feedback_index+1}: ").strip()
                
                # Update with feedback
                inputs_df.loc[feedback_index, 'Correct UNSPSC Code'] = feedback_code
                inputs_df.loc[feedback_index, 'Correct UNSPSC Description'] = feedback_description
                inputs_df.loc[feedback_index, 'Correct Category Code'] = feedback_category_code
                inputs_df.loc[feedback_index, 'Correct Category Description'] = feedback_category_description

                # Save feedback to a separate file
                feedback_df = inputs_df[['Description', 'Supplier Name', 'Correct UNSPSC Code', 'Correct UNSPSC Description', 'Correct Category Code', 'Correct Category Description']]
                feedback_file_path = 'outputs/feedback.csv'
                feedback_df.dropna().to_csv(feedback_file_path, index=False)
                logger.info(f"Feedback saved to '{feedback_file_path}'.")
            else:
                logger.warning(f"Invalid index: {feedback_index+1}. Please try again.")
        elif feedback_response == 'no':
            feedback_needed = False
            logger.info("Exiting feedback loop.")
        else:
            logger.warning("Invalid response. Please enter 'yes' or 'no'.")

if __name__ == "__main__":
    """
    Main function that runs the prediction and feedback loop.
    """
    continue_running = True
    
    while continue_running:
        logger.info("\nStarting new prediction cycle...")
        main_loop()

        continue_response = input("\nDo you want to continue running the script? (yes/no): ").strip().lower()

        if continue_response == 'no':
            continue_running = False
            logger.info("Exiting the script. Thank you!")
        elif continue_response != 'yes':
            logger.warning("Invalid response. Please enter 'yes' or 'no'.")

    logger.info("Script terminated successfully.")
