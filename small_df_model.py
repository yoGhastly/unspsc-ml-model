from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import logging
import re
import joblib
import coloredlogs
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Set up logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load models and vectorizers
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
encoder = joblib.load('models/encoder.joblib')
model_code = joblib.load('models/random_forest_code.joblib')
model_description = joblib.load('models/sgd_description.joblib')

# Load dataset
dataset = pd.read_excel('datasets/new/Training_a.xlsx', engine='openpyxl')
dataset = dataset[['Description', 'Supplier Name', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']]

# Preprocess the dataset
dataset['Description'] = dataset['Description'].astype(str)
dataset['Supplier Name'] = dataset['Supplier Name'].astype(str)
dataset['UNSPSC Description'] = dataset['UNSPSC Description'].astype(str)
dataset['Combined Text'] = dataset['Description'] + ' ' + dataset['Supplier Name'] + ' ' + dataset['UNSPSC Description']

# Preprocessing text data with stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

# Apply preprocessing
dataset['Cleaned Text'] = dataset['Combined Text'].apply(preprocess_text) # pyright: ignore[reportAttributeAccessIssue]

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

def load_feedback_file():
    feedback_file_path = 'outputs/feedback.csv'
    if os.path.exists(feedback_file_path):
        return pd.read_csv(feedback_file_path)
    else:
        return pd.DataFrame(columns=['Description', 'Supplier Name', 'Correct UNSPSC Code', 'Correct UNSPSC Description', 'Correct Category Code', 'Correct Category Description']) # pyright: ignore[reportArgumentType]

def validate_file(inputs_df: pd.DataFrame):
    required_columns = {'Description', 'Supplier Name'}
    if not required_columns.issubset(inputs_df.columns):
        raise ValueError(f"Input file must contain columns: {required_columns}")

def predict_unspsc(user_input: str, feedback_df: pd.DataFrame):
    try:
        preprocessed_input = preprocess_user_input(user_input)
        vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
        supplier_name = user_input.split()[-1]
        supplier_encoded_input = encoder.transform([[supplier_name]])
        user_features = hstack([vectorized_input, supplier_encoded_input])
        predicted_description = model_description.predict(user_features)[0]
        feedback_match = feedback_df[(feedback_df['Description'] + ' ' + feedback_df['Supplier Name']) == user_input]
        
        if not feedback_match.empty:
            feedback_row = feedback_match.iloc[0]
            return (user_input, feedback_row['Correct UNSPSC Code'], feedback_row['Correct UNSPSC Description'], feedback_row['Correct Category Code'], feedback_row['Correct Category Description'])
        else:
            predicted_code, category_code, category_description = lookup_details_by_description(predicted_description, dataset)
            return (user_input, predicted_code, predicted_description, category_code, category_description)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return user_input, None, None, None, None

@app.route('/predict', methods=['POST'])
def predict():
    feedback_df = load_feedback_file()

    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"error": "No input file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        logger.error("No file selected")
        return jsonify({"error": "No file selected"}), 400

    if not file:
        logger.error("No file found")
        return jsonify({"error": "No file found"}), 400

    try:
        # Ensure the file is an Excel file
        if not file.filename.endswith('.xlsx'): # pyright: ignore[reportOptionalMemberAccess]
            raise ValueError("Invalid file format. Only .xlsx files are supported.")
        
        # Process the file
        inputs_df = pd.read_excel(file, engine="openpyxl")
        validate_file(inputs_df)
    except Exception as e:
        logger.error(f"File processing error: {e}")
        return jsonify({"error": str(e)}), 400

    predictions = []
    combined_texts = get_combined_description_supplier(inputs_df)
    for user_input in combined_texts:
        prediction = predict_unspsc(user_input, feedback_df)
        predictions.append(prediction)
    
    # Convert predictions to DataFrame and save to output file
    output_df = pd.DataFrame(predictions, columns=['Original Input', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']) # pyright: ignore[reportArgumentType]
    output_path = 'outputs/predictions.xlsx'
    output_df.to_excel(output_path, index=False)

    # Send the file as a response
    return send_file(output_path, as_attachment=True, download_name='predictions.xlsx')

def get_combined_description_supplier(inputs_df: pd.DataFrame) -> list:
    return (inputs_df['Description'] + ' ' + inputs_df['Supplier Name']).tolist()

if __name__ == '__main__':
    app.run(debug=True)
