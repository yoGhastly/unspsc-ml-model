from flask import Flask, request, jsonify
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
        return pd.DataFrame(columns=['Description', 'Correct UNSPSC Code', 'Correct UNSPSC Description', 'Correct Category Code', 'Correct Category Description']) # pyright: ignore[reportArgumentType]

def validate_file(inputs_df: pd.DataFrame):
    required_columns = {'Description', 'Supplier Name'}
    if not required_columns.issubset(inputs_df.columns):
        raise ValueError(f"Input file must contain columns: {required_columns}")

    for column in required_columns:
        if not inputs_df[column].apply(isinstance, args=(str,)).all():
            raise ValueError(f"All values in column '{column}' must be strings")

def predict_unspsc(user_input: str, feedback_df: pd.DataFrame):
    try:
        user_input = str(user_input)
        preprocessed_input = preprocess_user_input(user_input)
        vectorized_input = tfidf_vectorizer.transform([preprocessed_input])
        supplier_name = user_input.split()[-1]
        supplier_encoded_input = encoder.transform([[supplier_name]])
        user_features = hstack([vectorized_input, supplier_encoded_input])
        predicted_description = model_description.predict(user_features)[0]
        feedback_match = feedback_df[(feedback_df['Description']) == user_input]
        
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
        if not file.filename.endswith('.xlsx'): # pyright: ignore[reportOptionalMemberAccess]
            raise ValueError("Invalid file format. Only .xlsx files are supported.")
        
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
    
    output_df = pd.DataFrame(predictions, columns=['Original Input', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description']) # pyright: ignore[reportArgumentType]
    result = output_df.to_dict(orient='records')
    
    return jsonify(result)


@app.route('/predict-file', methods=['GET'])
def predict_file():
    try:
        # Specify the file path for the input file
        input_file_path = 'PO Spend (Jun and Jul).xlsx'
        
        # Check if the input file exists
        if not os.path.exists(input_file_path):
            logger.error(f"Input file {input_file_path} not found")
            return jsonify({"error": f"Input file {input_file_path} not found"}), 400

        # Load the input file
        inputs_df = pd.read_excel(input_file_path, engine='openpyxl')

        # Fill missing values and convert 'Description' and 'Supplier Name' columns to strings
        inputs_df['Description'] = inputs_df['Description'].fillna('Unknown').astype(str)
        inputs_df['Supplier Name'] = inputs_df['Supplier Name'].fillna('Unknown').astype(str)
        
        # Validate the input file format
        validate_file(inputs_df)

        # Load the feedback data
        feedback_df = load_feedback_file()

        # Create predictions for each row in the input file
        predictions = []
        combined_texts = get_combined_description_supplier(inputs_df)
        for user_input in combined_texts:
            prediction = predict_unspsc(user_input, feedback_df)
            predictions.append(prediction)

        # Create a DataFrame for the predictions
        output_df = pd.DataFrame(predictions, columns=['Original Input', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description'])
        
        # Save the predictions to the root directory
        output_file_path = 'PO Spend (Jun and Jul) Predictions.xlsx'
        output_df.to_excel(output_file_path, index=False)

        # Log the success message
        logger.info(f"Predictions saved to {output_file_path}")
        
        return jsonify({"message": f"Predictions saved to {output_file_path}"}), 200

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({"error": f"Error processing file: {e}"}), 500


# TODO: Implement feedback route
@app.route('/save-feedback', methods=['POST'])
def save_feedback():
    feedback = request.json
    if not feedback:
        return jsonify({"error": "No feedback provided"}), 400

    formatted_feedback = []
    for feedback_item in feedback:
        if len(feedback_item) != 5:
            return jsonify({"error": "Feedback item must have exactly 5 values"}), 400
        
        description = feedback_item[0]['value']
        correct_unspsc_code = feedback_item[1]['value']
        correct_unspsc_description = feedback_item[2]['value']
        correct_category_code = feedback_item[3]['value']
        correct_category_description = feedback_item[4]['value']
        
        formatted_feedback.append([
            description, 
            correct_unspsc_code, 
            correct_unspsc_description, 
            correct_category_code, 
            correct_category_description
        ])
    # Save feedback to file
    feedback_df = load_feedback_file()
    feedback_df = pd.concat([feedback_df, pd.DataFrame(formatted_feedback, columns=feedback_df.columns)], ignore_index=True)
    feedback_df.to_csv('outputs/feedback.csv', index=False)

    return jsonify({"message": "Feedback saved successfully"})

def get_combined_description_supplier(inputs_df: pd.DataFrame) -> list:
    return (inputs_df['Description'] + ' ' + inputs_df['Supplier Name']).tolist()

if __name__ == '__main__':
    app.run(debug=True)
