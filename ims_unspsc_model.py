import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import joblib
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# Define custom stopwords
custom_stopwords = set([
    'the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'with', 'as', 'on', 'at', 'by', 'an', 'or', 'from', 'that'
])

# Preprocessing function to clean text
def preprocess_text(text: str) -> str:
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphanumeric characters
    words = text.split()  # Tokenize
    filtered_words = [word for word in words if word not in custom_stopwords]  # Remove stopwords
    return ' '.join(filtered_words)

# Load the combined dataset
combined_dataset_path = 'combined_training_data.xlsx'
combined_dataset = pd.read_excel(combined_dataset_path, engine='openpyxl')

# Preprocess the dataset
combined_dataset['Combined Text'] = combined_dataset['Description'] + ' ' + combined_dataset['Supplier Name'] + ' ' + combined_dataset['UNSPSC Description']
combined_dataset['Cleaned Text'] = combined_dataset['Combined Text'].apply(preprocess_text)

# Split the data into features and target
X = combined_dataset['Cleaned Text']
y = combined_dataset['UNSPSC Description']

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer()
X_vectorized = tfidf_vectorizer.fit_transform(X)

# Encode the categorical features with handle_unknown='ignore'
# on new versions the parameter sparse is called sparse_output
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore') # pyright: ignore[reportCallIssue]
supplier_encoded = encoder.fit_transform(combined_dataset[['Supplier Name']])

# Combine features
from scipy.sparse import hstack
X_combined = hstack([X_vectorized, supplier_encoded])

# Train the model
sgd_description_model = SGDClassifier()
sgd_description_model.fit(X_combined, y)

# Save models and artifacts
joblib.dump(tfidf_vectorizer, 'models/new/tfidf_vectorizer.joblib')
joblib.dump(sgd_description_model, 'models/new/sgd_description.joblib')
joblib.dump(encoder, 'models/new/encoder.joblib')

# Define the custom model class
class UNSPSCPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load models and dataset from context artifacts
        self.tfidf_vectorizer = joblib.load(context.artifacts['tfidf_vectorizer'])
        self.model_description = joblib.load(context.artifacts['sgd_description_model'])
        self.encoder = joblib.load(context.artifacts['encoder_model'])
        self.dataset = pd.read_excel(context.artifacts['training_data_path'], engine='openpyxl')

        # Check for missing columns
        required_columns = ['Description', 'Supplier Name', 'UNSPSC Description', 'Category Code', 'Category Description']
        missing_columns = [col for col in required_columns if col not in self.dataset.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in training data: {missing_columns}")

        # Handle missing values
        if self.dataset[required_columns].isnull().any().any():
            raise ValueError("Critical columns contain null values. Please clean the dataset.")

        # Preprocess dataset
        self.dataset['Combined Text'] = self.dataset['Description'] + ' ' + self.dataset['Supplier Name'] + ' ' + self.dataset['UNSPSC Description']
        self.dataset['Cleaned Text'] = self.dataset['Combined Text'].apply(preprocess_text)

    def lookup_details_by_description(self, predicted_description):
        # Lookup details by predicted description
        matched_rows = self.dataset[self.dataset['UNSPSC Description'] == predicted_description]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
            category_code = row['Category Code'] if not pd.isna(row['Category Code']) else 'Category Code not found'
            category_description = row['Category Description'] if not pd.isna(row['Category Description']) else 'Category Description not found'
            return row['UNSPSC Code'], category_code, category_description
        else:
            return None, 'Not found', 'Not found'

    def predict(self, context, model_input, params=None):
        # Check for None or empty DataFrame
        if model_input is None or model_input.empty:
            raise ValueError("Model input is empty or None")

        # Preprocess input text
        model_input['Combined Text'] = (model_input['Description'] + ' ' + model_input['Supplier Name']).apply(preprocess_text)

        predictions = []

        for _, row in model_input.iterrows():
            combined_text = row['Combined Text']
            vectorized_input = self.tfidf_vectorizer.transform([combined_text])

            # Prepare the input for the OneHotEncoder
            supplier_input_df = pd.DataFrame({'Supplier Name': [row['Supplier Name']]})
            supplier_encoded_input = self.encoder.transform(supplier_input_df)

            # Combine features
            user_features = hstack([vectorized_input, supplier_encoded_input])

            try:
                predicted_description = self.model_description.predict(user_features)[0]
            except Exception as e:
                raise RuntimeError(f"Prediction failed: {e}")

            predicted_code, category_code, category_description = self.lookup_details_by_description(predicted_description)

            # Collect predictions
            predictions.append({
                'Description': row['Description'],
                'Supplier Name': row['Supplier Name'],
                'UNSPSC Code': predicted_code,
                'UNSPSC Description': predicted_description,
                'Category Code': category_code,
                'Category Description': category_description
            })

        # Return the predictions as a DataFrame
        result_df = pd.DataFrame(predictions)
        print("Predictions Successfully Created")
        return result_df

# Example input for signature inference
example_input = pd.DataFrame({
    'Description': ['30% DOWN WITH PO ISSUANCE'],
    'Supplier Name': ['ABB INC']
})

# Infer the model signature
signature = infer_signature(example_input, pd.DataFrame(columns=[
    'Description', 'Supplier Name', 'UNSPSC Code', 'UNSPSC Description', 'Category Code', 'Category Description'
]))

# Example of input_example
input_example = example_input.copy()

# Register the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="UNSPSCPredictor",
        python_model=UNSPSCPredictor(),
        artifacts={
            "tfidf_vectorizer": "models/new/tfidf_vectorizer.joblib",
            "sgd_description_model": "models/new/sgd_description.joblib",
            "encoder_model": "models/new/encoder.joblib",
            "training_data_path": "combined_training_data.xlsx"
        },
        signature=signature,
        input_example=input_example,
        conda_env="conda.yaml"
    )
