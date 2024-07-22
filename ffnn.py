import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import logging
import re
import joblib
import coloredlogs
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load datasets with error handling
try:
    unspsc_large_df = pd.read_excel('unspsc_large_dataset.xlsx')
    unspsc_df = pd.read_excel('unspsc_dataset.xlsx')
    category_df = pd.read_excel('category_dataset.xlsx')
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
X = vectorizer.fit_transform(merged_df['Combined Text']).toarray()

# Define the target variable
y = merged_df['UNSPSC'].factorize()[0]
num_classes = len(np.unique(y))

# One-hot encode the target variable
y_encoded = to_categorical(y, num_classes=num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the FFNN model
model = Sequential()
model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate the model
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted', zero_division=1)
recall = recall_score(np.argmax(y_test, axis=1), y_pred, average='weighted', zero_division=1)
f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')

logger.info(f'Accuracy: {accuracy}')
logger.info(f'Precision: {precision}')
logger.info(f'Recall: {recall}')
logger.info(f'F1 Score: {f1}')

# Save the model and vectorizer
model.save('unspsc_predictor_model.h5')
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
    combined_input = vectorizer.transform([cleaned_description]).toarray()
    predicted_code_index = np.argmax(model.predict(combined_input), axis=1)[0]
    predicted_code = unspsc_large_df['UNSPSC'].unique()[predicted_code_index]
    predicted_description = unspsc_large_df.loc[unspsc_large_df['UNSPSC'] == predicted_code, 'UNSPSC Description'].values[0]

    return predicted_code, predicted_description

def get_category(predicted_code, predicted_description):
    logger.info(f'Searching for category using code: {predicted_code} and description: {predicted_description}')

    # Attempt to find the category using the predicted UNSPSC code
    category_row = category_df[category_df['UNSPSC'] == predicted_code]
    
    if not category_row.empty:
        return [category_row.iloc[0]['Category Level 4']]

    # If the code does not match, check for keyword presence in the predicted description
    matches = []
    for _, row in category_df.iterrows():
        description = str(row['UNSPSC Description (English)']).lower()
        if description in predicted_description.lower():
            matches.append(row['Category Level 4'])

    if matches:
        logger.info(f'Matching categories found: {matches[:5]}')
        return matches[:5]

    logger.warning(f"No category found for code: {predicted_code} and description: {predicted_description}")
    return None

def main():
    while True:
        user_input = input("Enter a product description (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            predicted_code, predicted_description = predict_unspsc(user_input)
            category = get_category(predicted_code, predicted_description)
            logger.info(f'Predicted UNSPSC Code: {predicted_code}')
            logger.info(f'Predicted Description: {predicted_description}')
            if category:
                logger.info(f'Category: {category}')
            else:
                logger.info('Category not found.')
        except Exception as e:
            logger.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
