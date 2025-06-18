import json
from datetime import datetime
import os
from transformers import pipeline

# Define the path for the feedback data file
FEEDBACK_FILE = 'data/feedback.json'

# Initialize the sentiment analysis pipeline.
# We're using 'distilbert-base-uncased-finetuned-sst-2-english' as it's a good balance
# of performance and size for general sentiment analysis.
# This model outputs 'POSITIVE' or 'NEGATIVE'.
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def load_feedback():
    """
    Loads feedback data from the JSON file.
    If the file doesn't exist or is empty/corrupted, returns an empty list.
    """
    try:
        # Ensure the data directory exists before trying to open the file
        if not os.path.exists(os.path.dirname(FEEDBACK_FILE)):
            os.makedirs(os.path.dirname(FEEDBACK_FILE))
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return an empty list if the file is not found or cannot be decoded
        return []

def save_feedback(feedback_data):
    """
    Saves the feedback data to the JSON file.
    """
    # Ensure the data directory exists before saving
    if not os.path.exists(os.path.dirname(FEEDBACK_FILE)):
        os.makedirs(os.path.dirname(FEEDBACK_FILE))
    with open(FEEDBACK_FILE, 'w') as f:
        # Use indent=4 for pretty-printing the JSON file
        json.dump(feedback_data, f, indent=4)

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text using the pre-trained model.
    Returns 'POSITIVE' or 'NEGATIVE'.
    """
    if not text.strip(): # Handle empty feedback
        return "Neutral" # Or "N/A", depending on desired behavior for empty input

    # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.999}]
    result = sentiment_pipeline(text)[0]
    return result['label'] # Returns 'POSITIVE' or 'NEGATIVE'

def get_employee_feedback(employee_name, all_feedback):
    """
    Filters all feedback to get entries specific to a given employee.
    Case-insensitive matching for employee names.
    """
    return [f for f in all_feedback if f['employee_name'].lower() == employee_name.lower()]

