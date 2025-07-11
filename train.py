# train.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

print("Starting the training process...")

# 1. Load Data
# To keep this simple, we'll use a built-in dataset from scikit-learn.
# We'll classify between two categories: 'sci.med' (medical science) and 'soc.religion.christian'.
# We can think of these as "Clinical" vs "Religious" sentiment for this example.
print("Loading data...")
categories = ['sci.med', 'soc.religion.christian']
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# These are our human-readable labels
class_names = ['Medical Science', 'Religion']

print(f"Data loaded. Number of training samples: {len(train_data.data)}")

# 2. Create a Machine Learning Pipeline
# A pipeline bundles a preprocessor and a classifier into one object.
# This makes it easy to apply the same transformations to training and prediction data.
print("Creating model pipeline...")
model_pipeline = Pipeline([
    # TfidfVectorizer: Converts text into a matrix of TF-IDF features (numerical representation of text).
    ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),

    # LogisticRegression: A simple but powerful classification algorithm.
    ('classifier', LogisticRegression(random_state=42))
])

# 3. Train the Model
# The .fit() method trains the entire pipeline. It first transforms the data using the
# vectorizer and then trains the classifier on the transformed data.
print("Training the model...")
model_pipeline.fit(train_data.data, train_data.target)
print("Model training complete.")

# 4. Save the Model
# We use joblib to save our trained pipeline to a file.
# This file will be loaded by our API later to make predictions.
model_filename = 'sentiment_model.joblib'
print(f"Saving model to {model_filename}...")
joblib.dump(model_pipeline, model_filename)

# Also save the class names for our API to use
class_names_filename = 'class_names.joblib'
print(f"Saving class names to {class_names_filename}...")
joblib.dump(class_names, class_names_filename)

print("Training script finished successfully!")