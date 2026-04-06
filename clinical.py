
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load labels
df = pd.read_csv("labels.csv")

# Read clinical notes
def read_note(filename):
    with open(os.path.join("clinical_notes", filename), "r", encoding="utf-8") as f:
        return f.read()

df["text"] = df["filename"].apply(read_note)

# Encode labels
df["label"] = df["label"].map({"flare": 1, "no flare": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)

# NLP pipeline
model = make_pipeline(
    TfidfVectorizer(stop_words="english"),
    MultinomialNB()
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["No Flare", "Flare"]))

# Save model (optional)
import joblib
joblib.dump(model, "flare_predictor.pkl")




import joblib

# Load model
model = joblib.load("flare_predictor.pkl")

# New note
new_note = """
The patient presents with increased joint swelling, prolonged morning stiffness, and fatigue. Labs pending.
"""

# Predict
pred = model.predict([new_note])[0]
print("Prediction:", "Flare" if pred == 1 else "No Flare")




