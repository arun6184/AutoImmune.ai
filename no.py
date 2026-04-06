from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)
    
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

flare_model = joblib.load("flare_predictor.pkl")    
    
    
    

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/clinical')
def clinical_page():
    return render_template('clinical_predict.html')

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    try:
        # Get data from form
        gender_map = {"Male": 0, "Female": 1}
        diagnosis_labels = {
            0: 'Autoimmune orchitis',
            1: "Graves' disease",
            2: 'Rheumatoid arthritis',
            3: 'Sjögren syndrome',
            4: 'Systemic lupus erythematosus (SLE)'
        }
        
        recommendations = {
    'Autoimmune orchitis': "Get plenty of rest, take prescribed anti-inflammatory medications, and avoid physical strain.",
    "Graves' disease": "Follow thyroid treatment regularly, manage stress, and avoid excessive iodine.",
    'Rheumatoid arthritis': "Stay active with low-impact exercise, use joint supports, and follow a balanced diet.",
    'Sjögren syndrome': "Use eye drops, stay hydrated, and avoid dry or dusty environments.",
    'Systemic lupus erythematosus (SLE)': "Avoid sun exposure, take medications on time, and get regular checkups."
}
        data = [
            request.form['id'],
            request.form['age'],
            gender_map[request.form['gender']],
            request.form['sickness_duration'],
            request.form['rbc_count'],
            request.form['hemoglobin'],
            request.form['hematocrit'],
            request.form['mcv'],
            request.form['mch'],
            request.form['mchc'],
            request.form['rdw'],
            request.form['reticulocyte_count'],
            request.form['wbc_count'],
            request.form['neutrophils'],
            request.form['lymphocytes'],
            request.form['monocytes'],
            request.form['eosinophils'],
            request.form['basophils'],
            request.form['plt_count'],
            request.form['mpv'],
            request.form['ana'],
            request.form['esbach'],
            request.form['mbl_level'],
            request.form['esr'],
            request.form['c3'],
            request.form['c4'],
            request.form['crp']
        ]

        features = np.array(data, dtype=float).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_diagnosis = diagnosis_labels.get(prediction, "Unknown Diagnosis")
        suggestion = recommendations.get(predicted_diagnosis, "Consult a specialist for proper care.")
        return render_template('clinical_predict.html',
                               prediction=f"Predicted Diagnosis: {predicted_diagnosis}",
                               suggestion=suggestion)

        # return render_template('clinical_predict.html', prediction=f"Disease Prediction: {predicted_diagnosis},")

    except Exception as e:
        return render_template('clinical_predict.html', prediction=f"Error: {str(e)}")



@app.route('/symptom_predict', methods=['GET', 'POST'])
def symptom_predict():
    if request.method == 'POST':
        file = request.files['note_file']
        if file and file.filename.endswith('.txt'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            with open(filepath, 'r') as f:
                note = f.read()

            # Predict using model
            pred = flare_model.predict([note])[0]
            result = "Flare" if pred == 1 else "No Flare"
            
            suggestions = {
                "Flare": "Get plenty of rest, take medications as prescribed, avoid stress, and stay hydrated. Consult your doctor for further treatment.",
                "No Flare": "Continue regular treatment and monitoring. Maintain a healthy lifestyle to prevent future flares."
            }

            return render_template('symptom_predict.html',
                                   note=note,
                                   prediction=result,
                                   suggestion=suggestions.get(result))
            

            # return render_template('symptom_predict.html', note=note, prediction=result)
        else:
            return render_template('symptom_predict.html', error="Please upload a valid .txt file.")

    return render_template('symptom_predict.html')









if __name__ == '__main__':
    app.run(debug=True)
