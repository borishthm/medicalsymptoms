from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import ast

app = FastAPI(title="Medical Recommendation API", version="1.0")

class SymptomsRequest(BaseModel):
    symptoms: List[str]

svc = pickle.load(open("svc.pkl", "rb"))
symptoms_df = pd.read_csv("symtoms_df.csv")
precautions_df = pd.read_csv("precautions_df.csv")
description_df = pd.read_csv("description.csv")
medications_df = pd.read_csv("medications.csv")
diets_df = pd.read_csv("diets.csv")
workout_df = pd.read_csv("workout_df.csv")
symptom_severity_df = pd.read_csv("Symptom-severity.csv")
training_data = pd.read_csv("Training.csv")

training_data.columns = training_data.columns.str.strip()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(training_data['prognosis'])

all_symptoms = list(svc.feature_names_in_)

def normalize_symptom(symptom: str) -> str:
    return symptom.strip().lower().replace(" ", "_")

def create_symptom_vector(symptoms, all_symptoms_list):
    symptom_vector = np.zeros(len(all_symptoms_list))
    for symptom in symptoms:
        normalized = normalize_symptom(symptom)
        if normalized in all_symptoms_list:
            index = all_symptoms_list.index(normalized)
            symptom_vector[index] = 1
    return symptom_vector.reshape(1, -1)

def get_disease_details(disease: str):
    try:
        desc = description_df[description_df['Disease'] == disease]['Description'].values[0]

        precautions = precautions_df[precautions_df['Disease'] == disease][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
        ].values.flatten().tolist()
        precautions = [p for p in precautions if pd.notna(p)]

        medications = ast.literal_eval(
            medications_df[medications_df['Disease'] == disease]['Medication'].values[0]
        )

        diets = ast.literal_eval(
            diets_df[diets_df['Disease'] == disease]['Diet'].values[0]
        )

        workouts = workout_df[workout_df['disease'] == disease]['workout'].tolist()

        return desc, precautions, medications, diets, workouts
    except Exception:
        return (
            "No description available.",
            [],
            [],
            [],
            []
        )

@app.get("/")
async def root():
    return {"message": "Medical Recommendation API is running 🚀"}

@app.post("/predict")
async def predict_disease(request: SymptomsRequest):
    symptoms = request.symptoms

    symptom_vector = create_symptom_vector(symptoms, all_symptoms)
    symptom_df = pd.DataFrame(symptom_vector, columns=all_symptoms)

    prediction_index = svc.predict(symptom_df)[0]
    predicted_disease = le.inverse_transform([prediction_index])[0]

    description_text, precautions_list, medications_list, diet_plan_list, workout_plan_list = get_disease_details(predicted_disease)

    return {
        "content": {
            "predicted_disease": predicted_disease,
            "disease_description": {
                "description": description_text,
                "medications": medications_list,
                "diet_plan": diet_plan_list,
                "precautions": precautions_list,
                "workout_plan": workout_plan_list
            }
        }
    }