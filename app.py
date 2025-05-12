import gradio as gr
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("trained_model.sav", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_churn(gender, senior, partner, dependents, tenure, monthly, total):
    input_df = pd.DataFrame([{
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": int(senior),
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total)
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return "Yes (Customer may churn)" if prediction[0] == 1 else "No (Customer likely to stay)"

# Gradio UI
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Radio([0, 1], label="Senior Citizen"),
        gr.Radio(["Yes", "No"], label="Partner"),
        gr.Radio(["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, value=12, label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs="text",
    title="Customer Churn Predictor",
    description="Predict whether a customer will churn based on demographic and account data."
)

if __name__ == "__main__":
    interface.launch()
