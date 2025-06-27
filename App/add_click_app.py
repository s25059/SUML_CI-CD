import gradio as gr
import skops.io as sio
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "ad_click_pipeline.skops")
pipe = sio.load(MODEL_PATH, trusted=["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList"])

def predict_click(gender, device_type, ad_position, browsing_history, time_of_day, age):
    columns = ["gender", "device_type", "ad_position", "browsing_history", "time_of_day", "age"]
    data = [[gender, device_type, ad_position, browsing_history, time_of_day, age]]
    X = pd.DataFrame(data, columns=columns)
    predicted_click = pipe.predict(X)[0]
    return f"Clicked: {'Yes' if predicted_click == 1 else 'No'}"

inputs = [
    gr.Radio(["Male", "Female", "Non-Binary"], label="Gender"),
    gr.Radio(["Desktop", "Mobile", "Tablet"], label="Device Type"),
    gr.Radio(["Top", "Side", "Bottom"], label="Ad Position"),
    gr.Radio(["Shopping", "Education", "Entertainment", "Social Media", "News"], label="Browsing History"),
    gr.Radio(["Afternoon", "Night", "Evening", "Morning"], label="Time of Day"),
    gr.Slider(18, 64, step=1, label="Age"),
]
outputs = gr.Label()

examples = [
    ["Male", "Desktop", "Top", "Shopping", "Afternoon", 25],
    ["Female", "Mobile", "Side", "News", "Night", 36],
    ["Non-Binary", "Tablet", "Bottom", "Entertainment", "Morning", 44],
]

gr.Interface(
    fn=predict_click,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title="Ad Click Prediction",
    description="Enter user details to predict if they will click on an ad.",
    article="This demo predicts whether a user will click on an advertisement, based on demographic and behavioral features. Powered by a Random Forest model.",
    theme=gr.themes.Soft(),
).launch()
