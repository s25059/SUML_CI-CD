import gradio as gr
import skops.io as sio

pipe = sio.load("./Model/ad_click_pipeline.skops", trusted=["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList"])

def predict_click(gender, device_type, ad_position, browsing_history, time_of_day, age):
    """
    Predict ad click based on user features.
    """
    features = [[gender, device_type, ad_position, browsing_history, time_of_day, age]]
    predicted_click = pipe.predict(features)[0]
    return f"Clicked: {'Yes' if predicted_click == 1 else 'No'}"

inputs = [
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Radio(["Mobile", "Desktop", "Tablet"], label="Device Type"),
    gr.Radio(["Top", "Bottom", "Sidebar"], label="Ad Position"),
    gr.Radio(["Sports", "News", "Tech", "Other"], label="Browsing History"),
    gr.Radio(["Morning", "Afternoon", "Evening", "Night"], label="Time of Day"),
    gr.Slider(10, 80, step=1, label="Age"),
]
outputs = gr.Label()

examples = [
    ["Male", "Mobile", "Top", "Sports", "Morning", 25],
    ["Female", "Desktop", "Sidebar", "News", "Evening", 37],
    ["Male", "Tablet", "Bottom", "Tech", "Night", 44],
]

title = "Ad Click Prediction"
description = "Enter user details to predict if they will click on an ad."
article = (
    "This demo predicts whether a user will click on an advertisement, "
    "based on demographic and behavioral features. Powered by a Random Forest model."
)

gr.Interface(
    fn=predict_click,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
