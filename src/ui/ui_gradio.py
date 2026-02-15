import io
import requests
import gradio as gr
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"


def predict_from_api(img: Image.Image):
    if img is None:
        return "No image provided.", None

    # Convert PIL Image -> JPEG bytes
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)

    files = {"file": ("upload.jpg", buf, "image/jpeg")}
    r = requests.post(API_URL, files=files, timeout=60)

    if r.status_code != 200:
        return f"API error {r.status_code}: {r.text}", None

    data = r.json()
    label = data["label"]
    probs = data["probabilities"]  # {"cat": x, "dog": y}

    # Gradio Label expects dict of class->confidence
    return f"Predicted: {label}", probs


with gr.Blocks(title="Cats vs Dogs Predictor") as demo:
    gr.Markdown("#  Cats vs Dogs - Prediction UI")
    gr.Markdown("Upload an image → calls FastAPI `/predict` → shows label + probabilities.")

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image")
        with gr.Column():
            out_text = gr.Textbox(label="Result", lines=2)
            out_label = gr.Label(label="Probabilities")

    btn = gr.Button("Predict")
    btn.click(fn=predict_from_api, inputs=inp, outputs=[out_text, out_label])

    gr.Markdown("API must be running at: `http://127.0.0.1:8000`")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
