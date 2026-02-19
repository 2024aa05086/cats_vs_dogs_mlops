import io
import os
import requests
import gradio as gr
from PIL import Image

# API_URL from environment variable
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


def predict_from_api(img: Image.Image):
    if img is None:
        return "No image provided.", None

    # Convert PIL Image -> JPEG bytes
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)

    files = {"file": ("upload.jpg", buf, "image/jpeg")}

    try:
        r = requests.post(API_URL, files=files, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"API request failed: {e}", None

    data = r.json()
    label = data.get("label", "unknown")
    # gr.Label expects a dict of {string_label: float_probability}
    probs = data.get("probabilities", {})

    return f"Predicted: {label}", probs


with gr.Blocks(title="Cats vs Dogs Predictor") as demo:
    gr.Markdown("# Cats vs Dogs - Prediction UI")

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload Image")
            btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            out_text = gr.Textbox(label="Top Prediction", interactive=False)
            # gr.Label is the correct component for classification probabilities
            out_probs = gr.Label(label="Probabilities", num_top_classes=2)

    btn.click(
        fn=predict_from_api,
        inputs=inp,
        outputs=[out_text, out_probs]
    )

    gr.Markdown(f"Backend API: `{API_URL}`")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )