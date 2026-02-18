import io
import os
import requests
import gradio as gr
from PIL import Image

# ---------------------------------------------------
# API configuration (Docker/K8s friendly)
# ---------------------------------------------------
API_URL = os.getenv("API_URL", "http://api:8000/predict")


# ---------------------------------------------------
# Prediction function
# ---------------------------------------------------
def predict_from_api(img: Image.Image):

    if img is None:
        return "No image provided.", None

    try:
        # Convert PIL -> JPEG bytes
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        buf.seek(0)

        files = {"file": ("upload.jpg", buf, "image/jpeg")}

        response = requests.post(API_URL, files=files, timeout=60)

        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text}", None

        data = response.json()

        label = data.get("label", "unknown")
        probs = data.get("probabilities", {})

        return f"Predicted: {label}", probs

    except Exception as e:
        return f"Request failed: {str(e)}", None


# ---------------------------------------------------
# UI Layout
# ---------------------------------------------------
with gr.Blocks(title="Cats vs Dogs Predictor") as demo:

    gr.Markdown("# Cats vs Dogs - Prediction UI")
    gr.Markdown(
        "Upload an image → UI calls FastAPI `/predict` → displays prediction."
    )

    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image")

        with gr.Column():
            out_text = gr.Textbox(label="Result", lines=2)
            out_label = gr.Label(label="Probabilities")

    btn = gr.Button("Predict", variant="primary")

    btn.click(
        fn=predict_from_api,
        inputs=inp,
        outputs=[out_text, out_label],
    )

    gr.Markdown(f"API Endpoint: `{API_URL}`")


# ---------------------------------------------------
# Launch (Container-safe)
# ---------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
