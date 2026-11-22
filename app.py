"""Flask image classifier based on the user's ResNet50 architecture."""
from io import BytesIO
from pathlib import Path

import torch
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from torch.nn import Sequential, Linear, SELU, Dropout, LogSigmoid
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, ToTensor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bt_resnet50_model.pt"
LABELS = ["None", "Meningioma", "Glioma", "Pituitary"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-only-change-this-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def create_model():
    # Matches the uploaded app.py architecture. Does not download pretrained weights.
    model = resnet50(weights=None)
    n_inputs = model.fc.in_features
    model.fc = Sequential(
        Linear(n_inputs, 2048), SELU(), Dropout(p=0.4),
        Linear(2048, 2048), SELU(), Dropout(p=0.4),
        Linear(2048, 4), LogSigmoid(),
    )
    return model


_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "Trained weights missing. Copy bt_resnet50_model.pt into the models folder."
            )
        model = create_model()
        # Load only trusted model files; PyTorch weights_only helps reduce pickle risk.
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def get_prediction(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = Compose([Resize((512, 512)), ToTensor()])(image).unsqueeze(0)
    with torch.inference_mode():
        outputs = get_model()(image_tensor.to(DEVICE))
        class_id = int(torch.argmax(outputs, dim=1).item())
    return LABELS[class_id]


@app.route("/")
def home():
    return render_template("DiseaseDet.html")


@app.route("/uimg", methods=["GET", "POST"])
def uimg():
    if request.method == "GET":
        return render_template("uimg.html")
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        flash("Please select an image first.")
        return redirect(url_for("uimg"))
    extension = uploaded.filename.rsplit(".", 1)[-1].lower()
    if "." not in uploaded.filename or extension not in ALLOWED_EXTENSIONS:
        flash("Upload a PNG, JPG, JPEG or GIF image.")
        return redirect(url_for("uimg"))
    try:
        prediction = get_prediction(uploaded.read())
    except (UnidentifiedImageError, OSError, ValueError):
        flash("Could not read that image. Please upload a valid image.")
        return redirect(url_for("uimg"))
    except (FileNotFoundError, RuntimeError) as exc:
        return render_template("error.html", message=str(exc)), 503
    return render_template("pred.html", result=prediction)


@app.errorhandler(413)
def too_large(_error):
    return render_template("error.html", message="File exceeds the 16 MB size limit."), 413


@app.errorhandler(500)
def server_error(_error):
    return render_template("error.html", message="An unexpected server error occurred."), 500


if __name__ == "__main__":
    app.run(debug=True)
