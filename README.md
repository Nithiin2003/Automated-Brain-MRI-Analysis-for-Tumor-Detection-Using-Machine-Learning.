# Brain MRI Image Classifier — PyCharm project

This project adapts the uploaded `app.py` into a PyCharm-ready Flask application with HTML templates and an upload interface.

## Setup in PyCharm

1. Extract this ZIP and choose **File → Open** in PyCharm, selecting the extracted folder.
2. Create/select a Python interpreter in **Settings → Project → Python Interpreter** (Python 3.10+ recommended).
3. Open PyCharm's Terminal and run `python -m pip install -r requirements.txt`.
4. **Required:** Place the original trained checkpoint at `models/bt_resnet50_model.pt`. It was **not included** in the upload, and the app cannot produce real predictions without it.
5. Right-click `app.py` → **Run 'app'** (or run `python app.py`).
6. Visit http://127.0.0.1:5000 in your browser.

## Notes

- The architecture and class order match the uploaded script: None, Meningioma, Glioma, Pituitary. The original misspelling of Pituitary was corrected only in the display label.
- PyTorch, torchvision and Python version compatibility matters. If the installation fails, follow the installation instructions on the official PyTorch website for your OS and GPU.
- The original checkpoint must contain a compatible model `state_dict`. Downloading standard ResNet50 weights is **not** a substitute for the trained classifier checkpoint.
- The server uses Flask `debug=True` for local development only. Do not expose the development server publicly.
- For **educational or research use only**, not medical diagnosis or clinical decision-making.

## Included PyCharm configuration

The `.idea/` directory contains PyCharm project/module settings and a **Run Flask App** run configuration. After opening the extracted folder, select the correct local Python interpreter (PyCharm cannot bundle one here), install `requirements.txt`, and select **Run Flask App** from the Run menu.
