# YOLO11n Object Detection – Streamlit App

## Project structure
```
yolo_streamlit/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Upload size / theme settings
└── README.md
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501. `yolo11n.pt` is downloaded automatically on first run.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → New app → select the repo → main file: `app.py`.
3. Deploy.

## Features
- Upload one or many images (JPG, PNG, BMP, WEBP)
- Webcam tab: capture a photo from your camera and detect objects (needs HTTPS or localhost)
- Adjustable confidence threshold
- Bounding boxes, labels and confidence scores
- Detection table and object counts
- Download annotated image and CSV results
- Multi-image summary table
