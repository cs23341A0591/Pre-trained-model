import io

import cv2
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="YOLO11n Object Detection", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO11n model...")
def load_model(weights: str = "yolo11n.pt"):
    # Downloads yolo11n.pt automatically the first time
    return YOLO(weights)


def detect(model, pil_image: Image.Image, conf: float):
    """Run detection; return annotated RGB image (numpy) and a detections DataFrame."""
    result = model(pil_image, conf=conf, verbose=False)[0]

    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        rows.append(
            {
                "Object": model.names[int(box.cls[0])],
                "Confidence": round(float(box.conf[0]), 3),
                "X1": int(x1),
                "Y1": int(y1),
                "X2": int(x2),
                "Y2": int(y2),
            }
        )
    df = pd.DataFrame(rows, columns=["Object", "Confidence", "X1", "Y1", "X2", "Y2"])

    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)  # plot() returns BGR
    return annotated_rgb, df


def to_png_bytes(rgb_array) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    return buf.getvalue()


def process_images(model, items, conf):
    """items: list of (name, PIL image). Shows results for each."""
    summary = []
    for idx, (name, image) in enumerate(items):
        st.markdown("---")
        st.subheader(f"{idx + 1}. {name}")

        with st.spinner("Detecting objects..."):
            annotated, df = detect(model, image, conf)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Input image", use_container_width=True)
        col2.image(annotated, caption="Detection result", use_container_width=True)

        st.metric("Objects detected", len(df))
        base = name.rsplit(".", 1)[0]

        if df.empty:
            st.warning("No objects found at this confidence threshold. Try lowering it.")
        else:
            left, right = st.columns([3, 2])
            with left:
                st.markdown("**Detections**")
                st.dataframe(df, use_container_width=True, hide_index=True)
            with right:
                st.markdown("**Object counts**")
                st.bar_chart(df["Object"].value_counts())

            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download annotated image",
                data=to_png_bytes(annotated),
                file_name=f"detected_{base}.png",
                mime="image/png",
                key=f"img_{idx}_{name}",
            )
            d2.download_button(
                "⬇️ Download detections (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"detections_{base}.csv",
                mime="text/csv",
                key=f"csv_{idx}_{name}",
            )

        summary.append(
            {
                "Image": name,
                "Number of Objects": len(df),
                "Detected Classes": ", ".join(df["Object"].tolist()) if not df.empty else "",
            }
        )

    if len(items) > 1:
        st.markdown("---")
        st.subheader("Summary of all images")
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="test_summary.csv",
            mime="text/csv",
        )


def load_pil(file) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(file)).convert("RGB")


# ---------------- UI ----------------
st.title("🎯 Object Detection using YOLO11n")
st.caption("Upload images or use your webcam to detect objects with a pre-trained YOLO11n model.")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05)
    st.markdown("---")
    st.markdown(
        "**Project:** Object Detection in Images Using YOLO11n  \n"
        "**Model:** Pre-trained `yolo11n.pt` (COCO, 80 classes)"
    )

model = load_model()

tab_upload, tab_webcam = st.tabs(["📁 Upload Images", "📷 Webcam"])

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        process_images(model, [(f.name, load_pil(f)) for f in uploaded_files], conf)
    else:
        st.info("👆 Upload an image to get started.")

with tab_webcam:
    st.write("Allow camera access in your browser, then click **Take Photo**.")
    photo = st.camera_input("Take a photo")
    if photo is not None:
        process_images(model, [("webcam_capture.png", load_pil(photo))], conf)
