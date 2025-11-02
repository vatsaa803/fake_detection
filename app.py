import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import io
import cv2
import numpy as np
import tempfile
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- Streamlit Setup ---
st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Deepfake Detection Dashboard")
st.write("Upload an **image** or **video** to detect deepfakes, visualize Grad-CAM heatmaps, and download detailed reports.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("deepfake_detector.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Grad-CAM ---
def generate_gradcam(model, image_tensor):
    grad_cam = None
    feature_maps = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grad_cam
        grad_cam = grad_out[0].detach()

    target_layer = model.conv_head
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    model.zero_grad()
    output[0, pred_class].backward()

    weights = grad_cam.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam

# --- Prediction ---
def predict_image(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return ("Real" if pred.item() == 0 else "Fake"), conf.item() * 100, probs[0, 1].item() * 100, img_tensor

# --- PDF Generator ---
def generate_pdf(report_data, chart_path=None, heatmap_paths=None, explanation=""):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    # --- Title ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1 * inch, height - 1 * inch, "Deepfake Detection Report")

    # --- Summary Data ---
    c.setFont("Helvetica", 12)
    y = height - 1.5 * inch
    for key, value in report_data.items():
        c.drawString(1 * inch, y, f"{key}: {value}")
        y -= 0.3 * inch

    # --- Explanation ---
    y -= 0.4 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y, "Analysis & Explanation:")
    y -= 0.3 * inch
    c.setFont("Helvetica", 11)
    text = c.beginText(1 * inch, y)
    text.textLines(explanation)
    c.drawText(text)

    # --- Graph ---
    if chart_path:
        y -= 4 * inch
        c.drawImage(chart_path, 1 * inch, y, width=5.5 * inch, preserveAspectRatio=True)

    # --- Heatmaps ---
    if heatmap_paths:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1 * inch, height - 1 * inch, "Grad-CAM Heatmaps:")
        y = height - 1.5 * inch
        for path in heatmap_paths[:3]:
            c.drawImage(path, 1 * inch, y - 3 * inch, width=5.5 * inch, preserveAspectRatio=True)
            y -= 3.5 * inch
            if y < 2 * inch:
                c.showPage()
                y = height - 1.5 * inch

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(1 * inch, 0.5 * inch, "Developed by Vatsa and Sanjay")

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    report_data = {}
    heatmap_paths = []
    chart_path = None
    explanation = ""

    # --- Image ---
    if "image" in file_type:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        label, confidence, fake_prob, tensor = predict_image(image)
        st.write(f"### Prediction: {'🟢 Real' if label=='Real' else '🔴 Fake'} ({confidence:.2f}% confidence)")

        cam = generate_gradcam(model, tensor)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img = np.array(image.resize((224, 224)))
        overlay = np.uint8(0.6 * heatmap + 0.4 * img)

        st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)

        temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        cv2.imwrite(temp_heatmap, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_paths.append(temp_heatmap)

        report_data = {
            "File Type": "Image",
            "Prediction": label,
            "Confidence": f"{confidence:.2f}%",
            "Fake Probability": f"{fake_prob:.2f}%",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        explanation = (
            f"The model analyzed this image using EfficientNet-B0 and Grad-CAM visualization. "
            f"The prediction '{label}' was made with {confidence:.2f}% confidence. "
            f"A fake probability of {fake_prob:.2f}% indicates that the system believes the image is "
            f"{'highly likely to be manipulated' if label == 'Fake' else 'authentic with minimal manipulation signals'}."
        )

    # --- Video ---
    elif "video" in file_type:
        st.video(uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, total_frames // 40)
        frame_indices = list(range(0, total_frames, sample_rate))

        results = []
        progress = st.progress(0)

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            label, conf, fake_prob, tensor = predict_image(image)
            results.append((frame_idx, label, conf, fake_prob))

            if fake_prob > 70:
                cam = generate_gradcam(model, tensor)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                overlay = np.uint8(0.6 * heatmap + 0.4 * np.array(image.resize((224, 224))))
                temp_heatmap = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                cv2.imwrite(temp_heatmap, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                heatmap_paths.append(temp_heatmap)

            progress.progress((i + 1) / len(frame_indices))

        cap.release()

        # --- Graph ---
        frame_nums = [f for f, _, _, _ in results]
        fake_probs = [p for _, _, _, p in results]
        plt.figure(figsize=(6, 3))
        plt.plot(frame_nums, fake_probs, marker="o")
        plt.title("Fake Probability Across Frames")
        plt.xlabel("Frame Number")
        plt.ylabel("Fake Probability (%)")
        plt.grid(True)
        chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        plt.savefig(chart_path, bbox_inches="tight")
        st.pyplot(plt)

        fake_frames = sum(1 for _, lbl, _, _ in results if lbl == "Fake")
        avg_conf = np.mean([c for _, _, c, _ in results])
        avg_fake_prob = np.mean(fake_probs)
        verdict = "Deepfake Detected" if fake_frames > len(results) / 2 else "Real Video"

        st.subheader("🎬 Final Verdict")
        if verdict == "Deepfake Detected":
            st.error(f"🔴 {verdict} (Avg Fake Probability: {avg_fake_prob:.2f}%)")
        else:
            st.success(f"🟢 {verdict} (Avg Fake Probability: {avg_fake_prob:.2f}%)")

        report_data = {
            "File Type": "Video",
            "Frames Analyzed": len(results),
            "Fake Frames": fake_frames,
            "Average Confidence": f"{avg_conf:.2f}%",
            "Average Fake Probability": f"{avg_fake_prob:.2f}%",
            "Verdict": verdict,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        explanation = (
            f"The video was analyzed using smart frame sampling ({len(results)} frames). "
            f"Each frame was evaluated for fake probability using EfficientNet-B0. "
            f"The model found {fake_frames} suspicious frames with an average fake probability of {avg_fake_prob:.2f}%. "
            f"This suggests the content is {'likely a deepfake' if verdict == 'Deepfake Detected' else 'authentic'}, "
            f"based on detected manipulation patterns and spatial inconsistencies."
        )

    # --- PDF ---
    pdf_buffer = generate_pdf(report_data, chart_path, heatmap_paths, explanation)
    st.download_button(
        label="📄 Download Detailed PDF Report",
        data=pdf_buffer,
        file_name=f"deepfake_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

st.caption("Model: EfficientNet-B0 | Grad-CAM Visualization | Smart Sampling | Developed by Vatsa and Sanjay")
