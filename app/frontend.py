import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np
import os

MODEL_DIR = "model"
IMAGE_DIR = "raw_images"
THRESH = 0.6


st.set_page_config("DETR Demo", layout="centered")
st.title("Apple / Orange / Mango DETR Object Detection")

CLASSES = ["Apple", "Mango", "Orange"]


@st.cache_resource
def load_model():
    processor = DetrImageProcessor.from_pretrained(MODEL_DIR)
    model = DetrForObjectDetection.from_pretrained(MODEL_DIR)
    return processor, model


def run_detr(processor, model, pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=THRESH
    )[0]
    return results


def draw_boxes(pil_img, results):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(x) for x in box]
        cls = model.config.id2label[int(label)]
        draw.rectangle(box, outline="yellow", width=3)
        draw.text((box[0], box[1] - 25), cls, fill="yellow", font=font)

    return img


def filter_overlaps(results, iou_thresh=0.5):
    scores = [float(s) for s in results["scores"]]
    labels = [int(l) for l in results["labels"]]
    boxes = [b.tolist() for b in results["boxes"]]

    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep_scores = []
    keep_labels = []
    keep_boxes = []

    def iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union

    for i in idxs:
        box = boxes[i]
        label = labels[i]
        score = scores[i]

        keep = True
        for kb in keep_boxes:
            if iou(box, kb) > iou_thresh:
                keep = False
                break

        if keep:
            keep_scores.append(score)
            keep_labels.append(label)
            keep_boxes.append(box)

    results["scores"] = keep_scores
    results["labels"] = keep_labels
    results["boxes"] = keep_boxes
    return results



processor, model = load_model()

model.config.id2label = {0: "Apple", 1: "Mango", 2: "Orange"}
model.config.label2id = {"Apple": 0, "Mango": 1, "Orange": 2}


files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]
choice = st.selectbox("Select an image:", files)

if choice:
    raw_path = os.path.join(IMAGE_DIR, choice)
    raw_img = Image.open(raw_path).convert("RGB")

    st.subheader("DETR predicts...")
    st.image(raw_img)

    results = run_detr(processor, model, raw_img)

    results = filter_overlaps(results, iou_thresh=0.5)

    TOP_K = 4  # or 1, 2, 3, 4 — your choice
    results["scores"] = results["scores"][:TOP_K]
    results["labels"] = results["labels"][:TOP_K]
    results["boxes"] = results["boxes"][:TOP_K]

    if len(results["labels"]) > 0:
        top_label = int(results["labels"][0])
        class_name = model.config.id2label[top_label]
    else:
        class_name = "Model Is Unable To Recognize"

    st.subheader(f"... this is a {class_name}.")

    labeled = draw_boxes(raw_img, results)
    st.image(labeled)
