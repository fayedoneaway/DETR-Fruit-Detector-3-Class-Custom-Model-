README

OVERVIEW:

This is a small end to end project where I trained a custom DETR model to detect three fruit classes — Apple, Mango, Orange — and wrapped the whole thing in a clean Streamlit frontend so anyone can try it out.
I started with a Roboflow dataset that originally had five classes, but I narrowed it down to the three I actually wanted to work with. DETR is a heavier architecture and takes longer to settle compared to YOLO style models, so I trained for 60 epochs to give it enough room to converge.
The goal wasn’t to chase perfect mAP — it was to build a clean, working pipeline that shows:
• custom dataset prep
• training a transformer based detector
• cleaning up DETR’s noisy predictions
• building a simple UI for real world use
And honestly, the combo works really well.

1. Custom DETR Model (3 Classes)
• Started from a pretrained DETR checkpoint
• Fine tuned on a Roboflow dataset
• Reduced from 5 → 3 classes
• Trained for 60 epochs (DETR needs the extra runway)
• Cleaned up predictions using IoU filtering + top K selection
DETR tends to output a lot of “creative” boxes, so I added a small post processing step to keep things clean and client friendly.

2. Streamlit Frontend
I built a lightweight Streamlit app so you can:
• upload or select an image
• run inference
• see the predicted class
• view the bounding boxes
• get a simple, clean UI instead of raw Python output
It’s minimal on purpose — just enough to show the model working without distractions.

3. Files
• model.py — DETR loading + inference
• logic.py — filtering, top K, and post processing
• streamlit_app.py — the frontend
• draw_boxes.py — visualization utilities
• README.md — you’re reading it
Everything is modular so you can swap out the model or dataset later.

4. Training Notes
• DETR converges slower than CNN based detectors
• 60 epochs gave stable results
• AdamW optimizer
• Standard DETR transforms
• Custom label mapping for the 3 class setup

5. Project Structure
• app/frontend.py
• model/
• demo/
• training/
• README

Demo Output
The app shows:
• the original image
• the model’s top prediction
• clean bounding boxes (yellow for visibility on grayscale images)
No clutter, no overlapping duplicates.

Why This Project Exists
I wanted a small but complete example of:
• dataset → model → inference → UI
• using a transformer based detector
• handling DETR’s quirks
• building something that looks and feels like a real product
It’s not meant to be a giant research project — just a clean, working, portfolio ready demo.




