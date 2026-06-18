import os
import torch
from PIL import Image
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

CLASSES = [
    "Apple", "Orange", "Mango"
]

def run_inference(model, processor, image_path, threshold=0.6):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print("BAD IMAGE:", image_path)
        print("ERROR:", e)
        return None, None

    np_img = np.array(image)

    if np_img.size == 0:
        print("EMPTY IMAGE:", image_path)
        return None, None

    h, w = np_img.shape[:2]

    inputs = processor(images=image, return_tensors="pt").to(model.device)

    try:
        outputs = model(**inputs)
    except Exception as e:
        print("INFERENCE ERROR on:", image_path)
        print(e)
        return None, image

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    return results, image



def draw_boxes(image, results):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box]
        cls = model.config.id2label[label.item()]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        cv2.putText(img, f"{cls} {score:.2f}", (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return img



def show_image_window(img):
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def predict_folder(model, processor, folder_path, output_folder="predictions"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, filename)

            results, pil_img = run_inference(model, processor, img_path)
            save_path = os.path.join(output_folder, filename)

            drawn_img = draw_boxes(pil_img, results)
            cv2.imwrite(save_path, drawn_img)

            print("Saved:", save_path)
            print("Detections:", len(results["scores"]))



if __name__ == "__main__":
    model_path = "model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path).to(device)

    model.config.id2label = {0: "Apple", 1: "Mango", 2: "Orange"}
    model.config.label2id = {"Apple": 0, "Mango": 1, "Orange": 2}

    predict_folder(model, processor, "fruit.v1i.yolov8/test/images")
