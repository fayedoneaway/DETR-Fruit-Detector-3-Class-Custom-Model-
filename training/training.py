import os
import torch
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    TrainingArguments,
    Trainer
)
import cv2
import numpy as np


CLASSES = ["Apple", "Mango", "Orange"]

def load_yolo_split(img_dir, label_dir):
    image_paths = []
    targets = []

    for filename in os.listdir(img_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, filename.rsplit(".", 1)[0] + ".txt")

        image_paths.append(img_path)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        ann_list = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    cls, xc, yc, bw, bh = map(float, parts)
                    cls = int(cls)

                    if cls < 0 or cls >= len(CLASSES):
                        continue

                    x_min = (xc - bw/2) * w
                    y_min = (yc - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h

                    ann_list.append({
                        "bbox": [x_min, y_min, box_w, box_h],
                        "category_id": cls,
                        "area": box_w * box_h,
                        "iscrowd": 0
                    })

        targets.append({
            "image_id": len(image_paths) -1,
            "annotations": ann_list
        })
    return Dataset.from_dict({"image": image_paths, "target": targets})



train_ds = load_yolo_split("fruit.v1i.yolov8/train/images", "fruit.v1i.yolov8/train/labels")
val_ds   = load_yolo_split("fruit.v1i.yolov8/valid/images", "fruit.v1i.yolov8/valid/labels")
test_ds  = load_yolo_split("fruit.v1i.yolov8/test/images", "fruit.v1i.yolov8/test/labels")

def has_valid_annotations(example):
    anns = example["target"]["annotations"]
    if len(anns) == 0:
        return False
    for ann in anns:
        # bbox must have 4 numbers
        if "bbox" not in ann or len(ann["bbox"]) != 4:
            return False
        # width and height must be > 0
        if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            return False
    return True


train_ds = train_ds.filter(has_valid_annotations)
val_ds   = val_ds.filter(has_valid_annotations)


dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds
})


def make_transform(processor):
    def transform(example):
        img = Image.open(example["image"]).convert("RGB")

        encoding = processor(
            images=img,
            annotations=example["target"],
            return_tensors="pt"
        )

        # remove batch dimension
        encoding["pixel_values"] = encoding["pixel_values"].squeeze(0)
        encoding["pixel_mask"] = encoding["pixel_mask"].squeeze(0)
        encoding["labels"] = encoding["labels"][0]

        return encoding

    return transform



_debug_seen = False

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels
    }



def run_inference(model, processor, image_path, score_threshold=0.7):
    img = Image.open(image_path).convert("RGB")

    # Processor handles resizing + normalization
    inputs = processor(images=img, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # DETR expects (H, W)
    target_sizes = torch.tensor([img.size[::-1]], device=model.device)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=score_threshold
    )[0]

    return img, results



def draw_boxes(img, results, class_names):
    img = np.array(img)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{class_names[label]} {score:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return Image.fromarray(img)



def predict_folder(model, processor, folder_path, output_folder="predictions"):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)

            img, results = run_inference(model, processor, img_path)
            img_with_boxes = draw_boxes(img, results, CLASSES)

            save_path = os.path.join(output_folder, filename)
            img_with_boxes.save(save_path)

            print("Saved:", save_path)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pt = "facebook/detr-resnet-50"
    processor = DetrImageProcessor.from_pretrained(
        model_pt,
        size={"shortest_edge": 480, "longest_edge": 640}
    )

    model = DetrForObjectDetection.from_pretrained(
        model_pt,
        num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    ).to(device)

    for param in model.model.backbone.parameters():
        param.requires_grad = False

    transform = make_transform(processor)

    dataset["train"] = dataset["train"].map(
        transform,
        batched=False,
        remove_columns=["image", "target"]
    )

    dataset["validation"] = dataset["validation"].map(
        transform,
        batched=False,
        remove_columns=["image", "target"]
    )

    dataset["train"].set_format(
        type="torch",
        columns=["pixel_values", "pixel_mask", "labels"]
    )

    dataset["validation"].set_format(
        type="torch",
        columns=["pixel_values", "pixel_mask", "labels"]
    )
    training_args = TrainingArguments(
        output_dir="./detr-output",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=60,
        learning_rate=5e-5,
        logging_steps=20,
        save_steps=200,
        evaluation_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        # no_cuda=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    trainer.train()

    model.save_pretrained("model")
    processor.save_pretrained("model")

    # Reload cleanly for inference
    model = DetrForObjectDetection.from_pretrained("model").to(device)
    processor = DetrImageProcessor.from_pretrained("model")

    # Run inference on a folder of raw images
    predict_folder(model, processor, "raw_images")



if __name__ == "__main__":
    main()

