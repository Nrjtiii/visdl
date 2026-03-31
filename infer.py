import torch
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

def run_batch_inference(model_path, data_dir, output_csv="submission.csv"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    model.eval()

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        f for f in os.listdir(data_dir) if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        return

    results = []

    with torch.no_grad():
        for filename in tqdm(image_files):
            img_path = os.path.join(data_dir, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_class_idx]

                results.append({
                    "image_name": filename,
                    "pred_label": predicted_label
                })
            except Exception:
                continue

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":

    CHECKPOINT_DIR = "./log/checkpoint-5184"        # The folder containing config.json and pytorch_model.bin
    INPUT_FOLDER = "./dataset/data/test" # The directory containing images you want to label
    OUTPUT_FILE = "submission.csv"

    run_batch_inference(CHECKPOINT_DIR, INPUT_FOLDER, OUTPUT_FILE)