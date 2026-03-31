import torch
import numpy as np
import evaluate
import pandas as pd
import os
import matplotlib.pyplot as plt 
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
from torchvision.transforms import v2
from datasets import load_dataset, Image

torch.cuda.empty_cache()

dataset = load_dataset("imagefolder", data_dir="dataset/data", drop_labels=False)
labels = dataset["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}


model_checkpoint = "timm/resnetrs200.tf_in1k"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)


train_transforms = v2.Compose([
    v2.RandomRotation(degrees=180),       
    v2.RandomHorizontalFlip(p=0.5),      
    v2.ColorJitter(brightness=0.3, saturation=0.2, hue=0.1),
    v2.GaussianBlur(kernel_size=(1,3), sigma=(0.1,5.0))     
])


def preprocess_train(example_batch):
    images = [train_transforms(x.convert("RGB")) for x in example_batch["image"]]
    inputs = image_processor(images, return_tensors="pt")
    inputs["label"] = example_batch["label"]
    return inputs


def preprocess_val(example_batch):
    images = [x.convert("RGB") for x in example_batch["image"]]
    
    inputs = image_processor(images, return_tensors="pt")
    inputs["pixel_values"]

    inputs["label"] = example_batch["label"]
    return inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


dataset["train"].set_transform(preprocess_train)
dataset["validation"].set_transform(preprocess_val)
dataset["test"].set_transform(preprocess_val)


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)


metric = evaluate.load("accuracy")

training_args = TrainingArguments(
    output_dir="./log",
    remove_unused_columns=False, 
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=7e-4,
    per_device_train_batch_size=48,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    label_smoothing_factor=0.1,
    dataloader_num_workers=8,
    num_train_epochs=10,
    warmup_ratio=0.05,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(), 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"], 
    processing_class=image_processor, 
    compute_metrics=compute_metrics,  
    data_collator=DefaultDataCollator(),
)

trainer.train()




#==========================================
# VISUALIZATION 
# =========================================
print("\nGenerating Visualizations...")


def plot_training_curves(log_history):
    train_loss, val_loss, val_acc, epochs_train, epochs_val = [], [], [], [], []
    
    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_loss.append(log["loss"])
            epochs_train.append(log["epoch"])
        elif "eval_loss" in log and "epoch" in log:
            val_loss.append(log["eval_loss"])
            val_acc.append(log["eval_accuracy"])
            epochs_val.append(log["epoch"])

    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_train, train_loss, label="Train Loss", color="blue")
    plt.plot(epochs_val, val_loss, label="Val Loss", color="red", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_val, val_acc, label="Val Accuracy", color="green", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()
    print("- Saved training_curves.png")

plot_training_curves(trainer.state.log_history)

print("\nGenerating predictions for benchmarking...")


output = trainer.predict(dataset["test"])
preds = np.argmax(output.predictions, axis=-1)


dataset["test"].reset_format() 
test_metadata = dataset["test"].cast_column("image", Image(decode=False))

image_names = []
for i in range(len(test_metadata)):
    file_path = test_metadata[i]["image"]["path"]
    image_names.append(os.path.basename(file_path))

predicted_labels = [id2label[p] for p in preds]

df = pd.DataFrame({
    "image_name": image_names,
    "pred_label": predicted_labels
})

df.to_csv("submission.csv", index=False)
print(f"Success! Saved {len(df)} rows to submission.csv")