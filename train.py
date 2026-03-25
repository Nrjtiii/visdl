import torch
import numpy as np
import evaluate
import pandas as pd
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
from datasets import load_dataset 

# 1. Load your local dataset
# Note: Ensure "cv_hw1_data/data/train" is a DatasetDict containing "train" and "test"
dataset = load_dataset("imagefolder", data_dir="dataset/data", drop_labels=False)


#dataset = load_dataset("nsarker/flower-detection")

print(f"Splits found: {dataset.keys()}")
print(f"{dataset}")
# Extract labels
labels = dataset["train"].features["label"].names
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}



model_checkpoint = "microsoft/resnet-152"

# 2. Load Processor & Define Preprocessing
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

def transform(example_batch):
    # Take a list of PIL images and turn them into pixel values
    inputs = image_processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["label"] = example_batch["label"]
    return inputs

# Apply transforms on-the-fly (saves RAM/Disk space)
dataset.set_transform(transform)

# 3. Load Model

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)

# 4. Define Accuracy Metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. Training Config
training_args = TrainingArguments(
    output_dir="./log",
    remove_unused_columns=False, 
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(), # Leverage your GPU
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"], 
    processing_class=image_processor, # Renamed from 'tokenizer' for clarity
    compute_metrics=compute_metrics,  # Added this!
    data_collator=DefaultDataCollator(),
)

# 7. GO!
trainer.train()

# --- 6. Final Test Evaluation ---
print("\n--- Evaluating on Test Set ---")
test_metrics = trainer.evaluate(dataset["test"])
print(f"Final Test Accuracy: {test_metrics['eval_accuracy'] * 100:.2f}%")

# --- 7. Generate Predictions for CSV ---
print("Generating predictions for CSV...")
# This gives us the raw logits for the test set
output = trainer.predict(dataset["test"])
preds = np.argmax(output.predictions, axis=-1)

# Map predicted IDs back to class names (0-100)
predicted_labels = [id2label[p] for p in preds]

# Get original ground truth labels (if you want to compare in the CSV)
true_labels = [id2label[l] for l in output.label_ids]

# Create the DataFrame
df = pd.DataFrame({
    "image_index": range(len(preds)),
    "true_label": true_labels,
    "predicted_label": predicted_labels,
    "is_correct": np.array(true_labels) == np.array(predicted_labels)
})

# Save to CSV
csv_filename = "test_predictions.csv"
df.to_csv(path_or_buf= "/log", name=csv_filename, index=False)
print(f"Predictions saved to {csv_filename}")
