import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback

# Load dataset
file_path = "training_data0320.csv"
df = pd.read_csv(file_path)

# Check for missing values and drop them
df.dropna(subset=['title', 'ab', 'newKey|1'], inplace=True)

# Define dataset class
class SciBERTDataset(Dataset):
    def __init__(self, titles, abstracts, labels, tokenizer, max_length=512):
        self.titles = titles
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        text = self.titles[idx] + " " + self.abstracts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Encode labels
df['newKey|1'] = df['newKey|1'].astype('category').cat.codes

# Train-test split
train_titles, val_titles, train_abs, val_abs, train_labels, val_labels = train_test_split(
    df['title'].tolist(), df['ab'].tolist(), df['newKey|1'].tolist(), test_size=0.2, random_state=42)

# Create dataset instances
train_dataset = SciBERTDataset(train_titles, train_abs, train_labels, tokenizer)
val_dataset = SciBERTDataset(val_titles, val_abs, val_labels, tokenizer)

# Load SciBERT model
model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=df['newKey|1'].nunique())

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=5,
)

# Initialize logging dictionary
history_loss = {"epoch": [], "loss": []}
history = {"epoch": [None], "loss": [None], "accuracy": [None], "f1": [None], "precision": [None], "recall": [None]}

def custom_accuracy(y_true, y_pred):
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    total = len(y_true)
    return correct / total if total > 0 else 0

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    # print(labels,  preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = custom_accuracy(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Callback function to record training progress
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            history_loss["epoch"].append(state.epoch)
            history_loss["loss"].append(logs["loss"])
        if "eval_loss" in logs:
            history["loss"].append(logs.get("eval_loss", 0))
            history["epoch"].append(logs.get("epoch", 0))
            history["accuracy"].append(logs.get("eval_accuracy", 0))
            history["f1"].append(logs.get("eval_f1", 0))
            history["precision"].append(logs.get("eval_precision", 0))
            history["recall"].append(logs.get("eval_recall", 0))

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[LoggingCallback()],
)

# Evaluate before training
eval_results = trainer.evaluate()
# history["loss"].append(eval_results.get("eval_loss", None))
# history["accuracy"].append(eval_results.get("eval_accuracy", None))
# history["f1"].append(eval_results.get("eval_f1", None))
# history["precision"].append(eval_results.get("eval_precision", None))
# history["recall"].append(eval_results.get("eval_recall", None))

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./scibert_model")
tokenizer.save_pretrained("./scibert_model")

# Evaluate on validation set
# eval_results = trainer.evaluate()
# print("Evaluation results:", eval_results)

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False)

history_loss_df = pd.DataFrame(history_loss)
history_loss_df.to_csv("training_history_loss.csv", index=False)

# Plot training metrics
plt.figure()
plt.plot(history_loss_df["epoch"], history_loss_df["loss"], label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs. Epoch")
plt.savefig("loss_vs_epoch.png")

plt.figure()
plt.plot(history["epoch"], history["accuracy"], label="Accuracy")
plt.plot(history["epoch"], history["f1"], label="F1-Score")
plt.plot(history["epoch"], history["precision"], label="Precision")
plt.plot(history["epoch"], history["recall"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Evaluation Metrics vs. Epoch")
plt.savefig("metrics_vs_epoch.png")