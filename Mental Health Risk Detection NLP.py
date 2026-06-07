import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

df=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\advance data science\activity\Final\archive\mental_health.csv").sample(
    1000, random_state=42)

X=df["text"].tolist()
y=df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TOKENIZATION
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=64)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self): return len(self.labels)

train_dataset = SentimentDataset(train_encodings, y_train)
val_dataset = SentimentDataset(val_encodings, y_test)

# 3. MODEL
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2,
    low_cpu_mem_usage=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    report_to="none")

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=SentimentDataset(train_encodings, y_train),
                  eval_dataset=SentimentDataset(val_encodings, y_test))
trainer.train()

#METRICS
raw_pred, _, _ = trainer.predict(SentimentDataset(val_encodings, y_test))
y_pred = np.argmax(raw_pred, axis=1)

print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))




