import numpy as np
import pandas as pd
import torch
import datetime
from torch import nn
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_model_path = f"models/model_{timestamp}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 4
learning_rate = 5e-5
batch_size = 16


class CustomBertModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBertModel, self).__init__()
        # Use BertForSequenceClassification directly
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)

        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = outputs.loss

        return {"loss": loss, "logits": logits}


def tokenize_data(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)


def create_dataset(df, tokenizer, label_mapping):
    texts = df['Example Description'].tolist()
    encodings = tokenize_data(texts, tokenizer)

    # Default to -1 for unknown labels
    labels = torch.tensor([label_mapping.get(x, -1)
                          for x in df['Artifact Id']])

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return dataset


def train_with_validation(model, train_loader, val_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_preds = []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            input_ids, attention_mask, labels = [
                item.to(device) for item in batch]

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate training metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Validation phase
        model.eval()
        val_correct = 0
        val_samples = 0
        val_all_labels = []
        val_all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                input_ids, attention_mask, labels = [
                    item.to(device) for item in batch]

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                logits = outputs['logits']

                preds = torch.argmax(logits, dim=-1)

                val_all_preds.extend(preds.cpu().tolist())
                val_all_labels.extend(labels.cpu().tolist())

                correct = (preds == labels).sum().item()
                val_correct += correct
                val_samples += labels.size(0)

        val_accuracy = accuracy_score(val_all_labels, val_all_preds) * 100
        val_f1 = f1_score(val_all_labels, val_all_preds, average="weighted")

        print(
            f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}, Accuracy: {accuracy:.6f}%, F1 Score: {f1:.6f}")
        print(
            f"Validation Accuracy: {val_accuracy:.6f}%, Validation F1 Score: {val_f1:.6f}")


def test(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids, attention_mask, labels = [
                item.to(device) for item in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate F1 score (weighted) for multiclass classification
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100

    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    return predictions, true_labels


df = pd.read_csv('csv/dataset.csv')

labels = df['Artifact Id']

label_counts = labels.value_counts()

filtered_labels_at_least_5 = label_counts[label_counts >= 5]

filtered_labels_at_least_5_list = filtered_labels_at_least_5.index.tolist()

filtered_df = df[df['Artifact Id'].isin(filtered_labels_at_least_5_list)]

print(filtered_df['Artifact Id'].value_counts())

filtered_labels_balance = label_counts[(
    label_counts >= 5) & (label_counts <= 200)]

filtered_labels_balance_list = filtered_labels_balance.index.tolist()

filtered_balance_df = df[df['Artifact Id'].isin(filtered_labels_balance_list)]

command_df = df[df['Artifact Id'] == 'd3f:Command']
sample_size = min(len(command_df), 16)
sampled_command_df = command_df.sample(n=sample_size, random_state=42)
combined_df = pd.concat([filtered_balance_df, sampled_command_df])

combined_df.reset_index(drop=True, inplace=True)

print(combined_df['Artifact Id'].value_counts())

label_mapping = {label: idx for idx, label in enumerate(
    filtered_labels_at_least_5_list)}

train_val_df, test_df = train_test_split(combined_df,
                                         test_size=0.2,
                                         stratify=combined_df['Artifact Id'],
                                         random_state=42)
train_df, val_df = train_test_split(train_val_df,
                                    test_size=0.2,
                                    stratify=train_val_df['Artifact Id'],
                                    random_state=42)

model = CustomBertModel(num_labels=len(filtered_labels_at_least_5_list))
optimizer = AdamW(model.parameters(), lr=learning_rate)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.to(device)


def main():

    train_dataset = create_dataset(train_df, tokenizer, label_mapping)
    val_dataset = create_dataset(val_df, tokenizer, label_mapping)
    test_dataset = create_dataset(test_df, tokenizer, label_mapping)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(
        train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(
        val_dataset), batch_size=batch_size)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(
        test_dataset), batch_size=batch_size)

    train_with_validation(model, train_loader, val_loader,
                          optimizer, device, num_epochs)
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
