import numpy as np
import pandas as pd
import torch
import datetime
from torch import nn
from tqdm.notebook import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# , label_mapping
from pys.data import train_df, val_df, test_df, filtered_labels_at_least_5_list
from pys.params import learning_rate, num_epochs, batch_size
from torch.optim.lr_scheduler import ReduceLROnPlateau


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_model_path = f"../models/model_{timestamp}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(model, train_loader, optimizer, device, num_epochs, scheduler=None):
    f1_train = []
    acc_train = []
    loss_train = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_accuracy = accuracy_score(all_labels, all_preds) * 100
        train_f1 = f1_score(all_labels, all_preds, average="weighted")
        train_loss = total_loss / len(train_loader)

        acc_train.append(train_accuracy)
        f1_train.append(train_f1)
        loss_train.append(train_loss)

        if scheduler:
            scheduler.step()

    return f1_train, acc_train, loss_train


def train_with_validation(model, train_loader, val_loader, optimizer, device, num_epochs, scheduler=None):
    f1_train = []
    f1_val = []
    acc_train = []
    acc_val = []
    loss_train = []
    loss_val = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
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

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_accuracy = accuracy_score(all_labels, all_preds) * 100
        train_f1 = f1_score(all_labels, all_preds, average="weighted")
        train_loss = total_loss / len(train_loader)

        acc_train.append(train_accuracy)
        f1_train.append(train_f1)
        loss_train.append(train_loss)

        model.eval()
        val_total_loss = 0
        val_all_labels = []
        val_all_preds = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                input_ids, attention_mask, labels = [
                    item.to(device) for item in batch]

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']

                preds = torch.argmax(logits, dim=-1)

                val_all_preds.extend(preds.cpu().tolist())
                val_all_labels.extend(labels.cpu().tolist())

                val_total_loss += loss.item()

        val_accuracy = accuracy_score(val_all_labels, val_all_preds) * 100
        val_f1 = f1_score(
            val_all_labels, val_all_preds, average="weighted")
        val_loss = val_total_loss / len(val_loader)

        acc_val.append(val_accuracy)
        f1_val.append(val_f1)
        loss_val.append(val_loss)

        print(
            f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.6f}, Accuracy: {train_accuracy:.6f}%, F1 Score: {train_f1:.6f}")
        print(
            f"Validation Accuracy: {val_accuracy:.6f}%, Validation F1 Score: {val_f1:.6f}")

        # if isinstance(scheduler, ReduceLROnPlateau):
        #     scheduler.step(val_loss)
        # else:
        #     scheduler.step()

    return f1_train, f1_val, acc_train, acc_val, loss_train, loss_val


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

    f1 = f1_score(true_labels, predictions, average='weighted')

    accuracy = accuracy_score(true_labels, predictions) * 100

    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    # return predictions, true_labels # Previous version before 31.12 changes
    return f1, accuracy


def main():

    model = CustomBertModel(num_labels=len(filtered_labels_at_least_5_list))
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.to(device)

    label_mapping = {label: idx for idx, label in enumerate(
        filtered_labels_at_least_5_list)}

    train_dataset = create_dataset(train_df, tokenizer, label_mapping)
    val_dataset = create_dataset(val_df, tokenizer, label_mapping)
    test_dataset = create_dataset(test_df, tokenizer, label_mapping)

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
