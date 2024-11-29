import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import datetime

num_epochs = 3
learning_rate = 1e-5
num_folds = 5  # Number of folds
batch_size = 16


def train_with_validation(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
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
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Calculate training metrics
        accuracy = total_correct / total_samples * 100
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Validation phase
        model.eval()
        val_loss = 0
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
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)

                val_all_preds.extend(preds.cpu().tolist())
                val_all_labels.extend(labels.cpu().tolist())

                correct = (preds == labels).sum().item()
                val_correct += correct
                val_samples += labels.size(0)

                val_loss += loss.item()

        val_accuracy = val_correct / val_samples * 100
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

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate F1 score (weighted) for multiclass classification
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Calculate accuracy
    accuracy = np.sum(np.array(predictions) == np.array(
        true_labels)) / len(true_labels) * 100

    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


df = pd.read_csv('dataset.csv')

labels = df['Artifact Id']

label_counts = labels.value_counts()

filtered_labels = label_counts[(label_counts >= 5) & (
    label_counts <= 200)]  # Remove "Command" label

filtered_labels_list = filtered_labels.index.tolist()

filtered_df = df[df['Artifact Id'].isin(filtered_labels_list)]

command_df = df[df['Artifact Id'] == 'd3f:Command']
sampled_command_df = command_df.sample(n=16, random_state=42)
combined_df = pd.concat([filtered_df, sampled_command_df])

combined_df.reset_index(drop=True, inplace=True)

print(combined_df['Artifact Id'].value_counts())

train_val_df, test_df = train_test_split(filtered_df,
                                         test_size=0.2,
                                         stratify=filtered_df['Artifact Id'],
                                         random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_encodings = tokenize_data(test_df['Example Description'].tolist())
test_labels = torch.tensor(test_df['Artifact Id'].map(
    lambda x: filtered_labels_list.index(x)).tolist())
test_dataset = TensorDataset(
    test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(
    test_dataset), batch_size=batch_size)

# K-Fold Cross Validation on 80% train/val data
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_results = []  # Store F1 score and accuracy for each fold

for fold, (train_index, val_index) in enumerate(skf.split(train_val_df, train_val_df['Artifact Id'])):
    print(f"\n--- Fold {fold+1}/{num_folds} ---")

    # Split the train/val set into training and validation data for this fold
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]

    # Tokenize the training and validation data
    train_encodings = tokenize_data(train_df['Example Description'].tolist())
    val_encodings = tokenize_data(val_df['Example Description'].tolist())

    # Map labels to indices for training and validation
    train_labels = torch.tensor(train_df['Artifact Id'].map(
        lambda x: filtered_labels_list.index(x)).tolist())
    val_labels = torch.tensor(val_df['Artifact Id'].map(
        lambda x: filtered_labels_list.index(x)).tolist())

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(
        train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(
        val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(
        train_dataset), batch_size=batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(
        val_dataset), batch_size=batch_size)

    # Initialize model, optimizer, and scheduler
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(filtered_labels_list))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Train the model with validation
    train_with_validation(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs
    )

# Average results over all folds
average_f1 = np.mean([result[0] for result in fold_results])
average_accuracy = np.mean([result[1] for result in fold_results])
print(f"\n--- K-Fold Results ---")
print(f"Average F1 Score (Weighted): {average_f1:.4f}")
print(f"Average Accuracy: {average_accuracy:.2f}%")

# Test the model on the test set
test(model, test_loader, device)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_model_path = f"model_{timestamp}.pth"

torch.save(model.state_dict(), output_model_path)
print(f"Model saved to {output_model_path}")

test_df.to_csv('test_data.csv', index=False)
print("DataFrame saved to test_data.csv")
