import pandas as pd

dataset_df = pd.read_csv('../csv/dataset.csv')
baseline_df = pd.read_csv('../csv/baseline.csv')
# dataset_aug_df = pd.read_csv('../csv/dataset_aug.csv')
dataset_aug_df = pd.read_csv('../csv/dataset_aug_less.csv')
test_df = pd.read_csv('../csv/cve_data.csv')

baseline_descriptions = baseline_df['Example Description'].tolist()

filtered_dataset_df = dataset_df[~dataset_df['Example Description'].isin(
    baseline_descriptions)]

train_df = pd.concat([filtered_dataset_df, dataset_aug_df], ignore_index=True)
val_df = baseline_df.copy()
test_df = test_df.copy()

train_labels = train_df['Artifact Id']
val_labels = val_df['Artifact Id']
test_labels = test_df['Artifact Id']

train_labels_counts = train_labels.value_counts()
val_labels_counts = val_labels.value_counts()
test_labels_counts = test_labels.value_counts()

all_unique_labels = list(set(train_labels.unique().tolist(
) + val_labels.unique().tolist() + test_labels.unique().tolist()))

label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
