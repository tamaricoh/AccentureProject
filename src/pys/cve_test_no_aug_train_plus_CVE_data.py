import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../csv/dataset.csv')
test_df = pd.read_csv('../csv/cve_data.csv')

train_labels = train_df['Artifact Id']
label_counts = train_labels.value_counts()
filtered_labels = label_counts[(label_counts >= 5) & (label_counts <= 200)]
filtered_labels_list = filtered_labels.index.tolist()
train_df = train_df[train_df['Artifact Id'].isin(filtered_labels_list)]
train_df.reset_index(drop=True, inplace=True)

test_df.reset_index(drop=True, inplace=True)
test_sample_size = int(len(test_df) / 5)

test_sample, remaining_test = train_test_split(
    test_df,
    test_size=(1 - test_sample_size / len(test_df)),
    stratify=test_df['Artifact Id'],
    random_state=42
)

train_df = pd.concat([train_df, test_sample]).reset_index(drop=True)
test_df = remaining_test.reset_index(drop=True)

train_labels = train_df['Artifact Id']
test_labels = test_df['Artifact Id']
unique_train_labels = train_labels.unique().tolist()
unique_test_labels = test_labels.unique().tolist()
all_unique_labels = list(
    set([str(label) for label in unique_train_labels + unique_test_labels]))
label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}

print(label_mapping)

train_df.to_csv('../csv/updated_train_dataset.csv', index=False)
test_df.to_csv('../csv/updated_test_dataset.csv', index=False)
