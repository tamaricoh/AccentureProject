import pandas as pd

train_df = pd.read_csv('../csv/dataset.csv')
train_labels = train_df['Artifact Id']
label_counts = train_labels.value_counts()
filtered_labels = label_counts[(label_counts >= 5) & (label_counts <= 200)]
filtered_labels_list = filtered_labels.index.tolist()
train_df = train_df[train_df['Artifact Id'].isin(filtered_labels_list)]
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv('../csv/cve_data.csv')
test_labels = test_df['Artifact Id']
test_labels_counts = test_labels.value_counts()
unique_train_labels = train_labels.unique().tolist()
unique_test_labels = test_labels.unique().tolist()
all_unique_labels = list(
    set([str(label) for label in unique_train_labels + unique_test_labels]))
label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}

test_df.reset_index(drop=True, inplace=True)
print(label_mapping)
