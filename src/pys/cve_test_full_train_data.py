import pandas as pd

# Load training and test datasets
# train_df = pd.read_csv('../csv/merged_aug.csv')
train_df = pd.read_csv('../csv/merged_aug_less.csv')
test_df = pd.read_csv('../csv/cve_data.csv')

train_labels = train_df['Artifact Id']
test_labels = test_df['Artifact Id']

train_labels_counts = train_labels.value_counts()
test_labels_counts = test_labels.value_counts()

train_labels_list = train_labels_counts.tolist()
test_labels_list = test_labels_counts.tolist()

unique_train_labels = train_labels.unique().tolist()
unique_test_labels = test_labels.unique().tolist()

all_unique_labels = list(set(unique_train_labels + unique_test_labels))

label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
