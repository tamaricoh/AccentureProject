import pandas as pd

df = pd.read_csv('../csv/dataset.csv')

labels = df['Artifact Id']
label_counts = labels.value_counts()

filtered_labels_at_least_5 = label_counts[label_counts >= 5]
filtered_labels_at_least_5_list = filtered_labels_at_least_5.index.tolist()

filtered_df = df[df['Artifact Id'].isin(filtered_labels_at_least_5_list)]

filtered_labels_balance = label_counts[(
    label_counts >= 5) & (label_counts <= 200)]
filtered_labels_balance_list = filtered_labels_balance.index.tolist()

filtered_balance_df = df[df['Artifact Id'].isin(filtered_labels_balance_list)]

command_df = df[df['Artifact Id'] == 'd3f:Command']
sample_size = min(len(command_df), 16)
sampled_command_df = command_df.sample(n=sample_size, random_state=42)

train_df = pd.concat([filtered_balance_df, sampled_command_df])
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv('../csv/cve_data.csv')

test_labels = test_df['Artifact Id']
test_labels_counts = test_labels.value_counts()
test_labels_list = test_labels_counts.tolist()

unique_train_labels = filtered_labels_balance.unique().tolist()
unique_test_labels = test_labels.unique().tolist()

all_unique_labels = list(set(unique_train_labels + unique_test_labels))

label_mapping = {label: idx for idx, label in enumerate(all_unique_labels)}

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
