import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../csv/dataset.csv')

labels = df['Artifact Id']

label_counts = labels.value_counts()

filtered_labels_at_least_5 = label_counts[label_counts >= 5]

filtered_labels_at_least_5_list = filtered_labels_at_least_5.index.tolist()

filtered_df = df[df['Artifact Id'].isin(filtered_labels_at_least_5_list)]

filtered_labels_balance = label_counts[(
    label_counts >= 5) & (label_counts <= 200)]

filtered_labels_balance_list = filtered_labels_balance.index.tolist()

filtered_balance_df = df[df['Artifact Id'].isin(
    filtered_labels_balance_list)]

command_df = df[df['Artifact Id'] == 'd3f:Command']
sample_size = min(len(command_df), 16)
sampled_command_df = command_df.sample(n=sample_size, random_state=42)
combined_df = pd.concat([filtered_balance_df, sampled_command_df])

combined_df.reset_index(drop=True, inplace=True)

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
