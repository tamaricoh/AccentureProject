# LLM Project for Accenture

This project implements a **multi-class classification** model using **BERT** for sequence classification. The model is designed to handle **imbalanced data** and evaluate performance using standard metrics like **F1 Score** and **Accuracy**.

The model is built using the following class definition:

```python
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
```

## Model Tracking

Each trained model is saved with a timestamped filename (e.g., `model_YYY-MM-DD.pth`) to keep track of different versions. The performance of the model is evaluated after training, and the following metrics are recorded:

- **F1 Score (Weighted)**
- **Accuracy**

For the model trained on **2024-12-16_12-53-34**, the performance on the baseline dataset is:

- **F1 Score (Weighted):** 0.4080
- **Accuracy:** 43.75%

For the model trained on **2024-12-16_14-53-35**, the performance on the baseline dataset is:

- **F1 Score (Weighted):** 0.5568
- **Accuracy:** 62.50%

For comparison, the baseline performance is:

- **Baseline Accuracy:** 18.75%
- **Baseline F1 Score:** 0.2083

## Environment Setup

The project was developed in a **Conda environment** using Python 3.12. All the necessary libraries and dependencies were installed within this environment to ensure smooth execution and reproducibility.

Detailed instructions for setting up and running the project are provided in the associated Jupyter notebook (`.ipynb`).

**Note:**  
A significant portion of the code is commented out because it was part of an earlier implementation that included K-fold cross-validation.

## Requirements

- Conda environment (Python 3.12)
- Necessary Python packages (listed in the notebook)
- Dataset file (`dataset.csv`)
