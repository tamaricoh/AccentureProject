# LLM Project for Accenture

## Environment Setup

The project was developed in a **Conda environment** using Python 3.12. All the necessary libraries and dependencies were installed within this environment to ensure smooth execution and reproducibility.

Detailed instructions for setting up and running the project are provided in the associated Jupyter notebook (`.ipynb`). This notebook outlines the steps to:

- Perform K-fold cross-validation for model training and evaluation.
- Train a BERT-based model for sequence classification.
- Evaluate the model's performance using F1 Score and accuracy metrics.
- Save the trained model for future use.
- Save the test data to a CSV file for comparison with baseline models.

**Note:**  
A significant portion of the code is commented out because it was part of the implementation before the addition of K-fold cross-validation.

## Model Tracking

Each trained model is saved with a timestamped filename (e.g., `model_YYY-MM-DD.pth`) to keep track of different versions. The performance of the model is evaluated after training, and the following metrics are recorded:

- **F1 Score (Weighted)**
- **Accuracy**

The model `model_2024-11-26_21-46-02.pth` achieved the following results on the test set:

- **F1 Score (Weighted): 0.3244**
- **Accuracy: 41.18%**

## Requirements

- Conda environment (Python 3.12)
- Necessary Python packages (listed in the notebook)
- Dataset file (`dataset.csv`)
