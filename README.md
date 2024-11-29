# LLM Project for Accenture

This project implements a **multi-class classification** model using **BERT** for sequence classification. The model is designed to handle **imbalanced data** and is trained using techniques like **K-fold cross-validation** for robust evaluation.

## Environment Setup

The project was developed in a **Conda environment** using Python 3.12. All the necessary libraries and dependencies were installed within this environment to ensure smooth execution and reproducibility.

Detailed instructions for setting up and running the project are provided in the associated Jupyter notebook (`.ipynb`).

**Note:**  
A significant portion of the code is commented out because it was part of the implementation before the addition of K-fold cross-validation.

## Model Tracking

Each trained model is saved with a timestamped filename (e.g., `model_YYY-MM-DD.pth`) to keep track of different versions. The performance of the model is evaluated after training, and the following metrics are recorded:

- **F1 Score (Weighted)**
- **Accuracy**

The model `model_2024-11-29_11-36-12.pth` achieved the following results on the test set:

- **F1 Score (Weighted):** 0.4048
- **Accuracy:** 50.00%

## Requirements

- Conda environment (Python 3.12)
- Necessary Python packages (listed in the notebook)
- Dataset file (`dataset.csv`)
