Here is the updated text as per your request:

---

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

### Model Performance:

- **Baseline Model:**

  - **F1 Score (Weighted):** 0.1667
  - **Accuracy:** 18.75%

- **Model with Weighted Average over the Same Set (model_avr):**

  - **F1 Score (Weighted):** 0.1556
  - **Accuracy:** 25.00%

- **Model with Weighted Average over CVE Data (model_avr):**
  - Predictions saved to `cve_data_predictions.csv`
  - **Accuracy:** 52.78%
  - **F1 Score (Weighted):** 0.38

**Note:** The performance on the CVE data appears suspicious and might require further validation.

## Requirements

- Conda environment (Python 3.12)
- Necessary Python packages (listed in the notebook)
- Dataset file (`dataset.csv`)

---
