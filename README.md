# AI Bias Detection in Healthcare

This project applies advanced artificial intelligence techniques to address biases in healthcare communication. Using fine-tuned transformer models, it identifies stereotypes in text data that relate to gender, age, and professions, ensuring equitable and unbiased communication within healthcare systems. By incorporating explainability tools and fairness metrics, the project not only detects biases but also explains the rationale behind AI decisions, making the results interpretable and actionable for stakeholders.

###### The core AI functionalities of the project are implemented in the `bias_detection.ipynb` file located in the `notebooks/` folder.

## Overview

Healthcare communication plays a critical role in patient outcomes and care delivery. However, it often reflects societal biases, which can inadvertently lead to inequitable treatment and decision-making. This project tackles these challenges by:
- **Detecting Biases**: Identifying and classifying stereotypes within healthcare-related text.
- **Quantifying Fairness**: Using fairness metrics to assess bias levels in predictions.
- **Explaining Model Predictions**: Employing SHAP and LIME to highlight influential features in biased decisions.
- **Ensuring Sustainability**: Tracking carbon emissions and energy consumption during AI model training.
- **Dynamic Data Analysis**: Scraping and analysing healthcare-related text from NHS websites for real-world relevance.

This project demonstrates how AI can contribute to ethical decision-making by integrating fairness and explainability in stereotype detection, fostering transparency and trust.

## Key Features

- Fine-tuning state-of-the-art transformer models (ALBERT, TinyBERT) for stereotype classification.
- Computing fairness metrics like demographic parity and equal opportunity for protected attributes (e.g., gender, age).
- Explaining bias detection using SHAP and LIME to provide feature-level and instance-level interpretability.
- Automating healthcare data scraping from NHS websites for analysis.
- Monitoring environmental sustainability with CodeCarbon during model training.

## Technologies Used

- **Machine Learning Frameworks**: Hugging Face Transformers, PyTorch, Scikit-learn.
- **Fairness and Bias Tools**: Holistic AI Metrics.
- **Explainability**: SHAP (Shapley Additive Explanations), LIME (Local Interpretable Model-Agnostic Explanations).
- **Sustainability**: CodeCarbon for tracking emissions and energy consumption.
- **Data Handling**: Pandas for data manipulation and BeautifulSoup for web scraping.


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jyoti-00/AI_Bias_Detection_Healthcare.git
   cd AI_Bias_Detection_Healthcare

2. Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Place your dataset files in the data/ directory.

Usage

1. Run the main workflow:
python src/main.py
2. Explore the notebook for detailed implementation:
jupyter notebook notebooks/bias_detection.ipynb

Contribution

We welcome contributions to improve the project! Please fork the repository, create a branch, and submit a pull request.

