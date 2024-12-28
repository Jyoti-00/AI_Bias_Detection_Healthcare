from preprocessing import load_data, preprocess_data
from training import train_model
from evaluation import evaluate_model
from explainability import explain_with_shap, explain_with_lime

def main():
    # Load and preprocess data
    data = load_data("data/balanced_BUG.csv")
    train_dataset, test_dataset = preprocess_data(data)

    # Train model
    model_name = "albert-base-v2"
    train_model(train_dataset, test_dataset, model_name=model_name)

    # Evaluate model
    # Add logic to load test labels and predictions
    y_true = [0, 1, 1, 0]  
    y_pred = [0, 1, 0, 0] 
    protected_attributes = {"group_a": [1, 1, 0, 0], "group_b": [0, 0, 1, 1]}
    evaluate_model(y_true, y_pred, protected_attributes)

    # Explain predictions
    texts = ["The doctor was biased.", "The treatment was fair."]
    explain_with_shap(None, None, texts)  # Please use your actual model and tokenizer
    explain_with_lime(None, None, texts)

if __name__ == "__main__":
    main()
