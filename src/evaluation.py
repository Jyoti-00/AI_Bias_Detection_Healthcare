from sklearn.metrics import classification_report
from holisticai.bias.metrics import classification_bias_metrics
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, protected_attributes, metric_type="both"):
    """Evaluate the model for fairness and performance."""
    # Performance Metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Fairness Metrics
    print("\nFairness Metrics:")
    bias_metrics = classification_bias_metrics(
        y_true=y_true,
        y_pred=y_pred,
        group_a=protected_attributes['group_a'],
        group_b=protected_attributes['group_b'],
        metric_type=metric_type
    )
    print(bias_metrics)

    # Visualise fairness metrics
    bias_metrics.plot(kind="bar", figsize=(10, 5), legend=False)
    plt.title("Fairness Metrics")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    # Example usage
    y_true = [0, 1, 1, 0]  # Replace with actual test labels
    y_pred = [0, 1, 0, 0]  # Replace with model predictions
    protected_attributes = {
        "group_a": [1, 1, 0, 0],  # e.g., Female
        "group_b": [0, 0, 1, 1]   # e.g., Male
    }
    evaluate_model(y_true, y_pred, protected_attributes)
