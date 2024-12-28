import shap
from lime.lime_text import LimeTextExplainer

def explain_with_shap(model, tokenizer, texts):
    """Generate SHAP explanations for predictions."""
    explainer = shap.Explainer(
        model,
        masker=shap.maskers.Text(tokenizer)
    )
    shap_values = explainer(texts)
    shap.plots.text(shap_values)

def explain_with_lime(model, tokenizer, texts):
    """Generate LIME explanations for predictions."""
    def model_wrapper(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.logits.softmax(dim=-1).detach().numpy()

    explainer = LimeTextExplainer()
    for text in texts:
        explanation = explainer.explain_instance(
            text, 
            model_wrapper, 
            num_features=10
        )
        explanation.show_in_notebook()

if __name__ == "__main__":
    # Example usage
    texts = ["The patient is treated well.", "Bias exists in certain medical professions."]
    # Replace with actual model and tokenizer
    model = None  
    tokenizer = None
    explain_with_shap(model, tokenizer, texts)
    explain_with_lime(model, tokenizer, texts)
