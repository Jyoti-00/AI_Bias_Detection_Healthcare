import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def tokenize_function(example, tokenizer):
    """Tokenize the input text using a Hugging Face tokenizer."""
    return tokenizer(example['sentence_text'], padding="max_length", truncation=True)

def train_model(train_dataset, test_dataset, model_name="albert-base-v2", output_dir="models"):
    """Train a transformer model and evaluate its performance."""
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    train_dataset = train_dataset.remove_columns(['sentence_text', 'stereotype'])
    test_dataset = test_dataset.remove_columns(['sentence_text', 'stereotype'])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="logs",
        logging_steps=50
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    print("Training complete. Evaluation metrics:", metrics)

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    from preprocessing import preprocess_data, load_data
    data = load_data("data/balanced_BUG.csv")
    train_dataset, test_dataset = preprocess_data(data)
    train_model(train_dataset, test_dataset)
