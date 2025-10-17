# Evaluate the trained model
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from utils import compute_metrics, plot_confusion_matrix
import numpy as np
import json
import os

def evaluate_model(model_path, model_type="full"):
    """
    Evaluate the model.
    Args:
        model_path: Path to the model.
        model_type: "full" or "lora".
    """
    print(f"\n{'='*50}")
    print(f"Evaluating {model_type.upper()} model")
    print(f"{'='*50}\n")
    
    # Load data
    print("[1/4] Loading dataset...")
    try:
        dataset = load_dataset('financial_phrasebank', 'sentences_allagree')
        dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        test_dataset = dataset["test"]
    except:
        # If HuggingFace fails, fall back to local CSV
        import pandas as pd
        df = pd.read_csv('data/financial_phrasebank.csv')
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=0.2, random_state=42)
        test_dataset = test_df
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load tokenizer
    print("\n[2/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load model
    if model_type == "lora":
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Inference
    print("\n[3/4] Making predictions...")
    predictions = []
    true_labels = []
    
    for example in test_dataset:
        if isinstance(example, dict):
            text = example['sentence']
            label = example['label']
        else:
            text = example.sentence
            label = example.label
            
        inputs = tokenizer(text, return_tensors="pt", 
                          padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            
        predictions.append(pred)
        true_labels.append(label)
    
    # Compute metrics
    print("\n[4/4] Computing metrics...")
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n" + classification_report(true_labels, predictions, 
                                       target_names=['Positive', 'Neutral', 'Negative']))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        true_labels, predictions,
        labels=['Positive', 'Neutral', 'Negative'],
        save_path=f'plots/confusion_matrix_{model_type}.png'
    )
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    os.makedirs(f'results/{model_type}', exist_ok=True)
    with open(f'results/{model_type}/eval_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to results/{model_type}/eval_results.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py [full|lora]")
        sys.exit(1)
    
    model_type = sys.argv[1]
    
    if model_type == "full":
        evaluate_model("./results/full/best_model", "full")
    elif model_type == "lora":
        evaluate_model("./results/lora/best_model", "lora")
    else:
        print("Invalid model type. Use 'full' or 'lora'")
