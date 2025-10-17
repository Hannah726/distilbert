import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from utils import compute_metrics, count_parameters, save_results

def main():
    print("="*50)
    print("Starting LoRA Fine-tuning")
    print("="*50)
    
    start_time = time.time()
    
    # 1. Load dataset
    print("\n[1/6] Loading dataset...")
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    print(f"✓ Dataset loaded successfully!")
    print(dataset)
    
    print("\nData sample:")
    print(dataset['train'][0])
    
    # Use validation set as evaluation set
    train_dataset = dataset['train']
    eval_dataset = dataset['validation'] if 'validation' in dataset else dataset['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 2. Load tokenizer and model
    print("\n[2/6] Loading model and tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Data preprocessing
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # 3. Configure LoRA
    print("\n[3/6] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    param_stats = count_parameters(model)
    print(f"\nTotal parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,}")
    print(f"Trainable %: {param_stats['trainable_percent']:.2f}%")
    
    # 4. Training setup
    print("\n[4/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results/lora",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs/lora",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        report_to="none",
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 5. Training
    print("\n[5/6] Training...")
    train_result = trainer.train()
    
    # 6. Evaluation
    print("\n[6/6] Evaluating...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    final_results = {
        "model": "DistilBERT + LoRA",
        "dataset": "lmassaron/FinancialPhraseBank",
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
    }
    
    os.makedirs("./results/lora", exist_ok=True)
    save_results(final_results, "./results/lora/metrics.json")
    trainer.save_model("./results/lora/best_model")
    
    print("\n" + "="*50)
    print("LoRA Training Completed!")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
