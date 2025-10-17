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

def train_lora_with_config(rank, alpha, epochs, experiment_name):
    """Train LoRA with specified configuration"""
    print("\n" + "="*60)
    print(f"Experiment: {experiment_name}")
    print(f"LoRA rank={rank}, alpha={alpha}, epochs={epochs}")
    print("="*60)
    
    start_time = time.time()
    
    # Load dataset
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    # Load model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Preprocessing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # Configure LoRA
    print(f"Configuring LoRA (r={rank}, alpha={alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    param_stats = count_parameters(model)
    
    # Training configuration
    output_dir = f"./results/{experiment_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=3e-4,
        logging_dir=f"./logs/{experiment_name}",
        logging_steps=30,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=42,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Training
    print(f"\nStarting training ({epochs} epochs)...")
    trainer.train()
    
    # Evaluation
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    results = {
        "experiment": experiment_name,
        "model": "DistilBERT + LoRA",
        "lora_config": {"r": rank, "alpha": alpha},
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "epochs": epochs
    }
    
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, f"{output_dir}/metrics.json")
    
    print("\n" + "-"*60)
    print(f"✓ {experiment_name} completed!")
    print(f"  Training time: {training_time/60:.2f} minutes")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
    print("-"*60)
    
    return results

def main():
    print("\n" + "="*60)
    print("LoRA Ablation Experiments - Testing Effects of Different Ranks and Epochs")
    print("="*60)
    
    all_results = []
    
    # Experiment combinations
    experiments = [
        # (rank, alpha, epochs, name)
        (4, 8, 8, "lora_r4_e8"),
        (8, 16, 8, "lora_r8_e8"),
        (16, 32, 8, "lora_r16_e8"),
        (32, 64, 6, "lora_r32_e6"),
    ]
    
    total_start = time.time()
    
    for rank, alpha, epochs, name in experiments:
        result = train_lora_with_config(rank, alpha, epochs, name)
        all_results.append(result)
        
        # Show current overall progress
        elapsed = (time.time() - total_start) / 60
        print(f"\nElapsed time: {elapsed:.1f} minutes\n")
    
    # Save summary
    import json
    with open("./results/lora_ablation_summary.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    total_time = (time.time() - total_start) / 60
    
    # Print summary table
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"\n{'Experiment':<15} {'Rank':<6} {'Epochs':<7} {'Accuracy':<10} {'F1 Score':<10} {'Time (min)':<12}")
    print("-"*60)
    for r in all_results:
        exp_name = r['experiment'].replace('lora_', '')
        rank = r['lora_config']['r']
        epochs = r['epochs']
        acc = r['metrics']['eval_accuracy']
        f1 = r['metrics']['eval_f1']
        time_min = r['training_time_minutes']
        print(f"{exp_name:<15} {rank:<6} {epochs:<7} {acc:<10.4f} {f1:<10.4f} {time_min:<12.2f}")
    
    print("-"*60)
    print(f"Total training time: {total_time:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()
