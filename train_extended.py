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

def train_lora_extended():
    """训练 LoRA Extended (15 epochs)"""
    print("\n" + "="*60)
    print("Training LoRA Extended (15 epochs)")
    print("="*60)
    
    start_time = time.time()
    
    # data loading
    print("\n[1/6] Loading dataset...")
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # model loading
    print("\n[2/6] Loading model and tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # set for LoRA
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
    
    # train settings
    print("\n[4/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results/lora_extended",
        num_train_epochs=15,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=32,
        warmup_steps=150,
        weight_decay=0.01,
        learning_rate=3e-4,
        logging_dir="./logs/lora_extended",
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
    
    # train
    print("\n[5/6] Training...")
    trainer.train()
    
    # eva
    print("\n[6/6] Evaluating...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    results = {
        "model": "DistilBERT + LoRA (Extended)",
        "dataset": "lmassaron/FinancialPhraseBank",
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "epochs": 15,
        "batch_size": 8
    }
    
    os.makedirs("./results/lora_extended", exist_ok=True)
    save_results(results, "./results/lora_extended/metrics.json")
    trainer.save_model("./results/lora_extended/best_model")
    
    print("\n" + "="*60)
    print("LoRA Extended Training Completed!")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print("="*60)
    
    return results

def train_full_extended():
    """训练 Full Fine-tuning Extended (15 epochs)"""
    print("\n" + "="*60)
    print("Training Full Fine-tuning Extended (15 epochs)")
    print("="*60)
    
    start_time = time.time()
    
    # data loading
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # model loading
    print("\n[2/5] Loading model and tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    param_stats = count_parameters(model)
    print(f"\nTotal parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,}")
    
    # train settings
    print("\n[3/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results/full_extended",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_steps=150,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir="./logs/full_extended",
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
    
    # train
    print("\n[4/5] Training...")
    trainer.train()
    
    # eva
    print("\n[5/5] Evaluating...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    results = {
        "model": "DistilBERT Full Fine-tuning (Extended)",
        "dataset": "lmassaron/FinancialPhraseBank",
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "epochs": 15,
        "batch_size": 8
    }
    
    os.makedirs("./results/full_extended", exist_ok=True)
    save_results(results, "./results/full_extended/metrics.json")
    trainer.save_model("./results/full_extended/best_model")
    
    print("\n" + "="*60)
    print("Full Extended Training Completed!")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print("="*60)
    
    return results

def main():
    """主函数：依次训练两个模型"""
    print("\n" + "="*60)
    print("Extended Training Experiments")
    print("15 epochs, batch_size=8")
    print("="*60)
    
    total_start = time.time()
    
    # LoRA Extended
    print("\n" + ">"*60)
    print("EXPERIMENT 1/2: LoRA Extended")
    print(">"*60)
    lora_results = train_lora_extended()
    
    # Full Extended
    print("\n" + ">"*60)
    print("EXPERIMENT 2/2: Full Fine-tuning Extended")
    print(">"*60)
    full_results = train_full_extended()
    
    # conclusion
    total_time = (time.time() - total_start) / 60
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print(f"\n总训练时间: {total_time:.2f} 分钟")
    print("\n结果对比:")
    print("-"*60)
    print(f"{'方法':<25} {'准确率':<12} {'F1分数':<12} {'时间(分钟)':<12}")
    print("-"*60)
    print(f"{'LoRA Extended':<25} {lora_results['metrics']['eval_accuracy']:<12.4f} "
          f"{lora_results['metrics']['eval_f1']:<12.4f} {lora_results['training_time_minutes']:<12.2f}")
    print(f"{'Full Extended':<25} {full_results['metrics']['eval_accuracy']:<12.4f} "
          f"{full_results['metrics']['eval_f1']:<12.4f} {full_results['training_time_minutes']:<12.2f}")
    print("-"*60)
    
    # save
    import json
    summary = {
        "lora_extended": lora_results,
        "full_extended": full_results,
        "total_time_minutes": total_time
    }
    with open("./results/extended_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n✓ 综合结果已保存到: ./results/extended_summary.json")
    print("="*60)

if __name__ == "__main__":
    main()
