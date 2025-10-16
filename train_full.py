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
from utils import compute_metrics, count_parameters, save_results

def main():
    print("="*50)
    print("Starting Full Fine-tuning")
    print("="*50)
    
    start_time = time.time()
    
    # 1. 加载数据集
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    print(f"✓ Dataset loaded successfully!")
    
    # 使用验证集作为评估集
    train_dataset = dataset['train']
    eval_dataset = dataset['validation'] if 'validation' in dataset else dataset['test']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 2. 加载tokenizer和模型
    print("\n[2/5] Loading model and tokenizer...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 数据预处理
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # 加载模型（Full Fine-tuning）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    param_stats = count_parameters(model)
    print(f"\nTotal parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,}")
    
    # 3. 训练配置
    print("\n[3/5] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results/full",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs/full",
        logging_steps=50,
        eval_strategy="epoch",  # 修复：evaluation_strategy -> eval_strategy
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
    
    # 4. 训练
    print("\n[4/5] Training...")
    train_result = trainer.train()
    
    # 5. 评估
    print("\n[5/5] Evaluating...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    final_results = {
        "model": "DistilBERT Full Fine-tuning",
        "dataset": "lmassaron/FinancialPhraseBank",
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
    }
    
    os.makedirs("./results/full", exist_ok=True)
    save_results(final_results, "./results/full/metrics.json")
    trainer.save_model("./results/full/best_model")
    
    print("\n" + "="*50)
    print("Full Fine-tuning Completed!")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
