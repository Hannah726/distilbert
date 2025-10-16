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
    print("Starting Advanced LoRA Fine-tuning")
    print("="*50)
    
    start_time = time.time()
    
    # 1. 加载数据集
    print("\n[1/6] Loading dataset...")
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # 2. 使用更大的模型：BERT-base 而不是 DistilBERT
    print("\n[2/6] Loading BERT-base model...")
    model_name = "bert-base-uncased"  # 更大：110M vs 67M
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
    
    # 3. 配置更复杂的LoRA
    print("\n[3/6] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # 从8增加到16
        lora_alpha=32,  # 从16增加到32
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"]  # 更多层
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    param_stats = count_parameters(model)
    
    # 4. 更长的训练配置
    print("\n[4/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./results/lora_advanced",
        num_train_epochs=10,  # 从3增加到10
        per_device_train_batch_size=8,  # 减小batch提高训练时间
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=2e-4,
        logging_dir="./logs/lora_advanced",
        logging_steps=20,
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
    
    # 5. 训练
    print("\n[5/6] Training...")
    train_result = trainer.train()
    
    # 6. 评估
    print("\n[6/6] Evaluating...")
    eval_results = trainer.evaluate()
    
    training_time = time.time() - start_time
    
    final_results = {
        "model": "BERT-base + LoRA (r=16)",
        "dataset": "lmassaron/FinancialPhraseBank",
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
    }
    
    os.makedirs("./results/lora_advanced", exist_ok=True)
    save_results(final_results, "./results/lora_advanced/metrics.json")
    trainer.save_model("./results/lora_advanced/best_model")
    
    print("\n" + "="*50)
    print("Advanced LoRA Training Completed!")
    print(f"Time: {training_time/60:.2f} minutes")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
