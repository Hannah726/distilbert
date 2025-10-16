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

def train_model(config_name, use_lora=False, lora_rank=None):
    """统一的训练函数"""
    print("\n" + "="*70)
    print(f"Training: {config_name}")
    print("="*70)
    
    start_time = time.time()
    
    # 加载数据
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    # 加载模型
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # 配置LoRA（如果需要）
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    param_stats = count_parameters(model)
    
    # 统一训练配置
    output_dir = f"./results/{config_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,  # 统一20 epochs
        per_device_train_batch_size=4,  # 小batch延长时间
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=2e-4 if use_lora else 2e-5,
        logging_dir=f"./logs/{config_name}",
        logging_steps=50,
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
    
    # 训练
    print(f"\n开始训练 (20 epochs, batch_size=4)...")
    trainer.train()
    
    # 评估
    eval_results = trainer.evaluate()
    training_time = time.time() - start_time
    
    results = {
        "experiment": config_name,
        "model": "DistilBERT + LoRA" if use_lora else "DistilBERT Full",
        "lora_config": {"r": lora_rank} if use_lora else None,
        "parameters": param_stats,
        "metrics": eval_results,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "epochs": 20
    }
    
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, f"{output_dir}/metrics.json")
    trainer.save_model(f"{output_dir}/best_model")
    
    print(f"\n✓ {config_name} 完成!")
    print(f"  时间: {training_time/60:.2f} 分钟")
    print(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1: {eval_results['eval_f1']:.4f}")
    
    return results

def main():
    """运行完整的标准化实验"""
    print("\n" + "="*70)
    print("Final Standardized Experiments")
    print("Configuration: 20 epochs, batch_size=4, DistilBERT")
    print("="*70)
    
    total_start = time.time()
    all_results = []
    
    # 实验配置
    experiments = [
        ("full_final", False, None),
        ("lora_r8_final", True, 8),
        ("lora_r16_final", True, 16),
        ("lora_r32_final", True, 32),
    ]
    
    for i, (name, use_lora, rank) in enumerate(experiments, 1):
        print(f"\n{'>'*70}")
        print(f"实验 {i}/{len(experiments)}")
        print(f"{'>'*70}")
        
        result = train_model(name, use_lora, rank)
        all_results.append(result)
        
        elapsed = (time.time() - total_start) / 60
        print(f"\n已用时间: {elapsed:.1f} 分钟")
    
    # 保存汇总
    import json
    with open("./results/final_comparison_summary.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    total_time = (time.time() - total_start) / 60
    
    # 打印汇总
    print("\n" + "="*70)
    print("实验汇总")
    print("="*70)
    print(f"\n{'实验':<20} {'Rank':<8} {'准确率':<12} {'F1':<12} {'时间(分钟)':<12}")
    print("-"*70)
    for r in all_results:
        name = r['experiment']
        rank = r['lora_config']['r'] if r['lora_config'] else 'Full'
        acc = r['metrics']['eval_accuracy']
        f1 = r['metrics']['eval_f1']
        time_min = r['training_time_minutes']
        print(f"{name:<20} {str(rank):<8} {acc:<12.4f} {f1:<12.4f} {time_min:<12.2f}")
    
    print("-"*70)
    print(f"总时间: {total_time:.2f} 分钟 ({total_time/60:.2f} 小时)")
    print("="*70)

if __name__ == "__main__":
    main()
