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
    """使用指定配置训练LoRA"""
    print("\n" + "="*60)
    print(f"实验: {experiment_name}")
    print(f"LoRA rank={rank}, alpha={alpha}, epochs={epochs}")
    print("="*60)
    
    start_time = time.time()
    
    # 加载数据
    dataset = load_dataset("lmassaron/FinancialPhraseBank")
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(eval_dataset)}")
    
    # 加载模型
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", 
                        truncation=True, max_length=128)
    
    print("预处理数据...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3
    )
    
    # 配置LoRA
    print(f"配置 LoRA (r={rank}, alpha={alpha})...")
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
    
    # 训练配置
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
    
    # 训练
    print(f"\n开始训练 ({epochs} epochs)...")
    trainer.train()
    
    # 评估
    print("评估模型...")
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
    print(f"✓ {experiment_name} 完成!")
    print(f"  训练时间: {training_time/60:.2f} 分钟")
    print(f"  准确率: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1分数: {eval_results['eval_f1']:.4f}")
    print("-"*60)
    
    return results

def main():
    print("\n" + "="*60)
    print("LoRA 消融实验 - 测试不同rank和epochs的影响")
    print("="*60)
    
    all_results = []
    
    # 实验组合
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
        
        # 显示当前总进度
        elapsed = (time.time() - total_start) / 60
        print(f"\n已用时间: {elapsed:.1f} 分钟\n")
    
    # 保存汇总
    import json
    with open("./results/lora_ablation_summary.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    total_time = (time.time() - total_start) / 60
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("实验汇总")
    print("="*60)
    print(f"\n{'实验':<15} {'Rank':<6} {'Epochs':<7} {'准确率':<10} {'F1分数':<10} {'时间(分钟)':<12}")
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
    print(f"总训练时间: {total_time:.2f} 分钟")
    print("="*60)

if __name__ == "__main__":
    main()
