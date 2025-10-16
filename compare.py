import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*50)
    print("Generating Comparison Analysis")
    print("="*50)
    
    # 读取结果
    with open("./results/full/metrics.json", "r") as f:
        full_results = json.load(f)
    
    with open("./results/lora/metrics.json", "r") as f:
        lora_results = json.load(f)
    
    # 创建对比表格
    comparison_data = {
        "Method": ["Full Fine-tuning", "LoRA (r=8)"],
        "Accuracy": [
            full_results["metrics"]["eval_accuracy"],
            lora_results["metrics"]["eval_accuracy"]
        ],
        "F1 Score": [
            full_results["metrics"]["eval_f1"],
            lora_results["metrics"]["eval_f1"]
        ],
        "Precision": [
            full_results["metrics"]["eval_precision"],
            lora_results["metrics"]["eval_precision"]
        ],
        "Recall": [
            full_results["metrics"]["eval_recall"],
            lora_results["metrics"]["eval_recall"]
        ],
        "Trainable Params (M)": [
            full_results["parameters"]["trainable"] / 1e6,
            lora_results["parameters"]["trainable"] / 1e6
        ],
        "Training Time (min)": [
            full_results["training_time_minutes"],
            lora_results["training_time_minutes"]
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 保存为CSV
    df.to_csv("./results/comparison.csv", index=False)
    print("\nComparison table saved to ./results/comparison.csv")
    print("\n" + df.to_string(index=False))
    
    # 生成可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 性能对比
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x = range(len(df))
    width = 0.35
    
    ax1 = axes[0, 0]
    for i, metric in enumerate(metrics):
        ax1.bar([j + i*width/4 for j in x], df[metric], width/4, 
                label=metric, alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks([i + width/4 for i in x])
    ax1.set_xticklabels(df["Method"], rotation=15, ha='right')
    ax1.legend()
    ax1.set_ylim([0.8, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 参数量对比
    ax2 = axes[0, 1]
    bars = ax2.bar(df["Method"], df["Trainable Params (M)"], color=['#ff6b6b', '#4ecdc4'])
    ax2.set_ylabel('Parameters (Millions)')
    ax2.set_title('Trainable Parameters Comparison')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df["Method"], rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom')
    
    # 3. 训练时间对比
    ax3 = axes[1, 0]
    bars = ax3.bar(df["Method"], df["Training Time (min)"], color=['#ff6b6b', '#4ecdc4'])
    ax3.set_ylabel('Time (minutes)')
    ax3.set_title('Training Time Comparison')
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df["Method"], rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}min', ha='center', va='bottom')
    
    # 4. 效率得分（F1/参数量）
    ax4 = axes[1, 1]
    efficiency = df["F1 Score"] / (df["Trainable Params (M)"] / 100)
    bars = ax4.bar(df["Method"], efficiency, color=['#ff6b6b', '#4ecdc4'])
    ax4.set_ylabel('Efficiency Score')
    ax4.set_title('Parameter Efficiency (F1 per 100M params)')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels(df["Method"], rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("./plots/comparison.png", dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to ./plots/comparison.png")
    
    # 生成总结
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("="*50)
    
    acc_diff = (lora_results["metrics"]["eval_accuracy"] - 
                full_results["metrics"]["eval_accuracy"]) * 100
    param_reduction = (1 - lora_results["parameters"]["trainable"] / 
                       full_results["parameters"]["trainable"]) * 100
    time_reduction = (1 - lora_results["training_time_minutes"] / 
                      full_results["training_time_minutes"]) * 100
    
    print(f"\n1. Performance:")
    print(f"   - LoRA accuracy: {acc_diff:+.2f}% vs Full Fine-tuning")
    print(f"   - F1 Score difference: {(lora_results['metrics']['eval_f1'] - full_results['metrics']['eval_f1'])*100:+.2f}%")
    
    print(f"\n2. Efficiency:")
    print(f"   - Parameters reduced: {param_reduction:.1f}%")
    print(f"   - Training time reduced: {time_reduction:.1f}%")
    
    print(f"\n3. Conclusion:")
    if abs(acc_diff) < 2 and param_reduction > 90:
        print(f"   ✓ LoRA achieves comparable performance with {param_reduction:.0f}% fewer parameters!")
    elif acc_diff < -2:
        print(f"   ! Full fine-tuning outperforms LoRA by {abs(acc_diff):.1f}%")
    else:
        print(f"   ✓ Both methods achieve strong performance on financial sentiment")
    
    print("="*50)

if __name__ == "__main__":
    main()
