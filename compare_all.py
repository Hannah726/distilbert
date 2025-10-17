import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """load all the results"""
    results = {}

    try:
        with open("results/lora/metrics.json") as f:
            results['LoRA (3 epochs)'] = json.load(f)
    except: pass
    
    try:
        with open("results/full/metrics.json") as f:
            results['Full (3 epochs)'] = json.load(f)
    except: pass
    
    # Extended实验
    try:
        with open("results/lora_extended/metrics.json") as f:
            results['LoRA Extended (15 epochs)'] = json.load(f)
    except: pass
    
    try:
        with open("results/full_extended/metrics.json") as f:
            results['Full Extended (15 epochs)'] = json.load(f)
    except: pass
    
    return results

def main():
    print("="*60)
    print("综合对比分析")
    print("="*60)
    
    results = load_results()
    
    if not results:
        print("✗ 没有找到结果文件")
        return

    data = []
    for name, result in results.items():
        data.append({
            'Method': name,
            'Accuracy': result['metrics']['eval_accuracy'],
            'F1 Score': result['metrics']['eval_f1'],
            'Trainable Params (M)': result['parameters']['trainable'] / 1e6,
            'Training Time (min)': result['training_time_minutes']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Accuracy', ascending=False)
    
    df.to_csv('results/comparison_all.csv', index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n✓ 已保存到: results/comparison_all.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df['Accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar([i + width/2 for i in x], df['F1 Score'], width, label='F1 Score', alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    bars = ax.barh(df['Method'], df['Training Time (min)'])
    ax.set_xlabel('Time (minutes)')
    ax.set_title('Training Time Comparison')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}min', ha='left', va='center')
        
    ax = axes[1, 0]
    efficiency = df['F1 Score'] / (df['Trainable Params (M)'] / 10)
    bars = ax.bar(range(len(df)), efficiency)
    ax.set_ylabel('Efficiency Score')
    ax.set_title('Parameter Efficiency (F1 per 10M params)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Method'], rotation=45, ha='right')
    
    ax = axes[1, 1]
    colors = ['#ff6b6b', '#4ecdc4', '#ff6b6b', '#4ecdc4']
    ax.scatter(df['Training Time (min)'], df['Accuracy'], 
              s=200, c=colors[:len(df)], alpha=0.6)
    for i, row in df.iterrows():
        ax.annotate(row['Method'].split('(')[0].strip(), 
                   (row['Training Time (min)'], row['Accuracy']),
                   fontsize=8, ha='center')
    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Training Time')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/comparison_all.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存图表到: plots/comparison_all.png")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
