import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_final_results():
    """Load final standardized experiment results"""
    results = {}
    
    experiments = [
        'full_final',
        'lora_r8_final', 
        'lora_r16_final',
        'lora_r32_final'
    ]
    
    for exp in experiments:
        path = f'results/{exp}/metrics.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[exp] = json.load(f)
                print(f"✓ Loaded: {exp}")
    
    return results

def create_comparison_table(results):
    """Create detailed comparison table"""
    data = []
    
    for name, result in results.items():
        # Determine if it's LoRA
        is_lora = 'lora' in name.lower()
        rank = result['lora_config']['r'] if is_lora else 'Full'
        
        data.append({
            'Method': name.replace('_final', '').replace('_', ' ').title(),
            'Type': 'LoRA' if is_lora else 'Full Fine-tuning',
            'LoRA Rank': rank,
            'Epochs': result.get('epochs', 20),
            'Accuracy': result['metrics']['eval_accuracy'],
            'F1 Score': result['metrics']['eval_f1'],
            'Precision': result['metrics']['eval_precision'],
            'Recall': result['metrics']['eval_recall'],
            'Trainable Params (M)': result['parameters']['trainable'] / 1e6,
            'Total Params (M)': result['parameters']['total'] / 1e6,
            'Trainable %': result['parameters']['trainable_percent'],
            'Training Time (min)': result['training_time_minutes']
        })
    
    df = pd.DataFrame(data)
    return df.sort_values('Accuracy', ascending=False)

def plot_performance_comparison(df):
    """Plot 1: Performance comparison"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    ax.bar(x - width, df['Accuracy'], width, label='Accuracy', 
            alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    ax.bar(x, df['F1 Score'], width, label='F1 Score',
            alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax.bar(x + width, df['Precision'], width, label='Precision',
            alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics Comparison', 
                  fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'], fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0.80, 0.88])
    ax.grid(axis='y', alpha=0.3)
    
    # Annotate values
    for i, (acc, f1) in enumerate(zip(df['Accuracy'], df['F1 Score'])):
        ax.text(i - width, acc + 0.002, f'{acc:.4f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i, f1 + 0.002, f'{f1:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1 saved: plots/1_performance_comparison.png")

def plot_lora_rank_impact(df):
    """Plot 2: LoRA Rank impact"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lora_df = df[df['Type'] == 'LoRA'].copy()
    lora_df['Rank_num'] = lora_df['LoRA Rank'].astype(int)
    lora_df = lora_df.sort_values('Rank_num')
    
    ax.plot(lora_df['Rank_num'], lora_df['Accuracy'], 
            marker='o', linewidth=2.5, markersize=12, label='Accuracy',
            color='#3498db')
    ax.plot(lora_df['Rank_num'], lora_df['F1 Score'],
            marker='s', linewidth=2.5, markersize=12, label='F1 Score',
            color='#e74c3c')
    
    # Add Full fine-tuning baseline
    full_acc = df[df['Type'] == 'Full Fine-tuning']['Accuracy'].values[0]
    full_f1 = df[df['Type'] == 'Full Fine-tuning']['F1 Score'].values[0]
    ax.axhline(y=full_acc, color='#3498db', linestyle='--', 
               alpha=0.5, linewidth=2, label='Full Accuracy Baseline')
    ax.axhline(y=full_f1, color='#e74c3c', linestyle='--',
               alpha=0.5, linewidth=2, label='Full F1 Baseline')
    
    ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('LoRA Rank Impact on Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xticks(lora_df['Rank_num'])
    
    plt.tight_layout()
    plt.savefig('plots/2_lora_rank_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2 saved: plots/2_lora_rank_impact.png")

def plot_parameter_comparison(df):
    """Plot 3: Parameter comparison"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'Full' in m else '#4ecdc4' for m in df['Method']]
    bars = ax.barh(df['Method'], df['Trainable Params (M)'], 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, params) in enumerate(zip(bars, df['Trainable Params (M)'])):
        width = bar.get_width()
        percentage = df.iloc[i]['Trainable %']
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{params:.2f}M ({percentage:.1f}%)',
                va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('Trainable Parameters Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(df['Trainable Params (M)']) * 1.3])
    
    plt.tight_layout()
    plt.savefig('plots/3_parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3 saved: plots/3_parameter_comparison.png")

def plot_training_time(df):
    """Plot 4: Training time comparison"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'Full' in m else '#4ecdc4' for m in df['Method']]
    bars = ax.bar(range(len(df)), df['Training Time (min)'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, time) in enumerate(zip(bars, df['Training Time (min)'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.15,
                f'{time:.2f}min',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Method'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/4_training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4 saved: plots/4_training_time.png")

def plot_parameter_efficiency(df):
    """Plot 5: Parameter efficiency"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    efficiency = (df['F1 Score'] * 100) / df['Trainable Params (M)']
    colors = ['#e74c3c' if 'Full' in m else '#4ecdc4' for m in df['Method']]
    bars = ax.bar(range(len(df)), efficiency, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{eff:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Efficiency Score', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency (F1×100 / Trainable Params in M)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['Method'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/5_parameter_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 5 saved: plots/5_parameter_efficiency.png")

def plot_accuracy_vs_parameters(df):
    """Plot 6: Accuracy vs parameters scatter plot"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c' if 'Full' in m else '#4ecdc4' for m in df['Method']]
    sizes = df['Training Time (min)'] * 50
    
    scatter = ax.scatter(df['Trainable Params (M)'], df['Accuracy'],
                         s=sizes, c=colors, alpha=0.6, 
                         edgecolors='black', linewidth=2)
    
    for i, row in df.iterrows():
        ax.annotate(row['Method'],
                    (row['Trainable Params (M)'], row['Accuracy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Trainable Parameters (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Parameters (bubble size = training time)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/6_accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 6 saved: plots/6_accuracy_vs_parameters.png")

def plot_metrics_heatmap(df):
    """Plot 7: Metrics heatmap"""
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select key metrics
    heatmap_data = df[['Method', 'Accuracy', 'F1 Score', 'Precision', 'Recall']].set_index('Method')
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, linewidths=1, linecolor='black',
                vmin=0.80, vmax=0.88, ax=ax)
    
    ax.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/7_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 7 saved: plots/7_metrics_heatmap.png")

def plot_full_vs_best_lora(df):
    """Plot 8: Full vs best LoRA comparison"""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    full_df = df[df['Type'] == 'Full Fine-tuning']
    lora_df = df[df['Type'] == 'LoRA']
    
    if len(full_df) > 0 and len(lora_df) > 0:
        best_full = full_df.iloc[0]
        best_lora = lora_df.iloc[0]
        
        categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        full_scores = [best_full['Accuracy'], best_full['F1 Score'], 
                      best_full['Precision'], best_full['Recall']]
        lora_scores = [best_lora['Accuracy'], best_lora['F1 Score'],
                      best_lora['Precision'], best_lora['Recall']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, full_scores, width, label='Full Fine-tuning', 
                      alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, lora_scores, width, label=f'Best LoRA (r={best_lora["LoRA Rank"]})', 
                      alpha=0.8, color='#4ecdc4', edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Full Fine-tuning vs Best LoRA', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.80, 0.88])
    
    plt.tight_layout()
    plt.savefig('plots/8_full_vs_best_lora.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Plot 8 saved: plots/8_full_vs_best_lora.png")

def print_insights(df):
    """Print key insights"""
    print("\n" + "="*70)
    print("🔍 Key Findings and Insights")
    print("="*70)
    
    best = df.iloc[0]
    print(f"\n🏆 Best Model: {best['Method']}")
    print(f"   ├─ Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")
    print(f"   ├─ F1 Score: {best['F1 Score']:.4f}")
    print(f"   ├─ Trainable Parameters: {best['Trainable Params (M)']:.2f}M ({best['Trainable %']:.2f}%)")
    print(f"   └─ Training Time: {best['Training Time (min)']:.2f} minutes")
    
    # LoRA vs Full comparison
    lora_best = df[df['Type'] == 'LoRA'].iloc[0]
    full = df[df['Type'] == 'Full Fine-tuning'].iloc[0]
    
    acc_diff = (lora_best['Accuracy'] - full['Accuracy']) * 100
    param_reduction = (1 - lora_best['Trainable Params (M)'] / full['Trainable Params (M)']) * 100
    
    print(f"\n⚡ Best LoRA vs Full Fine-tuning:")
    print(f"   ├─ Performance Difference: {acc_diff:+.2f}% (LoRA {'better' if acc_diff > 0 else 'slightly worse'})")
    print(f"   ├─ Parameter Reduction: {param_reduction:.1f}%")
    print(f"   ├─ LoRA Parameters: {lora_best['Trainable Params (M)']:.2f}M")
    print(f"   └─ Full Parameters: {full['Trainable Params (M)']:.2f}M")
    
    # LoRA rank impact
    lora_df = df[df['Type'] == 'LoRA'].sort_values('LoRA Rank')
    print(f"\n📈 LoRA Rank Impact Analysis:")
    for _, row in lora_df.iterrows():
        rank = row['LoRA Rank']
        acc = row['Accuracy']
        params = row['Trainable Params (M)']
        print(f"   ├─ r={rank:2}: Accuracy={acc:.4f}, Parameters={params:.2f}M")
    
    # Analyze trend
    ranks = lora_df['LoRA Rank'].astype(int).tolist()
    accs = lora_df['Accuracy'].tolist()
    if accs[-1] > accs[0]:
        trend = "overall upward"
    else:
        trend = "fluctuating"
    print(f"   └─ Trend: Rank from {ranks[0]} to {ranks[-1]}, performance {trend}")
    
    # Efficiency analysis
    efficiency = df['F1 Score'] / (df['Trainable Params (M)'] / 100)
    best_eff_idx = efficiency.idxmax()
    best_eff = df.loc[best_eff_idx]
    
    print(f"\n💡 Most Parameter-Efficient: {best_eff['Method']}")
    print(f"   └─ Efficiency Score: {efficiency[best_eff_idx]:.2f} (F1 per 100M params)")

def main():
    print("="*70)
    print("Final Experiment Results Comprehensive Analysis")
    print("="*70)
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Load results
    results = load_final_results()
    
    if not results:
        print("\n✗ No result files found")
        return
    
    print(f"\n✓ Successfully loaded {len(results)} experiment results\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Save CSV
    df.to_csv('results/final_comparison.csv', index=False)
    print("✓ Comparison table saved: results/final_comparison.csv")
    
    # Print table
    print("\n" + "="*70)
    print("Detailed Comparison Table")
    print("="*70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    
    # Generate visualizations (each plot saved separately)
    print("\n" + "="*70)
    print("Generating visualization plots...")
    print("="*70)
    
    plot_performance_comparison(df)
    plot_lora_rank_impact(df)
    plot_parameter_comparison(df)
    plot_training_time(df)
    plot_parameter_efficiency(df)
    plot_accuracy_vs_parameters(df)
    plot_metrics_heatmap(df)
    plot_full_vs_best_lora(df)
    
    # Print insights
    print_insights(df)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  results/final_comparison.csv")
    print("  plots/1_performance_comparison.png")
    print("  plots/2_lora_rank_impact.png")
    print("  plots/3_parameter_comparison.png")
    print("  plots/4_training_time.png")
    print("  plots/5_parameter_efficiency.png")
    print("  plots/6_accuracy_vs_parameters.png")
    print("  plots/7_metrics_heatmap.png")
    print("  Plots/8_full_vs_best_lora.png")
    print("="*70)

if __name__ == "__main__":
    main()
