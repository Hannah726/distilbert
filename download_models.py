"""
预下载所有需要的模型和tokenizer
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

def download_with_retry(model_name, num_labels=3, max_retries=5):
    """带重试机制的下载"""
    
    print("="*60)
    print(f"开始下载模型: {model_name}")
    print("="*60)
    
    # 下载 Tokenizer
    print("\n[1/2] 下载 Tokenizer...")
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False  # 使用缓存
            )
            print("  ✓ Tokenizer 下载成功!")
            break
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:100]}")
            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("  ✗ Tokenizer 下载失败，已达最大重试次数")
                return False
    
    # 下载 Model
    print("\n[2/2] 下载 Model...")
    print("  (这可能需要几分钟，模型文件约 260MB)")
    
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                resume_download=True,
                force_download=False
            )
            print("  ✓ Model 下载成功!")
            
            # 显示模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n  模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:100]}")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("  ✗ Model 下载失败，已达最大重试次数")
                return False
    
    return False

def main():
    print("\n" + "="*60)
    print("DistilBERT 模型下载工具")
    print("="*60)
    
    model_name = "distilbert-base-uncased"
    
    # 显示缓存路径
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"\n缓存目录: {cache_dir}")
    
    # 开始下载
    success = download_with_retry(model_name, num_labels=3)
    
    if success:
        print("\n" + "="*60)
        print("✓ 所有模型文件下载完成!")
        print("="*60)
        print("\n现在可以开始训练了，训练脚本会自动使用本地缓存。")
        print("\n提示: 如果训练时还需要下载，说明缓存路径不一致。")
        print(f"      请确保训练脚本使用相同的缓存目录: {cache_dir}")
    else:
        print("\n" + "="*60)
        print("✗ 下载失败")
        print("="*60)
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 稍后再试")
        print("3. 或使用VPN/代理")

if __name__ == "__main__":
    main()
