"""
laod the model and tokenizer
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

def download_with_retry(model_name, num_labels=3, max_retries=5):
    
    print("="*60)
    print(f"start to load: {model_name}")
    print("="*60)

    # Tokenizer
    print("\n[1/2] Tokenizer...")
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False
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
    
    # Model
    print("\n[2/2] Model...")
    
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                resume_download=True,
                force_download=False
            )

            # show the model
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n  total_params: {total_params:,} ({total_params/1e6:.1f}M)")
            
            return True
            
        except Exception as e:
            print(f"  ✗ fail: {str(e)[:100]}")
            if attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"  wait {wait_time} sec and retry...")
                time.sleep(wait_time)
            else:
                print("  ✗ Model loading failed and reach the max time")
                return False
    
    return False

def main():
    print("\n" + "="*60)
    print("="*60)
    
    model_name = "distilbert-base-uncased"
    
    # show the dir
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"\n cache dir: {cache_dir}")
    
    success = download_with_retry(model_name, num_labels=3)
    
    print("\n" + "="*60)
    print("✓ All model files downloaded!")
    print("="*60)
    print("\nYou can now start training. The training script will automatically use the local cache.")
    print("\nTip: If downloading is required during training, this indicates inconsistent cache paths.")
    print(f"Please make sure the training script uses the same cache directory: {cache_dir}")

if __name__ == "__main__":
    main()
