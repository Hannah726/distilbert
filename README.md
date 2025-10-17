# Financial Sentiment Classification with LoRA

NUS DSA4213 Assignment 3 - Fine-tuning Pretrained Transformers

## 📊 Project Overview

This project compares two fine-tuning strategies for financial sentiment analysis:
- **Full Fine-tuning**: All parameters trainable
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning

**Dataset**: Financial PhraseBank (4,840 sentences with sentiment labels)  
**Model**: DistilBERT-base-uncased  
**Task**: 3-class classification (Positive, Neutral, Negative)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd financial-sentiment-lora

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Note that the script has many modes and may be a bit vague, but it is important to note that the terminals here are for reference only.
# Train with LoRA (faster, fewer parameters)
python train_lora.py

# Train with Full Fine-tuning
python train_full.py

# Generate comparison analysis
python compare.py
```

## 📈 Results
To be shown in the report "4213.pdf".

**Key Findings**:
- LoRA achieves comparable performance with **98% fewer trainable parameters**
- Training time is not reduced, somehow contrary to our intuition
- Excellent parameter efficiency for financial tasks

## 📁 Project Structure
Just part of the structure...
The script train_final_comparison.py is the one train enough epochs for comparison the LoRA with 8, 16, 32 and FT.
```
.
├── train_full.py          # Full fine-tuning script
├── train_lora.py          # LoRA fine-tuning script
├── compare.py             # Generate comparison analysis
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
├── results/
│   ├── full/
│   │   └── metrics.json
│   └── lora/
│       └── metrics.json
└── plots/
    └── comparison.png
...
```

## 🔬 Methodology

### Full Fine-tuning
- Updates all 66M parameters of DistilBERT
- Standard approach for transfer learning
- Higher GPU memory requirement

### LoRA Fine-tuning
- Injects trainable low-rank matrices into attention layers
- Only ~1.8M trainable parameters (r=8, alpha=16)
- Significantly more efficient for deployment

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 across all classes
- **Precision & Recall**: Per-class performance
- **Training Time**: Wall-clock time per epoch
- **Parameter Efficiency**: Performance per million parameters

## 📝 License

This project is for academic purposes (NUS DSA4213).

