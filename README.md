```markdown
# MFC-LLM
# MFC-LLM Running Guide

## Prerequisites
- Python 3.8+ installed
- Sufficient disk space (≥20GB recommended for storing model weights and training files)
- CUDA-enabled GPU (≥16GB VRAM recommended for accelerated training)

## 1. Environment Setup
Install project dependencies with the following command:
```bash
pip install -r requirements.txt

```

> Note: If dependency conflicts occur during installation, try upgrading pip or using a virtual environment for isolation:
> ```bash
> pip install --upgrade pip
> python -m venv mfc-env
> # Activate virtual environment on Windows
> mfc-env\Scripts\activate
> # Activate virtual environment on Linux/Mac
> source mfc-env/bin/activate
> pip install -r requirements.txt
> 
> ```
> 
> 

## 2. Model Weight Download

Download the Qwen2.5-1.5B model from Hugging Face and place it in the specified directory:

1. Visit the Hugging Face model repository: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
2. Download all complete model files (including config.json, model-00001-of-00002.safetensors, etc.)
3. Create the path `MFC-LLM/LLM/qwen_weight` in the project root directory (if it doesn't exist)
4. Copy all downloaded model files to the `qwen_weight` directory. The final directory structure should be:
```
MFC-LLM/
└── LLM/
    └── qwen_weight/
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        ├── tokenizer_config.json
        └── tokenizer.model

```



> Optional: Auto-download using Hugging Face `transformers` (install `huggingface-hub` first):
> ```bash
> pip install huggingface-hub
> huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir MFC-LLM/LLM/qwen_weight --local-dir-use-symlinks False
> 
> ```
> 
> 

## 3. Dataset Preparation

Download the **hzmmmm/PHM2012_LLM** dataset from Hugging Face and organize it within the project:

1. Visit the dataset repository: [hzmmmm/PHM2012_LLM](https://www.google.com/search?q=https://huggingface.co/datasets/hzmmmm/PHM2012_LLM)
2. Create the directory `MFC-LLM/data` .
3. Copy all downloaded model files to the `data` directory. The final directory structure should be:
```
MFC-LLM/
└── data/
    └── PHM2012_data.hdf5
    └── PHM2012_data.sqlite
        


## 4. Path Configuration Modification

Modify all path-related configurations in the project according to your actual deployment environment:

* **Model weight path:** Ensure the model loading path in the code points to `MFC-LLM/LLM/qwen_weight`
* **Dataset path:** Update the reading path for pre-training/fine-tuning datasets to point to `MFC-LLM/Dataset` (or where you downloaded it in Step 3)
* **Output path:** Specify the save path for model checkpoints and log files
* **Other paths:** Adjust paths for configuration files, cache files, etc. (locate via code search)

> Tip: Use global search for keywords like `path`, `dir`, or `load_from` to quickly find path configurations that need modification.

## 5. Pre-training

After completing the above preparations, run the pre-training script:

```bash
python pre_training.py

```

> Note:
> * Pre-training may take a long time. Adjust parameters like batch_size based on your GPU performance.
> * If training is interrupted, check if the script supports resuming from checkpoints or restart the training process.
> 
> 

## 6. Fine-tuning

After pre-training is complete, run the fine-tuning script:

```bash
python fine_tuning.py

```

> Explanation:
> * Fine-tuning relies on pre-trained model checkpoints. Ensure the script correctly configures the path to load the pre-trained model.
> * Adjust hyperparameters like learning rate and training epochs according to task requirements.
> 
> 

## Common Issues

1. Dependency installation failures: Verify Python version compatibility or install specific dependency versions manually.
2. Slow download: Use a Hugging Face mirror or manually download and upload files to the server.
3. Insufficient VRAM during training: Reduce batch_size, use gradient accumulation, or switch to a GPU with larger VRAM.
4. Path errors: Double-check all path configurations to match the actual file locations.

For other issues, refer to project log files or submit an Issue for support.

```

```