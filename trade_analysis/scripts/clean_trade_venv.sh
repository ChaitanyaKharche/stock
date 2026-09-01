#!/bin/bash

# Clean up HuggingFace and ML caches for trade-venv environment
echo "🧹 Starting comprehensive cache cleanup for trade-venv environment..."

# Set environment variables to current conda environment
CONDA_ENV_PATH=$CONDA_PREFIX
HF_CACHE_DIR="$HOME/.cache/huggingface"
TORCH_CACHE_DIR="$HOME/.cache/torch"
PIP_CACHE_DIR="$HOME/.cache/pip"

# Check if we're in the correct environment
if [[ "$CONDA_DEFAULT_ENV" != "trade-venv" ]]; then
    echo "❌ Not in trade-venv environment. Please activate it first:"
    echo "conda activate trade-venv"
    exit 1
fi

echo "✅ Detected trade-venv environment at: $CONDA_ENV_PATH"

# 1. Clean HuggingFace cache
echo ""
echo "🔄 Step 1: Cleaning HuggingFace cache..."
if [ -d "$HF_CACHE_DIR" ]; then
    echo "Found HuggingFace cache at: $HF_CACHE_DIR"
    du -sh "$HF_CACHE_DIR"

    # Use huggingface-cli if available
    if command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli for safe cleanup..."
        huggingface-cli delete-cache --disable-tui --yes
    else
        echo "Manual cleanup of HuggingFace cache..."
        # Clean specific problematic models
        rm -rf "$HF_CACHE_DIR/hub/models--deepseek-ai--DeepSeek-V3"
        rm -rf "$HF_CACHE_DIR/hub/models--meta-llama--Meta-Llama-3-70B-Instruct"
        rm -rf "$HF_CACHE_DIR/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1"
        rm -rf "$HF_CACHE_DIR/hub/models--Qwen--Qwen2.5-72B-Instruct"

        # Clean incomplete downloads
        rm -rf "$HF_CACHE_DIR/hub/*.lock"
        rm -rf "$HF_CACHE_DIR/hub/tmp*"
        find "$HF_CACHE_DIR" -name "*.incomplete" -delete
    fi

    echo "After cleanup:"
    du -sh "$HF_CACHE_DIR" 2>/dev/null || echo "Cache cleaned"
else
    echo "No HuggingFace cache found"
fi

# 2. Clean PyTorch cache
echo ""
echo "🔄 Step 2: Cleaning PyTorch cache..."
if [ -d "$TORCH_CACHE_DIR" ]; then
    echo "Found PyTorch cache at: $TORCH_CACHE_DIR"
    du -sh "$TORCH_CACHE_DIR"
    rm -rf "$TORCH_CACHE_DIR"/*
    echo "PyTorch cache cleaned"
else
    echo "No PyTorch cache found"
fi

# 3. Clean pip cache (only for current environment)
echo ""
echo "🔄 Step 3: Cleaning pip cache..."
if [ -d "$PIP_CACHE_DIR" ]; then
    echo "Found pip cache at: $PIP_CACHE_DIR"
    du -sh "$PIP_CACHE_DIR"
    pip cache purge
    echo "Pip cache cleaned"
else
    echo "No pip cache found"
fi

# 4. Clean conda cache for this environment
echo ""
echo "🔄 Step 4: Cleaning conda cache..."
conda clean -a --yes

# 5. Clear GPU memory if CUDA is available
echo ""
echo "🔄 Step 5: Clearing GPU memory..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU cache cleared')
else:
    print('No CUDA available')
" 2>/dev/null

# 6. Clean temporary files
echo ""
echo "🔄 Step 6: Cleaning temporary files..."
rm -rf /tmp/tmp*transformers*
rm -rf /tmp/tmp*torch*
rm -rf /tmp/tmp*huggingface*

# 7. Check final disk usage
echo ""
echo "📊 Final disk usage summary:"
echo "Home directory usage:"
du -sh $HOME/.cache 2>/dev/null || echo "Cache directory cleaned"
df -h $HOME | tail -1

echo ""
echo "✅ Cleanup complete! Your trade-venv environment cache has been cleaned."
echo ""
echo "🔧 Next steps:"
echo "1. Set your environment variables:"
echo '   export FINNHUB_API_KEY="your-finnhub-key"'
echo '   export TWELVE_KEY="your-twelve-data-key"'
echo '   export REDDIT_CLIENT_ID="your-reddit-client-id"'
echo '   export REDDIT_CLIENT_SECRET="your-reddit-client-secret"'
echo '   export REDDIT_USER_AGENT="script:stock-opinion-analyzer:v1.0 (by /u/your-reddit-username)"'
echo ""
echo "2. Also fix the model identifier in your code:"
echo "   Change: 'FinGPT/fingpt-mt_llama2-13b_lora'"
echo "   To: 'FinGPT/fingpt-sentiment_llama2-13b_lora'"
echo ""
echo "3. Then restart your API server:"
echo "   python -m uvicorn trade_analysis.enhanced_api:app --host 0.0.0.0 --port 8000"
