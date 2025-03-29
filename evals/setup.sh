HF_TOKEN=$1
# Check if HF_TOKEN is provided
if [ -z "$HF_TOKEN" ]; then
    echo "Error: Hugging Face token is required."
    echo "Usage: ./setup.sh <huggingface_token>"
    exit 1
fi

# Make sure all submodules are downloaded
git submodule update --init --recursive

# Install lm_eval
cd lm-evaluation-harness
pip install -e .
cd ..

# Install other dependencies
cd LiveCodeBench
pip install -e .
cd ..

# Login to Hugging Face
huggingface-cli login --token $HF_TOKEN