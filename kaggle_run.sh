#!/bin/bash
# kaggle_run.sh: Setup environment and run 3DGS training on 2x T4 GPU

echo "[1] Checking CUDA and PyTorch versions..."
nvcc --version
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA {torch.version.cuda}')"

echo "[2] Synchronizing PyTorch with Kaggle's NVCC to prevent CUDA compilation trap..."
# Khuyến nghị cài đặt torch 2.1.2 cu118 (do Kaggle NVCC thường là 11.8)
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

echo "[3] Installing Submodules (diff-gaussian-rasterization & simple-knn)..."
pip install plyfile tqdm wandb opencv-python
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn

echo "[3.5] Logging into Weights & Biases..."
wandb login 

echo "[4] Starting Multi-GPU Training (Data distribution)..."
DATA_DIR="/kaggle/input/bts-digital-twin-phase1/phase1"
OUTPUT_DIR="/kaggle/working/output"

# We expect scenes in public_set and private_set1
PUBLIC_SCENES=(${DATA_DIR}/public_set/*)
PRIVATE_SCENES=(${DATA_DIR}/private_set1/*)
ALL_SCENES=("${PUBLIC_SCENES[@]}" "${PRIVATE_SCENES[@]}")

# Function to run training on a specific GPU
train_scene() {
    SCENE_PATH=$1
    GPU_ID=$2
    SCENE_NAME=$(basename $SCENE_PATH)
    echo "Starting training for $SCENE_NAME on GPU $GPU_ID..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        -s $SCENE_PATH \
        -m $OUTPUT_DIR/$SCENE_NAME \
        --use_wandb \
        --wandb_project "bts-digital-twin-kaggle" \
        --iterations 30000 > $OUTPUT_DIR/${SCENE_NAME}_train.log 2>&1
        
    echo "Finished training $SCENE_NAME, generating test poses..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
        -m $OUTPUT_DIR/$SCENE_NAME \
        --skip_train \
        --iteration 30000 > $OUTPUT_DIR/${SCENE_NAME}_render.log 2>&1
}

mkdir -p $OUTPUT_DIR

# Dispatch scenes across 2 GPUs
for i in "${!ALL_SCENES[@]}"; do
    SCENE=${ALL_SCENES[$i]}
    GPU=$(($i % 2))
    
    train_scene $SCENE $GPU &
    
    # Wait for the batch of 2 GPUs to finish before launching next batch
    if [ $(($i % 2)) -eq 1 ]; then
        wait
    fi
done
wait

echo "All scenes processed!"
bash submission_gen.sh
