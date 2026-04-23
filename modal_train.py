import os
import modal
from pathlib import Path

# 1. Define persistent storage for model weights and tokenizers
# NetworkFileSystem is better for real-time visibility of your checkpoints.
nfs = modal.NetworkFileSystem.from_name("transformer-storage", create_if_missing=True)

# 2. Define the container image with all necessary dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio", 
        "datasets", 
        "tokenizers", 
        "evaluate", 
        "tqdm", 
        "tensorboard",
        "scipy",
        "sacremoses",
        "requests"
    )
    .add_local_file("model.py", remote_path="/root/model.py")
    .add_local_file("config.py", remote_path="/root/config.py")
    .add_local_file("dataset.py", remote_path="/root/dataset.py")
    .add_local_file("train.py", remote_path="/root/train.py")
)

app = modal.App("en-hi-transformer-train")

# 3. Define the training function
# We request a T4 GPU (affordable and sufficient for this model size)
@app.function(
    image=image,
    gpu="T4",
    network_file_systems={
        "/root/storage": nfs,
    },
    timeout=3600 * 24, # 24 hour timeout
)
def train():
    from train import train_model
    from config import get_config
    import torch
    import os
    import glob

    # Configuration for cloud storage
    config = get_config()
    config['model_folder'] = "/root/storage/weights"
    config['tokenizer_file'] = "/root/storage/tokenizers/tokenizer_{0}.json"
    config['experiment_name'] = "/root/storage/runs/tmodel"
    
    # Auto-resume logic: find the latest epoch or step in the storage
    weights_path = Path(config['model_folder'])
    weights_path.mkdir(parents=True, exist_ok=True)
    Path("/root/storage/tokenizers").mkdir(parents=True, exist_ok=True)

    weight_files = glob.glob(str(weights_path / "*.pt"))
    latest_checkpoint = None

    if weight_files:
        best_epoch = -1
        best_step = -1
        
        for f in weight_files:
            basename = os.path.basename(f)
            try:
                # Handle tmodel_step_5000.pt
                if "step_" in basename:
                    step_val = int(basename.split("step_")[1].split(".")[0])
                    if step_val > best_step:
                        best_step = step_val
                # Handle tmodel_01.pt
                else:
                    epoch_val = int(basename.split("_")[1].split(".")[0])
                    if epoch_val > best_epoch:
                        best_epoch = epoch_val
            except:
                continue
        
        # Decide which one is more recent (simplify by prioritizing steps if they exist in the current epoch)
        # Actually, if we have a step checkpoint, it's likely more recent than the last epoch file.
        if best_step != -1:
            latest_checkpoint = f"step_{best_step}"
        elif best_epoch != -1:
            latest_checkpoint = f"{best_epoch:02d}"

    if latest_checkpoint:
        config['preload'] = latest_checkpoint
        print(f"Resuming from latest checkpoint: {config['preload']}")
    else:
        config['preload'] = None
    
    print(f"Starting training on Modal with GPU: {torch.cuda.get_device_name(0)}")
    train_model(config)

@app.local_entrypoint()
def main():
    train.remote()

if __name__ == "__main__":
    # This allows running locally for debugging, but remote is preferred
    main()
