import sys
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
current_file_path = os.path.abspath(__file__)
vgc_dir = os.path.dirname(current_file_path)
sys.path.append(vgc_dir)

# Add DP3 path for utilities if needed
policy_dir = os.path.dirname(vgc_dir)
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)

@hydra.main(config_path="config", config_name="vgc_policy", version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    # print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.training.device)
    
    # 1. Dataset
    # Instantiate dataset using the config
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    
    dataloader = DataLoader(dataset, **cfg.dataloader)
    
    # 2. Model
    # Instantiate model using the config
    model = hydra.utils.instantiate(cfg.policy)
    
    # Set Normalizer
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    
    model.to(device)
    
    # 3. Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    # 4. Training Loop
    num_epochs = cfg.training.num_epochs
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in loop:
            # Move batch to device
            n_batch = {}
            n_batch['action'] = batch['action'].to(device)
            n_batch['target_keypose'] = batch['target_keypose'].to(device)
            n_batch['obs'] = {}
            for k, v in batch['obs'].items():
                n_batch['obs'][k] = v.to(device)
            
            optimizer.zero_grad()
            
            loss, loss_dict = model(n_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), diff=loss_dict['loss_diffusion'].item(), key=loss_dict['loss_keypose'].item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            # Hydra changes cwd to outputs/..., so we can save there
            save_path = os.path.join(os.getcwd(), f"checkpoints/vgc_model_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
