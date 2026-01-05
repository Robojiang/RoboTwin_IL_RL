import sys
import os
import hydra
import torch
import numpy as np
import copy
import random
import wandb
import pathlib
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from termcolor import cprint

# Add paths
current_file_path = os.path.abspath(__file__)
vgc_dir = os.path.dirname(current_file_path)
sys.path.append(vgc_dir)

# Add DP3 path for utilities
policy_dir = os.path.dirname(vgc_dir)
dp3_pkg_path = os.path.join(policy_dir, 'DP3', '3D-Diffusion-Policy')
sys.path.append(dp3_pkg_path)
sys.path.append(os.path.join(dp3_pkg_path, 'diffusion_policy_3d'))

from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainVGCWorkspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        print(">>> [Init] Starting Workspace initialization...")
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        print(">>> [Init] Instantiating Model (Loading DINO)...")
        self.model = hydra.utils.instantiate(cfg.policy)
        print(">>> [Init] Model Instantiated.")

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0
        print(">>> [Init] Workspace initialization complete.")

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        print(">>> [Run] Starting Training Run...")
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.dataloader.batch_size = 2
            cfg.val_dataloader.batch_size = 2
            print("DEBUG MODE: Enabled. Config updated for quick run.")

        # configure dataset
        print(">>> [Run] Loading Dataset (This may take a while)...")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        print(">>> [Run] Dataset Loaded.")
        assert isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env_runner
        env_runner = None
        if 'task' in cfg and 'env_runner' in cfg.task:
             env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        # configure logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cfg.logging.name = f"{cfg.task.name}_{timestamp}"
        
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # training loop
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            train_losses = list()
            
            self.model.train()
            loop = tqdm(train_dataloader, desc=f"Epoch {self.epoch}", leave=False)
            
            for batch_idx, batch in enumerate(loop):
                # Move batch to device
                n_batch = {}
                n_batch['action'] = batch['action'].to(device)
                n_batch['target_keypose'] = batch['target_keypose'].to(device)
                n_batch['obs'] = {}
                for k, v in batch['obs'].items():
                    n_batch['obs'][k] = v.to(device)
                
                self.optimizer.zero_grad()
                
                loss, loss_dict = self.model(n_batch)
                
                loss.backward()
                self.optimizer.step()
                lr_scheduler.step()
                
                if cfg.training.use_ema:
                    ema.step(self.model)
                
                train_losses.append(loss.item())
                
                if (self.global_step % cfg.training.sample_every) == 0:
                    step_log = {
                        'train_loss': loss.item(),
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(loss_dict)
                    wandb.log(step_log, step=self.global_step)
                
                loop.set_postfix(loss=loss.item())
                self.global_step += 1
                
                if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                    break

            # Epoch end
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss
            
            # Validation
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            
            # Validation Loop
            if (self.epoch % cfg.training.val_every) == 0:
                with torch.no_grad():
                    val_losses = list()
                    for batch_idx, batch in enumerate(val_dataloader):
                        n_batch = {}
                        n_batch['action'] = batch['action'].to(device)
                        n_batch['target_keypose'] = batch['target_keypose'].to(device)
                        n_batch['obs'] = {}
                        for k, v in batch['obs'].items():
                            n_batch['obs'][k] = v.to(device)
                        
                        loss, loss_dict = policy(n_batch)
                        val_losses.append(loss.item())
                        
                        if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                            break
                    
                    if len(val_losses) > 0:
                        step_log['val_loss'] = np.mean(val_losses)

            # Run validation runner
            if (self.epoch % cfg.training.rollout_every) == 0:
                if env_runner is not None:
                    runner_log = env_runner.run(policy)
                    if runner_log is not None:
                        step_log.update(runner_log)
            
            # Checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # 1. Save periodic checkpoint (History) - 也就是你想要的“带上代数”
                # 这会生成如 epoch=0030.ckpt, epoch=0060.ckpt 的文件，保留训练历史
                self.save_checkpoint(tag=f"epoch={self.epoch:04d}")

                # 2. Save latest checkpoint (Latest) - 用于断点续训
                # 对应配置文件中的 save_last_ckpt: True
                if cfg.checkpoint.get('save_last_ckpt', True):
                    self.save_checkpoint(tag="latest")

                # 3. Save TopK (Best) - 保存指标最好的模型
                if env_runner is not None:
                    metric_key = cfg.checkpoint.topk.monitor_key
                    metric_value = step_log.get(metric_key, 0.0)
                    data = {
                        'epoch': self.epoch,
                        metric_key: metric_value
                    }
                    topk_path = topk_manager.get_ckpt_path(data)
                    if topk_path is not None:
                        self.save_checkpoint(path=pathlib.Path(topk_path))

            wandb.log(step_log, step=self.global_step)
            self.epoch += 1

    def save_checkpoint(self, path=None, tag="latest"):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state_dicts": {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema_model": self.ema_model.state_dict() if self.ema_model else None,
            },
            "epoch": self.epoch,
            "global_step": self.global_step
        }
        torch.save(payload, path.open("wb"))
        print(f"Checkpoint saved to: {path.absolute()}")
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

@hydra.main(config_path="config", config_name="vgc_policy", version_base="1.2")
def main(cfg):
    workspace = TrainVGCWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

