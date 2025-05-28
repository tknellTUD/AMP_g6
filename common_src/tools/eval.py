import os
import sys
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)
    
import hydra
from  omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
from torch.utils.data import DataLoader

from common_src.model.detector import CenterPoint
from common_src.dataset import ViewOfDelft, collate_vod_batch


@hydra.main(version_base=None, config_path='../config', config_name="eval")
def eval(cfg: DictConfig) -> None:
    print('Evaluating model...')
    L.seed_everything(cfg.seed, workers=True)
    
    val_dataset = ViewOfDelft(data_root=cfg.data_root, split='val')
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1, 
                                num_workers=cfg.num_workers, 
                                shuffle=False,
                                collate_fn=collate_vod_batch)

    checkpoint = torch.load(cfg.checkpoint_path, weights_only=False)
    checkpoint_params = DictConfig(checkpoint["hyper_parameters"])
    print('checkpoint params:', checkpoint_params.keys())
    print('checkpoint params cfg:', checkpoint_params.config.keys())
    
    model = CenterPoint.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)
    model.eval()
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
    )
    
    trainer.validate(model = model,
                     dataloaders=val_dataloader)
                     
    
if __name__ == '__main__':
    eval()