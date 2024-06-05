import sidechainnet as scn

import engineer

from engineer.schedulers.cosine import CosineAnnealingLR
import torch
torch.set_default_dtype(torch.float64)

def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.test_loader()
    test_loader = dataset.test_loader()
    model_config = config["model"]
    model = engineer.load_module(model_config.pop("module"))(**model_config)

    model = model.cuda()
    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    trainer_module = engineer.load_module(config["trainer"].pop("module"))
    
    trainer_config = config["trainer"]
    trainer_config['run_dir'] = config['run_dir']
    trainer_config['use_wandb'] = 'wandb' in config
    trainer = trainer_module(
        **trainer_config,
    )
    trainer.fit(model, optimizer, train_loader, val_loader=val_loader, test_loader=test_loader)

if __name__ == "__main__":
    engineer.fire(main)
