import sidechainnet as scn

import engineer

from engineer.schedulers.cosine import CosineAnnealingLR
import torch
torch.set_default_dtype(torch.float64)

def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)
    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()
    model_config = config["model"]
    model = engineer.load_module(model_config.pop("module"))(**model_config)

    model = model.cuda()
    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    # scheduler_config = config['scheduler']
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode=scheduler_config["mode"], 
    #     factor=scheduler_config["factor"], 
    #     patience=scheduler_config["patience"])

    steps = config["trainer"]["max_steps"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    scheduler = CosineAnnealingLR(
        optimizer,
        steps,
        warmup_steps=int(1 / 32 * steps),
        decay_steps=int(1 / 4 * steps),
    )
    # scheduler=None

    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config['run_dir'] = config['run_dir']
    trainer_config['use_wandb'] = 'wandb' in config
    trainer = trainer_module(
        **trainer_config,
    )
    trainer.fit(model, optimizer, train_loader, scheduler, val_loader, test_loader=test_loader)


if __name__ == "__main__":
    engineer.fire(main)
