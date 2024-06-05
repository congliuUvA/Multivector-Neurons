import wandb
import os
import torch
import numpy
import random


def split_path(file, k):
    f = file
    for _ in range(k):
        f = os.path.split(f)[0]
    return f


def save_wandb(file, previous_file, metadata=None):
    if previous_file is not None:
        to_remove = previous_file.replace(split_path(previous_file, 2), '')[1:]
    else:
        to_remove = None
    run_dir = wandb.run.entity + '/' + wandb.run.project + '/' + wandb.run.id
    api = wandb.Api()
    run = api.run(run_dir)
    files = run.files()
    for f in files:
        if f.name == to_remove:
            f.delete()
   
    # Method 1
    wandb.save(file, base_path=split_path(file, 2))

    

    

    # Method 2
    # name = str(wandb.run.id) + "-" + "checkpoint"
    # artifact = wandb.Artifact(name, type="checkpoint", metadata=metadata)
    # artifact.add_file(file)
    # wandb.log_artifact(artifact)

    # Remove old artifacts
    # project = wandb.run.project
    # entity = wandb.run.entity
    # id = wandb.run.id
    # run = wandb.Api().run(f"{entity}/{project}/{id}")
    # for v in run.logged_artifacts():
    #         if len(v.aliases) == 0:
    #             v.delete()

def load_checkpoint(checkpoint_path, model, trainer, optimizer):

    state_dict = torch.load(checkpoint_path)

    model_state_dict = state_dict["model"]
    optimizer_state_dict = state_dict["optimizer"]
    train_state_dict = state_dict["train_state"]
    random_state_dict = state_dict["random_state"]

    model.load_state_dict(model_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    if trainer is not None:
        trainer.train_state.update(train_state_dict)

    torch.set_rng_state(random_state_dict["torch"])
    torch.cuda.set_rng_state(random_state_dict["cuda"])
    torch.cuda.set_rng_state_all(random_state_dict["cuda_all"])
    numpy.random.set_state(random_state_dict["numpy"])
    random.setstate(random_state_dict["random"])

    print(f"\nSuccessfully restored complete state from: {checkpoint_path}\n")


class Checkpointer:
    def __init__(self, run_dir: str, metrics: dict = None):
        super().__init__()

        self.run_dir = run_dir
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        # self._cached_model_state_dict = None
        # self._cached_optimizer_state_dict = None
        # self._cached_epoch = None
        # self._cached_step = None

        # if dir is not None:
        #     metrics = self.load_checkpoint(dir)

        # if type(metrics) == str:
        #     metrics = (metrics,)
        # if type(metrics) in (list, tuple):
        #     metrics = {m: float("inf") for m in metrics}
        self.best_metrics = {}
        self.directions = {}
        self.best_paths = {}
        if metrics is not None:
            for k, metric in metrics:
                if metric.direction == "up":
                    self.best_metrics[k] = -float("inf")
                    self.directions[k] = "up"
                else:
                    self.best_metrics[k] = float("inf")
                    self.directions[k] = "down"
                self.best_paths[k] = None

        # self.best_metrics = metrics

        # self.save_paths = {}

    #     def load_checkpoint(self, dir):
    #         state_dict = torch.load(dir)
    #         model = state_dict["model"]
    #         optimizer = state_dict["optimizer"]
    #         metrics = state_dict["metrics"]
    #         epoch = state_dict["epoch"]
    #         step = state_dict["step"]
    #         self._cached_model_state_dict = model
    #         self._cached_optimizer_state_dict = optimizer
    #         self._cached_epoch = epoch
    #         self._cached_step = step
    #         return metrics

    #     def restore(self, trainer, model, optimizer):
    #         if self._cached_model_state_dict is not None:
    #             if torch.distributed.is_initialized():
    #                 model.module.load_state_dict(self._cached_model_state_dict)
    #             else:
    #                 model.load_state_dict(self._cached_model_state_dict)
    #             print(f"Successfully restored model state dict from {self.dir}!")
    #         if self._cached_optimizer_state_dict is not None:
    #             optimizer.load_state_dict(self._cached_optimizer_state_dict)
    #             print(f"Successfully restored optimizer state dict from {self.dir}!")

    #         if self._cached_epoch is not None:
    #             trainer.current_epoch = self._cached_epoch
    #             print(f"Set current epoch to {self._cached_epoch}.")

    #         if self._cached_step is not None:
    #             trainer.global_step = self._cached_step
    #             print(f"Set global step to {self._cached_step}.")

    #         self._cached_epoch = None
    #         self._cached_step = None
    #         self._cached_model_state_dict = None
    #         self._cached_optimizer_state_dict = None

    @property
    def _is_master(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        else:
            return True

    def on_test_end(self, train_state, model, optimizer, metrics, prefix):

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        is_distributed = torch.distributed.is_initialized()
        if is_distributed:
            raise NotImplementedError("Should support multiple random states.")
        should_write = torch.distributed.get_rank() == 0 if is_distributed else True

        if not should_write:
            return

        def get_scalar(v):
            if isinstance(v, (float, int)):
                return v
            elif isinstance(v, torch.Tensor) and v.dim() == 0:
                return v.cpu().item()

        scalar_metrics = {
            k: get_scalar(v) for k, v in metrics.items() if get_scalar(v) is not None
        }
        # Drop anything with 's_it' in it.
        scalar_metrics = {k: v for k, v in scalar_metrics.items() if "s_it" not in k}

        metrics_str = "-".join([f"{k}={v:.4f}" for k, v in scalar_metrics.items()])
        metrics_str = metrics_str.replace("/", "_")
        filename = os.path.join(
            self.checkpoint_dir,
            f"step={train_state['global_step']}-epoch={train_state['current_epoch']}-{metrics_str}",
        )

        any_improved = False
        for m, v in self.best_metrics.items():

            direction = self.directions[m]
            m_key = prefix + "/" + m
            improved = (direction == "up" and metrics[m_key] > v) or (
                direction == "down" and metrics[m_key] < v
            )
            any_improved = any_improved or improved

            if not improved:
                continue

            print(f"Metric {m} improved to {metrics[m_key]:.4f}, saving checkpoint.")

            self.best_metrics[m] = metrics[m_key]

            model_state_dict = (
                model.module.state_dict()
                if torch.distributed.is_initialized()
                else model.state_dict()
            )

            random_state = {
                "torch": torch.get_rng_state(),
                "numpy": numpy.random.get_state(),
                "random": random.getstate(),
                "cuda": torch.cuda.get_rng_state(),
                "cuda_all": torch.cuda.get_rng_state_all(),
            }

            checkpoint = {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
                "train_state": train_state,
                "random_state": random_state,
            }

            best_filename = filename + f'_best_{m}.pt'
            torch.save(checkpoint, best_filename)
            if wandb.run is not None:
                save_wandb(best_filename, self.best_paths[m], metadata={"filename": best_filename})
            
            if self.best_paths[m] is not None:
                os.remove(self.best_paths[m])
            self.best_paths[m] = best_filename

            print(f"Successfully saved checkpoint to {os.path.dirname(best_filename)}")
        return any_improved


