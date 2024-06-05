import sys
import warnings
import datetime
import os
import random
import socket
import string
import subprocess
from typing import Any

import torch
import torch.distributed as dist
import yaml

from .argparse.argparse import parse_args
from .utils.seed import set_seed
from .tee import Tee


USE_WANDB = (
    "WANDB_ENABLED" in os.environ and os.environ["WANDB_ENABLED"].lower() == "true"
)
import wandb

USE_DISTRIBUTED = "NCCL_SYNC_FILE" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ


def generate_run_id():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=7))


def generate_dirname(run_id):
    now = datetime.datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    name = f"run-{date_time}-{run_id}"
    return name


def _add_sweep_name(name: str) -> str:
    if "WANDB_SWEEP_ID" in os.environ:
        project = os.environ["WANDB_PROJECT"]
        entity = os.environ["WANDB_ENTITY"]
        sweep_id = os.environ["WANDB_SWEEP_ID"]
        api = wandb.Api()
        sweep: Any = api.sweep(entity + "/" + project + "/" + sweep_id)
        sweep_config: dict[str, Any] = sweep.config
        if "name" in sweep_config:
            sweep_name: str = sweep_config["name"]
            name = sweep_name + "_" + name
    return name


def _setup_torchelastic():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")

    return rank, local_rank, world_size


def _setup_slurm():
    slurm_procid = int(os.environ["SLURM_PROCID"])
    slurm_nodeid = int(os.environ["SLURM_NODEID"])
    slurm_localid = int(os.environ["SLURM_LOCALID"])
    # slurm_nodename = os.environ["SLURMD_NODENAME"]
    # slurm_job_nnodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    slurm_ntasks = int(os.environ["SLURM_NTASKS"])

    tasks_per_node = slurm_procid // slurm_nodeid if slurm_nodeid > 0 else slurm_procid

    # Calculate the local rank and world size
    local_rank = slurm_localid
    world_size = slurm_ntasks
    rank = slurm_nodeid * tasks_per_node + slurm_localid

    dist.init_process_group(
        backend="nccl",
        init_method=f'file://{os.environ["NCCL_SYNC_FILE"]}',
        world_size=world_size,
        rank=rank,
    )

    return rank, local_rank, world_size


def _ddp_setup():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError("Cannot initialize NCCL without visible CUDA devices.")

    hostname = socket.gethostname()
    print(f"Setting up DDP on {hostname}.")
    if "TORCHELASTIC_RUN_ID" in os.environ:
        print("TorchElastic detected.")
        _setup = _setup_torchelastic
    elif "NCCL_SYNC_FILE" in os.environ:
        print("Detected NCCL_SYNC_FILE. Assuming SLURM cluster.")
        _setup = _setup_slurm
    else:
        raise ValueError("Unable to detect DDP setup.")

    rank, local_rank, world_size = _setup()

    print(
        f"{hostname} ready! Rank: {rank}. Local rank: {local_rank}. World size: {world_size}."
    )
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    device = f"cuda:{int(devices[local_rank])}"
    torch.cuda.set_device(device)

    assert dist.is_initialized()

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def _setup_wandb(*args, **kwargs):

    if "WANDB_SWEEP_ID" in os.environ:

        sweep_id = os.environ["WANDB_SWEEP_ID"]
        commit_hash = subprocess.getoutput("git rev-parse HEAD")

        # Get the tag associated with that commit, if it exists
        tag = subprocess.getoutput(f"git tag --contains {commit_hash}")

        if tag != sweep_id:
            raise ValueError(
                f"Tag {tag} does not match sweep id {sweep_id}. Commit hash: {commit_hash}."
            )

    if dist.is_initialized():
        should_initialize = dist.get_rank() == 0
    else:
        should_initialize = True

    if should_initialize:
        return wandb.init(*args, **kwargs)


def restore_wandb(run_dir, config):  # pragma: no cover
    api = wandb.Api()
    run = api.run(config["continue"])

    for file in run.files():
        file.download(
            root=os.path.join(run_dir, "files"), replace=True
        )  # point to run_dir

    for artifact in run.logged_artifacts():
        if artifact.type == "checkpoint":
            artifact.download(root=os.path.join(run_dir, "files", "checkpoints"))

    config["continue"] = run_dir


def fire(function):
    config, name, experiment = parse_args()

    config["cwd"] = os.getcwd()
    run_dir = os.path.join(os.getcwd(), "runs")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    seed = config["seed"]
    assert isinstance(seed, int)
    deterministic = config.get("deterministic", False)

    if "dtype" in config:
        dtype = config["dtype"]
        print("\nUsing dtype", dtype)
        if dtype == "float64":
            torch.set_default_dtype(torch.float64)
        elif dtype == "float32":
            torch.set_default_dtype(torch.float32)
        else:
            raise ValueError(f"Unknown dtype {dtype}.")

    if USE_DISTRIBUTED:
        dist_cfg = _ddp_setup()
        config["dist"] = dist_cfg

    if USE_WANDB:
        name = _add_sweep_name(name)
        wandb_kwargs = dict(
            config=config.copy(),
            dir=run_dir,
            name=name,
        )
        wandb_cfg = _setup_wandb(**wandb_kwargs)
        config["wandb"] = wandb_cfg

        files_dir = wandb_cfg.dir
        run_dir = os.path.dirname(files_dir)
    else:
        run_id = generate_run_id()
        files_dir = os.path.join(run_dir, "devrun", generate_dirname(run_id), "files")
        os.makedirs(files_dir, exist_ok=True)
        run_dir = os.path.dirname(files_dir)
        yaml.dump(config, open(os.path.join(run_dir, "config.yaml"), "w"))
    
    seed = set_seed(seed, deterministic)
    config["run_dir"] = run_dir

    # if 'continue' in config and config["continue"] is not None:
    #     if wandb_cfg is not None:  # pragma: no cover
    #         restore_wandb(run_dir, config)
    #     else:
    #         dir_util.copy_tree(config["continue"], run_dir)

    stdout_file = os.path.join(run_dir, "stdout.txt")
    with open(stdout_file, "w") as f:
        tee = Tee(sys.stdout, f)
        sys.stdout = tee
        print("\nSaving files to", run_dir, "\n")

        function(config)

        print("\nProgram completed. See results at ", run_dir)

    sys.stdout = sys.__stdout__

    if dist.is_initialized():
        dist.destroy_process_group()
