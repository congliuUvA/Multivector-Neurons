import itertools
import os
import re
import subprocess
import sys
import warnings

import wandb
import yaml


def generate_sbatch_lines(slurm_args_str):
    slurm_args = slurm_args_str.split()
    sbatch_lines = ["#!/bin/bash"]

    i = 0
    while i < len(slurm_args):
        arg = slurm_args[i]
        if arg.startswith("--"):
            arg_name = arg[2:]
            if "=" in arg_name:
                arg_name, arg_value = arg_name.split("=")
                sbatch_lines.append(f"#SBATCH --{arg_name}={arg_value}")
            else:
                arg_value = slurm_args[i + 1]
                sbatch_lines.append(f"#SBATCH --{arg_name}={arg_value}")
                i += 1
        elif arg.startswith("-"):
            arg_name = arg[1:]
            arg_value = slurm_args[i + 1]
            sbatch_lines.append(f"#SBATCH -{arg_name} {arg_value}")
            i += 1
        i += 1

    return sbatch_lines


def commit_files(sweep_id):
    command = f"""
        find * -size -4M -type f -print0 | xargs -0 git add
        git add -u
        git commit --allow-empty -m {sweep_id}
        git tag {sweep_id}
        git push
        git push origin {sweep_id}
    """
    os.system(command)


def write_jobfile(slurm_string, n_jobs, command, directory, jobfile_path, sweep_id):
    sbatch_lines = generate_sbatch_lines(slurm_string)

    sbatch_lines.append(f"#SBATCH --array=1-{n_jobs}")
    sbatch_lines.append(f"#SBATCH --output={os.path.join(directory, 'slurm-%j.out')}")
    sbatch_lines.append(f"cd {directory}")
    sbatch_lines.append(f"git checkout {sweep_id}")
    sbatch_lines.append("source env/activate.sh")
    sbatch_lines.append(str(command))
    sbatch_script = "\n".join(sbatch_lines)

    with open(jobfile_path, "w") as f:
        f.write(sbatch_script)


def replace_variables(command, locals):
    # Use a regular expression to find words between "{}" brackets in the string
    pattern = re.compile(r"\{([^}]+)\}")
    matches = pattern.findall(command)

    # For each word found, replace it with its value from the locals_dict dictionary
    for match in matches:
        command = command.replace("{" + match + "}", str(locals[match]))
    return command


def git_status():  # pragma: no cover

    # Fetch the remote changes without applying them
    subprocess.run(
        ["git", "fetch"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Get the status comparing local and remote
    status = subprocess.getoutput("git rev-list --left-right --count HEAD...@{u}")

    local, remote = map(int, status.split("\t"))

    if local and remote:
        return "diverged"
    elif local:
        return "ahead"
    elif remote:
        return "behind"
    else:
        return "up-to-date"


def git_detached():
    # Get the output of 'git status'
    git_status_output = subprocess.getoutput("git status")
    return "HEAD detached" in git_status_output


def check_git_detached():
    if git_detached():
        raise RuntimeError(f"git is a detached HEAD. Please checkout a branch.")


def process_args_and_load_config(local=False):
    check_git_detached()

    if not local:
        status = git_status()
        if status == "behind":
            warning = f"git is behind remote. Please pull changes first."
        elif status == "diverged":
            warning = f"git has diverged from remote. Please push or pull changes."
        else:
            warning = None

        if warning is not None:
            warnings.warn(warning)
            cont = input("Continue? [y/N]")
            if cont.lower() != "y":
                raise RuntimeError("Aborting.")

    argv = sys.argv
    config = argv[1]

    with open(config) as f:
        config = yaml.load(f, yaml.SafeLoader)

    name = config["name"]
    project = config["project"]
    entity = config["entity"]

    os.environ['WANDB_NAME'] = name
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_ENTITY'] = entity
    
    return config


def add_clargs_to_command(argv, config):
    for kwarg in argv[2:]:
        config["command"].append(kwarg)
    return config


def main():

    config = process_args_and_load_config()
    entity = config['entity']
    project = config['project']
    config = add_clargs_to_command(sys.argv, config)
    sweep_id = wandb.sweep(sweep=config, project=project, entity=entity)
    on_cluster = "cluster" in config
    command = "WANDB_ENABLED=TRUE wandb agent --count 1 {entity}/{project}/{sweep_id}"
    
    if on_cluster:
        cluster_config = config["cluster"]
        slurm_arguments = cluster_config["slurm"]
        directory = replace_variables(cluster_config["directory"], {'project_dir': os.environ['PROJECT_DIR']})
        all_values = [config["parameters"][k]["values"] for k in config["parameters"]]
        num_jobs = len(tuple(itertools.product(*all_values)))
        command = replace_variables(command, locals())

        rel_path = os.path.join("slurm", f"{sweep_id}.slurm")
        os.makedirs(os.path.dirname(rel_path), exist_ok=True)
        write_jobfile(slurm_arguments, num_jobs, command, directory, rel_path, sweep_id)
    else:
        command = replace_variables(command, locals())
        cluster_config = None
        directory = None

    commit_files(sweep_id)

    if on_cluster:
        assert cluster_config is not None
        assert directory is not None
        print("\nSuccessfully submitted sweep. To fire remotely, run:")
        print(f"ssh {cluster_config['address']}")
        print(
            f"cd {directory} && git pull && git checkout {sweep_id} && sbatch {rel_path} && git switch -\n"
        )

    else:
        print(f"Run this sweep with:")
        print(f"git checkout {sweep_id} && {command}")


if __name__ == "__main__":
    main()
