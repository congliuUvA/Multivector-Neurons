import itertools
import subprocess
import sys

import yaml

from .sweep import check_git_detached, process_args_and_load_config


def main():
    check_git_detached()
    config = process_args_and_load_config(local=True)

    parameters = config["parameters"]
    base_command = config["command"]
    parameters = config["parameters"]
    base_command = config["command"]

    args = sys.argv[2:]

    for i, c in enumerate(base_command):
        if c == "${env}":
            base_command[i] = "/usr/bin/env"
        elif c == "${interpreter}":
            base_command[i] = "python -u"
        elif c == "${program}":
            base_command[i] = config["program"]
        elif c == "${args}":
            del base_command[i]

    for k, v in parameters.items():
        parameters[k] = parameters[k]["values"]

    keys, values = zip(*parameters.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for d in permutations_dicts:
        print("\nRunning with configuration:")
        print(yaml.dump(d))
        print()
        command = base_command + [f"--{k}={v}" for k, v in d.items()]
        command = " ".join(command + args)
        print(command)
        result = subprocess.call(command, shell=True)

        if result != 0:
            break


if __name__ == "__main__":
    main()
