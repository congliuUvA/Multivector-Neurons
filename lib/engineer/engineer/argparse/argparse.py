import re
import argparse
import ast
import inspect
import os
import sys
import typing

import yaml

from ..utils.load_module import load_module

EXCEPTIONS = {"weight_decay": float, "lr": float}


def try_literal_eval(v):

    if v is None:
        return v

    if type(v) in (float, int, bool, list, tuple):
        return v

    try:
        return ast.literal_eval(v)
    except:
        pass

    if not isinstance(v, str):
        return v

    if v.startswith("[") and v.endswith("]"):
        assert " " not in v, f"Got space in list: {v}"
        try:
            return ast.literal_eval(v)
        except:
            return v[1:-1].split(",")
    elif v.startswith("(") and v.endswith(")"):
        assert " " not in v, f"Got space in tuple: {v}"
        try:
            return ast.literal_eval(v)
        except:
            return tuple(v[1:-1].split(","))
    else:
        return v


def get_type(v):
    if isinstance(v, bool):
        t = lambda x: (str(x).lower() == "true")
    else:
        t = type(v)
    return t


def pretty(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + k)
            pretty(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")


def unflatten(dictionary, sep="."):
    result = dict()
    for k, v in dictionary.items():
        parts = k.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = v
    return result


def check_optional(obj: typing.Any) -> bool:
    return (
        typing.get_origin(obj) is typing.Union
        and len(typing.get_args(obj)) == 2
        and typing.get_args(obj)[1] == type(None)
    )


def get_default_args(func):
    signature = inspect.signature(func)
    types = typing.get_type_hints(func)

    arguments = {}

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue

        if k in types and check_optional(types[k]):
            arguments[k] = None
        else:
            arguments[k] = v.default

    return arguments


def get_run_name(argv: list):
    name_parts: list = []
    for v in argv:
        if v.startswith("-C"):
            v = v[3:]
        if v.startswith("--"):
            name_parts.append(v[2:])
        elif os.path.exists(v):
            name_parts.append(os.path.splitext(os.path.basename(v))[0])

    return "_".join(name_parts)


def merge_dict(a, b):
    result = {**a}  # Copy a to result
    for key, value in b.items():
        if key in result and isinstance(result[key], dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def parse_args():
    argv = sys.argv

    # for i in range(len(argv)):
    #     if argv[i].startswith("--_"):
    #         argv[i] = argv[i].split("=", maxsplit=1)[1]

    argv = [v for v_ in argv for v in v_.replace("'", "").split()]

    # Check if additional YAML files should be parsed.
    yamls = []
    i = 0
    while i < len(argv):
        if argv[i] == "-C":
            yamls.append(argv[i + 1])
            del argv[i]
            del argv[i]
        else:
            i += 1

    config_dict = {}
    for y in yamls:
        with open(y, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
            if config is not None:
                config_dict = merge_dict(config_dict, config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--deterministic', type=try_literal_eval, default=False)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cpu")

    # Find the modules that should be loaded.
    i = 0
    while i < len(argv):
        arg = argv[i]
        regex = r"^--[^-.]+\.module"
        if bool(re.match(regex, arg)):
            k, v = arg.split("=")
            k = k.split(".")[0][2:]
            # print(f"Detected module '{k}' with value {v}. Adding to config...")
            config_dict[k] = {"module": v}
            del argv[i]
        else:
            i += 1

    kwargs = {}
    for key_value in argv:
        if not key_value.startswith("--"):
            continue
        key_value = key_value[2:]
        group_key, value = key_value.split("=", maxsplit=1)
        if "." in group_key:
            group, key = group_key.split(".")
            if group not in kwargs:
                kwargs[group] = {}
            kwargs[group][key] = value
        else:
            kwargs[group_key] = value

    for k in config_dict:
        if "module" in config_dict[k].keys():
            module = load_module(config_dict[k]["module"])
            argspec = get_default_args(module.__init__)
            group = parser.add_argument_group(k)
            v = config_dict[k].pop("module")

            group.add_argument(
                f"--{k}.module", default=v, type=str, help=f"Default: {v}"
            )

            config = {**argspec, **config_dict[k], **kwargs.get(k, {})}

            for k_, v in config.items():

                if k_ in config_dict[k]:
                    v = config_dict[k].pop(k_)

                v = try_literal_eval(v)

                group.add_argument(
                    f"--{k}.{k_}",
                    default=v,
                    type=try_literal_eval,
                    help=f"Default: {v}",
                )

            if len(config_dict[k]) > 0:
                raise KeyError(
                    f"Got unknown keys for {k} config: {tuple(config_dict[k].keys())}."
                )
        else:
            raise KeyError(f"Got key without module: {k}.")

    args = parser.parse_args(argv[1:])
    config = unflatten(vars(args))

    print("\nConfiguration\n---")
    pretty(config)

    name = get_run_name(sys.argv[1:])
    experiment = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    return config, name, experiment
