import argparse
from os import path, makedirs
from experiments import select_experiment
import torch
import yaml
import os
import silence_tensorflow.auto


def create_dir_structure(config, model_name):
    subdirs = ["ckpt", "config", "generated", "log"]


    # model_name = config['model_name'] if model_name is None else model_name
    structure = {subdir: path.join(config["base_dir"],config["experiment"],subdir,model_name) for subdir in subdirs}
    return structure

def load_parameters(config_name, restart, model_name):
    with open(config_name,"r") as f:
        cdict_old = yaml.load(f,Loader=yaml.FullLoader)
    cdict_old['general']['model_name'] = model_name
    # if we just want to test if it runs
    dir_structure = create_dir_structure(cdict_old["general"], model_name)
    saved_config = path.join(dir_structure["config"], "config.yaml")
    if restart:
        if path.isfile(saved_config):
            with open(saved_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

        [makedirs(dir_structure[d]) for d in dir_structure if not path.isdir(dir_structure[d])]

        cdict['testing'] = cdict_old['testing']
        cdict['general']['model_name'] = model_name

    else:
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        if path.isfile(saved_config) and not cdict_old["general"]["debug"]:
            print(f"\033[93m" + "WARNING: Model has been started somewhen earlier: Resume training (y/n)?" + "\033[0m")
            while True:
                answer = input()
                if answer == "y" or answer == "yes":
                    with open(saved_config,"r") as f:
                        cdict = yaml.load(f, Loader=yaml.FullLoader)
                    cdict['testing'] = cdict_old['testing']
                    restart = True
                    break
                elif answer == "n" or answer == "no":
                    with open(saved_config, "w") as f:
                        yaml.dump(cdict_old, f, default_flow_style=False)
                    cdict = cdict_old
                    break
                else:
                    print(f"\033[93m" + "Invalid answer! Try again!(y/n)" + "\033[0m")
        else:
            with open(saved_config, "w") as f:
                yaml.dump(cdict_old,f,default_flow_style=False)

            cdict = cdict_old


    return cdict, dir_structure, restart

def check_ckpt_paths(config):
    if "DATAPATH" not in os.environ:
        return config

    for key in config:
        for k in config[key]:
            if k == "ckpt":
                config[key][k] = path.join(os.environ["DATAPATH"],config[key][k][1:])


    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/latent_flow_net.yaml",
                        help="Define config file")
    parser.add_argument("-m","--model_name", type=str, required=True,help="Run name for the project that shall be resumed for training or testing.")
    parser.add_argument("-r","--resume", default=False ,action="store_true", help='Whether or not to resume the training.')
    parser.add_argument("-g","--gpus",default=[0], type=int,
                        nargs="+",help="GPU to use.")
    parser.add_argument("--test",default='none', type=str, choices=['none','fvd','accuracy','samples','diversity', 'kps_acc', 'transfer', 'control_sensitivity'],help="Whether to start in  infer mode?")
    parser.add_argument("--last_ckpt",default=False,action="store_true",help="Whether to load the last checkpoint if resuming training.")
    parser.add_argument("--target_version",default=None,type=int,help="The target version for loading checkpoints from.")

    args = parser.parse_args()

    config, structure, restart = load_parameters(args.config, args.resume or args.test !='none', args.model_name)
    config["general"]["restart"] = restart
    config["general"]["last_ckpt"] = args.last_ckpt
    config["general"]["test"] = args.test
    if args.target_version is not None:
        config["general"]["target_version"] = args.target_version


    config = check_ckpt_paths(config)

    devices = ",".join([str(g) for g in args.gpus]) if isinstance(args.gpus,list) else str(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    args.gpus = [i for i,_ in enumerate(args.gpus)]



    # if len(args.gpus) == 1:
    #     gpus = int(args.gpus[0])
    # else:
    #     gpus = args.gpus

    experiment = select_experiment(config, structure, args.gpus)

    # start selected experiment

    if args.test != 'none':
        experiment.test()
    else:
        experiment.train()