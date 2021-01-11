import sys
import pathlib
import argparse
import torch
import git
from torch import optim
import torch.nn.functional as F  # NOQA This is necessary for dynamic loading
from termcolor import colored
import yaml
from src.configuration import Configuration

project_directory = pathlib.Path(__file__).parent.absolute()
import_modules = list((project_directory / 'submodules').iterdir())
import_modules = [str(p) for p in import_modules]
import_modules.append(str(project_directory))
sys.path.extend(import_modules)

from src import (
    train,
    schema,
    data_loader,  # NOQA This is necessary for dynamic loading
    model,  # NOQA This is necessary for dynamic loading
    evaluation,  # NOQA This is necessary for dynamic loading
    utils,
)


def create_model(config):
    model_type = eval(f"model.{config.model_type}")
    mdl = model_type(**config.model_parameters)

    if 'bootstrap_model_file' in config:
        # TODO fix this
        assert False, "CANNOT YET LOAD MODEL IN DISTRIBUTED JOBS"
        checkpoint = torch.load(config.bootstrap_model_file)
        mdl.load_state_dict(checkpoint['model_state_dict'])
        # opt.load_state_dict(checkpoint['optimizer_state_dict'])
        message = (
            f"Loaded model from file "
            f"{colored(config.bootstrap_model_file, 'green')} \n"
            f"Epoch: {colored(checkpoint.get('epoch', None), 'blue')} \n"
            f"Train Loss: {colored(checkpoint.get('train_loss', None), 'red')} \n"
            f"Test Loss: {colored(checkpoint.get('test_loss', None), 'red')} \n"
            f"Eval Loss: {colored(checkpoint.get('evaluation_loss', None), 'red')} \n"
        )
        print(message)

    # Use this when debugging to ensure that GPU is working
    # print(next(mdl.parameters()).device)
    return mdl


def get_dataloaders(config):
    train_loader = eval(f"data_loader.{config.train_data_loader_type}")
    test_loader = eval(f"data_loader.{config.test_data_loader_type}")
    eval_loader = eval(f"data_loader.{config.eval_data_loader_type}")
    train_dl = train_loader.get_dataloader(
        directory=config.data_directory,
        batch_size=config.batch_size,
        acquire_test_dataset=False,
        shuffle=True,
        mini=config.mini,
        **config.data_loader_parameters.all,
        **config.data_loader_parameters.train,
    )
    test_dl = test_loader.get_dataloader(
        directory=config.data_directory,
        batch_size=config.batch_size,
        acquire_test_dataset=True,
        shuffle=False,
        mini=config.mini,
        **config.data_loader_parameters.all,
        **config.data_loader_parameters.test,
    )
    eval_dl = eval_loader.get_dataloader(
        directory=config.data_directory,
        batch_size=config.batch_size,
        acquire_test_dataset=True,
        shuffle=False,
        mini=config.mini,
        **config.data_loader_parameters.all,
        **config.data_loader_parameters.eval,
    )
    return train_dl, test_dl, eval_dl


def run(config):
    loss_function = eval(config.loss_function)
    train_dl, test_dl, eval_dl = get_dataloaders(config)

    if not config.cpu and torch.cuda.is_available():
        print("Training with GPU")
        device = torch.device("cuda")
    else:
        print("Training with CPU")
        device = torch.device("cpu")

    mdl = create_model(config)

    trainer = train.Trainer(
        mdl,
        loss_function,
        'multigpu_parameters' in config,
    )
    trainer.fit(train_dl, test_dl, eval_dl, config)


def check_for_uncommitted_changes():
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty(untracked_files=True):
        raise Exception(
            'Uncommitted changes found in local git repo. '
            'Commit all changes before running experiments '
            'to maintain reproducibility.'
        )


def confirm_allow_dirty_repo():
    """
    Ask user to enter Y or N (case-insensitive).
    Code is borrowed from here: https://gist.github.com/gurunars/4470c97c916e7b3c4731469c69671d06
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    message = (
        "you have set --allow-dirty-repo which will run an experiment with"
        " uncommitted changes. This will forcibly disable logging, as logged"
        " experiments should be reproducible. Do you wish to continue? [y/n]"
    )
    while answer not in ["y", "yes", "n", "no"]:
        answer = input(colored("Warning:", "red") + message).lower()
    return answer == "y" or answer == "yes"

def edit_config(config, find_replace, modify_pairs, delete_keys):
    if find_replace:
        find_replace_dict = {}
        assert len(find_replace) % 2 == 0, "Replacements must be given as pairs"
        for i in range(len(find_replace) // 2):
            find_replace_dict[find_replace[2 * i]] = find_replace[2 * i + 1]
        config = utils.find_replace_config_strings(config, find_replace_dict)
    if modify_pairs:
        assert len(modify_pairs) % 2 == 0, "Replacements must be given as pairs"
        for i in range(len(modify_pairs) // 2):
            key = modify_pairs[2 * i]
            value = modify_pairs[2 * i + 1]
            config[key] = type(config[key])(value)
    if delete_keys:
        config = {k: v for k, v in config.items() if k not in delete_keys}
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_config', type=str)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--mini', action='store_true', help="Use a tiny dataset (for testing code)")
    parser.add_argument('--no-logging', action='store_true', help="Don't log (just for testing)")
    parser.add_argument('--allow-dirty-repo', action='store_true', help="Force reload the dataset")
    parser.add_argument('--y', action='store_true', help="Autoyes for any prompts")
    parser.add_argument('--find-replace', nargs='+', help="Strings to find in the configuration and modify (typically paths that need to change for running a local test without changing the runconfig)")
    parser.add_argument('--delete', nargs='+', help="Keys to delete from runconfig (for example, multigpu)")
    parser.add_argument('--modify', nargs='+', help="Keys to directly modify in the runconfig")

    args = parser.parse_args()
    if args.allow_dirty_repo and (args.y or confirm_allow_dirty_repo()):
        args.no_logging = True
    else:
        check_for_uncommitted_changes()
    with open(args.yaml_config) as f:
        configuration = yaml.safe_load(f)
    for field in configuration:
        if field not in schema._job_configuration:
            print(colored("Unrecognized configuration field--ignoring", "red"), f": {field}")
    configuration = {
        **{field: configuration[field] for field in schema._job_configuration.keys() if field in configuration},
        **vars(args),
    }
    configuration = edit_config(configuration, args.find_replace, args.modify, args.delete)

    valid_schema = schema.JobConfigurationValidator(configuration)
    if valid_schema is False:
        print("Job schema found to be invalid with errors as follows")
        print(schema.JobConfigurationValidator.errors)
        raise Exception("Schema invalid. Fix and rerun")
    run(Configuration(configuration))
