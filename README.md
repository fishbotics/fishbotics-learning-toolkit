# Fishbotics Learning Toolkit
This repository is a template for learning-based projects with managed, reproducible experiments that can run in either a single- or multi-GPU environment. 
Multi-computer (also sometimes called multi-node) environments are not currently supported.

If things go my way, I will make this into a lightweight library instead of a template, which should make it easier to use. For now, the best way to use this library would be to fork it, and then commit on top. If you make any significant changes to the core functionality, please push them upstream!


## Usage

This system is built around reproducibility. As with any other learning framework, you will create a model and a dataloader and kick off the job. When you kick off the job, you will start it with a config file. 
The config file will hold all the metadata for your dataloader (i.e. what type of dataloader and the specific parameters) and for your model 
(again, what type and the parameters). You then commit the config file (and any code changes) and start the job. 

When you run the job, your config will be saved alongside your logs and model checkpoints. The saved config file will have the corresponding git commit attached to it. 
This way, you will always be able to go back in time to rerun an experiment. 
The system will prevent you from running a job without committing your changes first to ensure reproducibility. 
If you're quickly testing things, you can run without committing, but nothing will be logged (model, config, or results).

### Configs
This system is based around reproducibility and therefore your config file will hold all of the info you need to start a job. You can see the currently supported config structure in [`sample_runconfig.yaml`](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/sample_runconfig.yaml). 
When running a job, this config will serve both as a way to reproduce the results, but also a reminder of what you did in this particular experiment. 
The `description` field is meant solely for your future self's benefit as a way to remind you of what the purpose and/or changes of that particular job were.

If you modify the template and want to add additional fields into the config. You can modify the schema in [`schema.py`](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/schema.py).

### Data

In the config file, you will see a `data_directory` key. This points to where your data lives on the local machine 
(or within the docker container if you are running your job using docker). The data directory should have a subfolder called `train/` and a subfolder called `test/`.

When writing your dataloader, you can build off of the [`FishboticsDataloader`](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/data_loader.py#L41) class. 
There are a few functions in that class which are necessary for the template to work. They should be fairly well self-documented. 
The one important thing to keep in mind with this file, as well as other files, is that you want to put your new dataloader in that same [file](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/data_loader.py).

The templated dataset class has a flag called `mini`. For fast iteration on your pipeline, you can use this flag to only load a small subset (say, 100 elements) of your dataset. This makes coding and testing much faster, especially if you want to make sure the whole training pipeline works.

### Model

This was originally designed for robot control, where you might train and test with a computationally cheap(-ish) model, which you then modify for a smaller-scale full evaluation.
Practically speaking, this means that training has three phases, where one is much much more expensive. And, the [model class](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/model.py) has a flag (called `rollout_mode`, which changes the behavior of the model 
to do the more expensive thing, such as doing a full motion rollout instead of a truncated rollout.

If you don't care about having three phases in the model, you can disable the more expensive phase by commenting out _all_ calls to `Trainer.evaluation_loop` ([an example](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/train.py#L109)) and the resulting `eval_loss` ([an example](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/train.py#L124))
 in [`train.py`](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/src/train.py).
I realize this is pretty yucky, and will make it a little more user friendly at some point if other people decide to use this library.

### Multithreading
In the `sample_runconfig.yaml`, [the multigpu arguments are commented out](https://github.com/fishbotics/fishbotics-learning-toolkit/blob/main/sample_runconfig.yaml#L18). 
In order to train in a multigpu environment, you can just uncomment these out. Be sure to set the number of GPUs correctly! There is some magic in the dataloader class which will turn your dataloader into a multigpu dataloader. 

Currently, model checkpoint loading is not supported with multi-gpu training. Again, if this proves useful to anyone, I can add it (or you can and submit a PR!).

### Runtime Flags
In addition to the config files, there are some runtime flags which can be useful. These will also be saved in the config after running the job, so you'll always a record.

Briefly, these flags are:

- `--cpu`: Will train with CPU instead of GPU
- `'--mini`: Will set the mini flag in the dataloader, which (assuming you set it up correctly) tells the dataloader to use a mini dataset for testing
- `'--no-logging'`: Turns off all logging and checkpointing
- `'--allow-dirty-repo'`: This will allow you to use a repo without committing, but will also turn off all logging (you shouldn't be saving results if they aren't reproducible!)
- `'--y'`: The `--allow-dirty-repo` flag gives you a prompt, so this is for convenience to auto-yes the prompt.
- `'--find-replace'`: [USE WITH CAUTION] If you're running a quick test and you want to change some arbitrary string in the config file without actually changing it in the file. For example, if you want to change the data path for local test without removing the server's data path from the committed config. You use this by including a even-lengthed list of items where the 2i-th element is the thing to replace and the (2i+1)-th element is the replacement 
- `'--delete`: [USE WITH CAUTION] Similarly used for tests, a list of top-level keys you want to delete before running (such as the multi-gpu keys, which would make the job default back to single-gpu training).
- `'--modify'`: [USE WITH CAUTION] Similar to find-replace, but will replace the entire top-level config key with the new value instead of doing find-replace.

### Kicking off a job
Jobs in this system are started from the `run.py` script. In order to run the script, you would run it with:

```bash
python run.py sample_runconfig.yaml
```

I tested it with Python 3.8, but it should work with 3.7 (and likely 3.6 as well). 
