from cerberus import Validator

_job_configuration = {
    'data_directory': {'type': 'string', 'required': True},
    'model_type': {'type': 'string', 'required': True},
    'loss_function': {'type': 'string', 'required': True},
    'train_data_loader_type': {'type': 'string', 'required': True},
    'test_data_loader_type': {'type': 'string', 'required': False},
    'eval_data_loader_type': {'type': 'string', 'required': False},
    'epochs': {'type': 'integer', 'required': True},
    'bootstrap_model_file': {'type': 'string', 'required': False},
    'batch_size': {'type': 'integer', 'required': True},
    'model_parameters': {'type': 'dict', 'required': False},
    'data_loader_parameters': {'type': 'dict', 'required': False},
    'description': {'type': 'string', 'required': True},
    'checkpoint_interval': {'type': 'integer', 'required': True},
    'multigpu_parameters': {
        'type': 'dict',
        'required': False,
        'schema': {
            'ngpus_per_node': {'type': 'integer', 'required': True},
            'num_nodes': {'type': 'integer', 'required': True},
            'node_idx': {'type': 'integer', 'required': True},
            'backend': {'type': 'string', 'required': True},
            'url': {'type': 'string', 'required': True},
        },
    }
}

JobConfigurationValidator = Validator(_job_configuration, allow_unknown=True)
