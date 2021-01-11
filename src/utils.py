def send_batch_to_device(batch, keys, device):
    device_batch = {}
    for key in keys:
        device_batch[key] = batch[key].to(device)
    return device_batch

def find_replace_config_strings(config, replacements):
    for e, v in config.items():
        if isinstance(v, str):
            orig_path = v
            for orig, new in replacements.items():
                path = v.replace(orig, new)
                if path != orig_path:
                    config[e] = path
                    break
        elif isinstance(v, dict):
            find_replace_config_strings(v, replacements)
    return config
