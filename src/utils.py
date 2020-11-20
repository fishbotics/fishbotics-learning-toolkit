def send_batch_to_device(batch, keys, device):
    device_batch = {}
    for key in keys:
        device_batch[key] = batch[key].to(device)
    return device_batch
