import torch

def copy_network(network_to, network_from, config, force_cpu=False):
    """ Copies networks and set them to device or cpu.
    Args:
        networks_to: Netwoks to which we want to copy (destination).
        networks_from: Networks from which we want to copy (source). These
            networks will be changed.
        force_cpu: Boolean, if True the desitnation nateworks will be placed on
            the cpu.  If not the current device will be used.
    """
    network_from_dict = network_from.state_dict()
    if force_cpu:
        for key, val in network_from_dict.items():
            network_from_dict[key] = val.cpu()
    else:
        move_to_cuda(config)
    network_to.load_state_dict(network_from_dict)
    if force_cpu:
        network_to = network_to.to('cpu')
    else:
        network_to.to(ptu.device)
    network_to.eval()
    return network_to

def batch_to_torch(batch, device='cpu'):
    new_dict = {}
    for key,value in batch.items():
        new_dict[key] = torch.from_numpy(batch[key]).to(dtype=torch.float32, device=device)
    return new_dict
