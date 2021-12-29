import os
import torch
import torch.multiprocessing as mp

from utils.io import safe_printout


def load_geo_dist(geo_dist_path, retain_geo_dist=False):
    """
    Load pre-computed geometric distance table.
    @param geo_dist_path:    Path to the geometric distance dict.
    @param retain_geo_dist:  Flag to retain the original geo-distance tensor.
    """
    assert os.path.exists(geo_dist_path)
    safe_printout('Load geometric distance from {:s}...'.format(geo_dist_path))
    geo_dist_dict = torch.load(geo_dist_path)
    if retain_geo_dist:
        geo_dist = geo_dist_dict['geo_dist'].cpu()  # raw distance table, debug purpose
    else:
        geo_dist = None
    sim_data = geo_dist_dict['sim_data'].cpu()
    dict_name_to_idx = geo_dist_dict['dict_name_to_idx']
    dict_idx_to_name = geo_dist_dict['dict_idx_to_name']
    hyper_params = geo_dist_dict['hyper_params']
    dict_supercon = geo_dist_dict['dict_supercon']
    feasible_anchor = geo_dist_dict['feasible_anchor']

    # sanity check for index-based synthetic data flag
    for i in range(len(sim_data)):
        if i < sim_data.sum().item():
            assert sim_data[i]
        else:
            assert ~sim_data[i]

    safe_printout('%d / %d samples are feasible anchors after pre-screening.' % (len(feasible_anchor), len(sim_data)))

    return geo_dist, sim_data, dict_name_to_idx, dict_idx_to_name, hyper_params, dict_supercon, feasible_anchor

def get_supercon_dataloader(trainset_supercon, shuffle=True):
    """
    Wrapper to reset the supercon dataloder.
    """
    sampler_supercon = trainset_supercon.get_supercon_sampler(shuffle=shuffle)
    loader_supercon = torch.utils.data.DataLoader(trainset_supercon, batch_sampler=sampler_supercon,
                                                  num_workers=mp.cpu_count() // 2,
                                                  pin_memory=True, collate_fn=trainset_supercon.batch_resize)
    return loader_supercon