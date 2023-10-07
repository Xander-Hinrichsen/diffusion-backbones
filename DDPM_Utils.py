import torch
import torch.nn as nn

##for vizualizing 
def unroll_samples(sample_list):
    """ Visualizes DDPM sampling, at different timesteps
        Use the ddpm model sampling method, to sample at different timesteps
        input is a list of n (b,3,h,w)
        output is (3, h*b, n*w) -> an image of generated images
    """
    #get initial stuffs
    init_lists = []
    for i in range(len(sample_list)):
        init_lists.append(sample_list[i][0])
    #iterate over batch
    for i in range(len(sample_list)):
        for j in range(1, sample_list[0].shape[0]):
            init_lists[i] = torch.cat([init_lists[i], sample_list[i][j]], dim=-2)
    
    ret = init_lists[0]
    for i in range(1, len(sample_list)):
        ret = torch.cat([ret, init_lists[i]], dim=-1)
    return ret
