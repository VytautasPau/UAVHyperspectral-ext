import torch
import torch.nn


def loss_simmilarity(x, y=None, dim=0):
    # minimize average magnitude of cosine similarity
    # performe pairwise endmember simmilarity calculation
    # input shape: [num of classes, bands]
    x = x.squeeze()
    nc = x.shape[0]
    x = x.T
    # Generate the endmember matrix and a 90 degree rotated identical matrix
    x_row = x[:, None, :].expand(-1, nc, -1)
    x_col = x[:, :, None].expand(-1, -1, nc)
    sim = nn.functional.cosine_similarity(x_col, x_row, dim=dim)
    return sim.abs().mean()

loss_sim = utils.loss_simmilarity(endm) 
# scale similarity loss
loss_sim /= 10**(int(torch.log10(loss_sim)) - int(torch.log10(loss_re)) + 1)
