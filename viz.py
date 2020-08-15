import matplotlib.pyplot as plt
import numpy as np


def prep_ims_for_imshow(ims):
    rank = len(ims.shape)
    assert rank in [3, 4]
    if rank == 3:
        ims = np.expand_dims(ims, 1)
    assert ims.shape[1] in [1, 3]  # greyscale or rgb
    bs, ch, height, width = ims.shape
    nrow = ncol = np.ceil(np.sqrt(bs)).astype(int)
    ims = np.transpose(ims, [0, 2, 3, 1])\
            .reshape(nrow, ncol, height, width, ch)
    ims = np.transpose(ims, [0, 2, 1, 3, 4])\
            .reshape(nrow*height, ncol*width, ch)
    if ch == 1:
        ims.squeeze(2)
    return ims

    if len(ims.shape) == 4:
        assert ims.shape[1]


def discrete_cmap(N, base_cmap='Set2'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    cmap = base(np.linspace(0, 1, N))
    return cmap
