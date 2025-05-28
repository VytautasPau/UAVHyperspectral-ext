import numpy as np
from cie import x_cmf, z_cmf, y_cmf


def integrate(data, bands, normalize=False):
    X = np.trapz(data * x_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = (X - X.min()) / (X.max() - X.min())
    else:
        image = X
    X = None
    Y = np.trapz(data * y_cmf(bands) * 1.2, bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Y - Y.min()) / (Y.max() - Y.min())))
    else:
        image = np.vstack((image, Y))
    Y = None
    Z = np.trapz(data * z_cmf(bands), bands.reshape((1, -1)), axis=1)
    if normalize:
        image = np.vstack((image, (Z - Z.min()) / (Z.max() - Z.min())))
    else:
        image = np.vstack((image, Z))
    Z = None
    return image.T
