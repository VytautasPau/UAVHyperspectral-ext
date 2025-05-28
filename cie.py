import colout
import numpy as np


def x_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 0]

def y_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 1]

def z_cmf(band):
    return colour.colorimetry.wavelength_to_XYZ(band)[:, 2]

def get_bands(wavelength):
    return wavelength[np.logical_and(380 < wavelength, wavelength < 780)]
