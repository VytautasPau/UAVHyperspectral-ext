import numpy as np


def reflectance(data, calibration_spectra,
                reflection_percentages, wl_axis=-1,
                method="average", calibration_curve_degree=1):
    """
    Calculate cube reflectance from given DRS
    (Diffuse Reflectance Standard) spectra.

    Parameters
    calibration_spectra : np.array shape -> (N, WL), (N, M, WL)
        input of calibration spectra in singular array
    reflectance_percentages : np.array shape -> (N), (N, WL)
        the reflectance percentager for each of calibration spectra.
    wl_axis : int
        which calibration spectra axis is the wavelenght axis

    """
    shp1 = calibration_spectra.shape
    # get reflectance calibration functions for each wavelength
    calib = []
    for wl in range(shp1[wl_axis]):
        cal_spectra = np.take(calibration_spectra, wl, axis=wl_axis)
        z = np.polyfit(cal_spectra[:, wl],
            reflection_percentages[:, wl], calibration_curve_degree)
        calib.append(z.copy())
    calib = np.array(calib)

    # calibrate data
    for wl in range(shp1[wl_axis]):
        tmp = data[..., wl]
        data[..., wl] = tmp * calib[wl, 0] + calib[wl, 1]
