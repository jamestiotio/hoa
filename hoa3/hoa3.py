# 10.008 Hands-on Activity 3 Verifier Script
# Check validity of inverse-square relationship between intensity and height
# Also check validity of black-body radiation relationship between spectral irradiance and wavelength
# Created by James Raphael Tiovalen (2020)

# Import scientific libraries
import numpy as np
from scipy.optimize import curve_fit
from sklearn import metrics


# Define inverse-square law function
def inverse_square(h, k):
    return k / (h ** 2)


if __name__ == "__main__":
    heights = np.array([6.5, 9.0, 12.0, 17.0]).astype(np.float64)

    red_intensities = np.array(
        [0.8319219036, 0.4896690263, 0.249988924, 0.1288602701]
    ).astype(np.float64) # λ = 633 nm
    green_intensities = np.array(
        [0.410783097, 0.2675019527, 0.1692426359, 0.0913581608]
    ).astype(np.float64) # λ = 524 nm
    blue_intensities = np.array(
        [1.002523336, 0.5135877745, 0.398544113, 0.1931090032]
    ).astype(np.float64) # λ = 460 nm

    for intensity in [red_intensities, green_intensities, blue_intensities]:
        popt, pcov = curve_fit(inverse_square, heights, intensity, method="lm")
        r_squared = round(
            metrics.r2_score(inverse_square(heights, *popt), intensity), 3
        )

        print(f"Equation: I = {popt[0]} / h^2")
        print(f"R^2 = {r_squared}")
