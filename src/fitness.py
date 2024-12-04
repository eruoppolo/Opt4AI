from skimage.color import rgb2lab
import numpy as np


def calculate_fitness(individual: np.ndarray, target:np.ndarray , type="deltaE")-> float:
    """
    Computes fitness function selected between Delta E (CIE1976) and MSE.
    Args:
        individual (np.ndarray): Inidividual rapresented as RGB image (NxMx3).
        target (np.ndarray): Target rapresented as RGB image (NxMx3).
    Returns:
        float: Average Delta E or MSE between individual and target images (the lower the better).
    """    
    # Delta E
    if type == "deltaE":
        ind = rgb2lab(individual / 255)
        tar = rgb2lab(target / 255)
        delta_e = np.sqrt(np.sum(np.square(tar - ind), axis=-1))
        fit = np.mean(delta_e)
    # Mean squared error
    elif type == "mse":
        # Compute the squared error between the individual and the target
        error = np.subtract(individual.astype(np.float32), target.astype(np.float32))
        squared_error = np.square(error)
        # Compute the mean squared error
        fit = np.mean(squared_error)
    else:
        raise ValueError(f"Unknown fitness type: {type}")
    return fit