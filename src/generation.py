import numpy as np
def generate_individual(resolution:tuple , palette_size:int) -> np.ndarray:
    """
    Generates a single individual by creating a monocolor RGB image and modifying
    some pixels randomly.
    Args:
        resolution (tuple): Resolution (width, height) of the image.
        palette_size (int): Maximum intensity value for each color channel (e.g., 255 for 8-bit images).
    Returns:
        np.ndarray: A 3D array representing the individual's image with RGB channels.
    """
    # Step 1: Create a monocolor RGB image with a random color
    base_color = np.random.randint(0, palette_size, size=(3,))  # RGB base color
    image = np.full((*resolution, 3), base_color, dtype=np.uint8)  # RGB image

    # Step 2: Choose a random number of pixels to modify
    num_pixels_to_modify = np.random.randint(np.prod(resolution)//2, np.prod(resolution))  # e.g., from 50% up to 100% of pixels

    # Step 3: Change the selected pixels to random RGB colors
    for _ in range(num_pixels_to_modify):
        # Randomly select a pixel
        x = np.random.randint(0, resolution[0])
        y = np.random.randint(0, resolution[1])
        # Change its color to a random RGB value
        image[x, y] = np.random.randint(0, palette_size, size=(3,))
    
    return image

def initialize_population(pop_size: int, resolution: tuple, palette_size: int) -> list:
    """
    Initializes the population with a specified number of individuals.
    Args:
        pop_size (int): Number of individuals in the population.
        resolution (tuple): Resolution (width, height) of the images.
        palette_size (int): Number of colors in the palette.
    Returns:
        list: Population of individuals (2D arrays of indices).
    """
    population = [generate_individual(resolution, palette_size) for _ in range(pop_size)]
    return population

