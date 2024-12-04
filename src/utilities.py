import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from pandas import DataFrame


def load_image(image_path:str , target_resolution:tuple) -> np.ndarray:
    """
    Loads the target image and resizes it to the target resolution.
    Args:
        image_path (str): Path to the high-resolution target image.
        target_resolution (tuple): Desired resolution (width, height).
    Returns:
        np.ndarray: Resized target image as a NumPy array.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_resolution, Image.Resampling.LANCZOS)
    return np.array(img)



def display_image(n:int , generation:int , generations:int , best_fitness:float , mutation_rate:float , best_individual:np.ndarray , target:str) -> None:
    """
    Displays a grid with the best individual and the target image.
    Args:
        n (int): Identifier for the run.
        generation (int): Current generation index.
        generations (int): Total number of generations.
        best_fitness (float): Best fitness value.
        mutation_rate (float): Mutation rate.
        best_individual (np.ndarray): Best individual image.
        target (str): path to target image for comparison.
    """
    print(f"Generation {generation+1}/{generations}. Best fitness: {best_fitness:.3f}. Mutation rate: {mutation_rate:.3f}, time [{time.strftime('%H:%M:%S')}]")
    
    # Create a figure for the grid
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the best individual
    axes[0].imshow(best_individual)
    axes[0].set_title("Best Individual")
    axes[0].axis('off')
    
    # Display the target image
    target_image = Image.open(target).convert("RGB")
    axes[1].imshow(target_image)
    axes[1].set_title("Target Image")
    axes[1].axis('off')
    
    # Add a suptitle with the generation number
    fig.suptitle(f"Resolution {n}x{n}, Generation {generation+1}, Fitness = {best_fitness:.2f}", fontsize=16)
    
    # Create a directory to save the images if it does not exist
    repo = f"../evolution_{n}"
    if not os.path.exists(repo):
        os.makedirs(repo)
    
    im_repo = f"{repo}/images_{n}"
    if not os.path.exists(im_repo):
        os.makedirs(im_repo)
    
    # Format generation number to 5 digits
    gen_str = str(generation + 1).zfill(5)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the suptitle
    plt.savefig(f"{im_repo}/gen_{gen_str}.png")
    plt.show()

def export_data(data:dict , n:int) -> None:
    """
    Exports the data dictionary to a CSV file.
    Args:
        data (dict): Data dictionary to be exported.
        n (int): Size identifier for the run.
    """
    repo = f"../evolution_{n}"
    if not os.path.exists(repo):
        os.makedirs(repo)
    data_df = DataFrame(data)
    data_df.to_csv(f"{repo}/data_size_{n}.csv")

    print("Data exported successfully.")

def log(message:str) -> None:
    """
    Logs a message with the current time.
    Args:
        message (str): Message to be logged.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}")
    

