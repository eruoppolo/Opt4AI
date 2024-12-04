import random
import numpy as np 

def multi_point_crossover_horizontal(parent1 : np.ndarray , parent2 : np.ndarray, num_points: int =3) -> np.ndarray:
    """
    Horizontal bands crossover with multiple points.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
        num_points (int): Number of crossover points.
    Returns:
        np.ndarray: New child individual.
    """
    h, w, _ = parent1.shape
    points = sorted(np.random.randint(0, h, size=num_points))

    points.insert(0, 0)  
    if num_points % 2 == 1:
        points.append(h)
    
    child = parent1.copy()

    for i in range(len(points) - 1):
        if i % 2 == 1:
            child[points[i]:points[i + 1], :, :] = parent2[points[i]:points[i + 1], :, :]

    return child

def multi_point_crossover_vertical(parent1 : np.ndarray , parent2 : np.ndarray , num_points=3) -> np.ndarray:
    """
    Vertical bands crossover with multiple points.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
        num_points (int): Number of crossover points.
    Returns:
        np.ndarray: New child individual.
    """
    h, w, _ = parent1.shape
    points = sorted(np.random.randint(0, w, size=num_points))
    points.insert(0, 0)
    
    if num_points % 2 == 1:
        points.append(w)
    
    child = parent1.copy()

    for i in range(len(points)-1):
        if i % 2 == 1:
            child[:, points[i]:points[i+1]] = parent2[:, points[i]:points[i+1]]

    return child

def vertical_crossover(parent1 : np.ndarray , parent2 : np.ndarray) -> np.ndarray:
    """
    Vertical half crossover.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
    Returns:
        np.ndarray: New child individual.
    """
    h, w, c = parent1.shape
    split = w // 2
    child = np.hstack((parent1[:, :split], parent2[:, split:]))
    return child

def horizontal_crossover(parent1 : np.ndarray , parent2 : np.ndarray) -> np.ndarray:
    """
    Horizontal half crossover.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
    Returns:
        np.ndarray: New child individual.
    """
    h, w, c = parent1.shape
    split = h // 2
    child = np.vstack((parent1[:split, :], parent2[split:, :]))
    return child

def diagonal_crossover(parent1 : np.ndarray , parent2 : np.ndarray) -> np.ndarray:
    """
    Diagonal half crossover.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
    Returns:
        np.ndarray: New child individual.
    """
    h, w, c = parent1.shape
    if random.random() < 0.5:
        mask = np.triu(np.ones((h, w), dtype=bool))  # Upper triangular
    else:
        mask = np.tril(np.ones((h, w), dtype=bool))  # Lower triangular
    child = np.where(mask[..., None], parent1, parent2)
    return child

def blend_crossover(parent1 : np.ndarray , parent2 : np.ndarray)-> np.ndarray:
    """
    Blending crossover.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
    Returns:
        np.ndarray: New child individual.
    """
    alpha = np.random.random()  # Weight for blending
    child = (alpha * parent1 + (1 - alpha) * parent2).astype(parent1.dtype)
    return child

def crossover(parent1 : np.ndarray , parent2 : np.ndarray, method : str = None) -> np.ndarray:
    """
    Performs crossover between two parent individuals by using one of the crossover methods.
    Args:
        parent1 (np.ndarray): First parent.
        parent2 (np.ndarray): Second parent.
        method (str): Crossover method ("half", "partial", "blending").
    Returns:
        np.ndarray: New child individual.
    """
    if method is None:
        method = random.choice(["half", "partial", "blending"])
    # Ensure dimensions match
    if parent1.shape != parent2.shape:
        raise ValueError(f"Parent shapes must match for crossover. Got {parent1.shape} and {parent2.shape}")

    if method == "half":  # Half of each parent
        coin = random.randint(0,2)
        if coin == 0: # vertical
            child = vertical_crossover(parent1, parent2)
        elif coin == 1: # horizontal
            child = horizontal_crossover(parent1, parent2)
        else: # diagonal
            child = diagonal_crossover(parent1, parent2)
    
    elif method == "partial":
        n_points = random.randint(2,5)
        if np.random.random() < 0.5: # vertical bands
            child = multi_point_crossover_vertical(parent1, parent2, num_points=n_points)
        else: # horizontal bands
            child = multi_point_crossover_horizontal(parent1, parent2, num_points=n_points)

    elif method == "blending":
        child = blend_crossover(parent1, parent2)

    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return child

def mutate(individual : np.ndarray , palette_size : int, mutation_rate=0.01) -> np.ndarray:
    """
    Applies mutation to an individual by randomly changing pixels.
    Args:
        individual (np.ndarray): Individual to mutate (3D array for RGB image).
        palette_size (int): Size of the palette.
        mutation_rate (float): Probability of mutation per pixel.
    Returns:
        np.ndarray: Mutated individual.
    """
    # Generate a mask for which pixels will mutate
    mask = np.random.random(individual.shape[:2]) < mutation_rate  # 2D mask for the image shape
    # Apply random changes to RGB values only for selected pixels
    random_mutation = np.random.randint(-30, 31, size=individual.shape, dtype=np.int16)  # Random RGB changes
    mutated_individual = individual.astype(np.int16) + (random_mutation * mask[..., None])
    # Clip values to the valid range and return as original dtype
    mutated_individual = np.clip(mutated_individual, 0, palette_size - 1).astype(individual.dtype)
    return mutated_individual

def tournament_selection(population : list, fitness_scores : list, k=3)-> np.ndarray:
    """
    Selects a parent using tournament selection.
    Args:
        population (list): Population of individuals.
        fitness_scores (list): Fitness scores for the population.
        k (int): Tournament size.
    Returns:
        np.ndarray: Selected parent.
    """
    selected = random.sample(list(zip(population, fitness_scores)), k)
    return min(selected, key=lambda x: x[1])[0]

def update_mutation_rate(min_mutation_rate : float, rate : float, decay : float, generation : int) -> float:
    """
    Updates the mutation rate using an exponential decay.
    Args:
        min_mutation_rate (float): Minimum mutation rate.
        rate (float): Initial mutation rate.
        decay (float): Decay rate.
        generation (int): Current generation.
    Returns:
        float: Updated mutation rate.
    """
    update = rate * np.exp(-decay * generation)
    return max(min_mutation_rate, update)