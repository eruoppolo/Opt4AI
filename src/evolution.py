import numpy as np
import random
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from operators import tournament_selection, crossover, mutate, update_mutation_rate, multi_point_crossover_horizontal, multi_point_crossover_vertical, blend_crossover, vertical_crossover, horizontal_crossover, diagonal_crossover
from fitness import calculate_fitness
from utilities import load_image, display_image, log, export_data
from generation import generate_individual, initialize_population

def evolve(image_path : str = None, 
           pop_size : int = 150, 
           palette_size : int = 255, 
           generations : int = 4000, 
           mutation_rate : float = 0.05,
           tournament_size : int = 5,
           replacement : int = 20,
           min_mutation_rate : float = 0.001,
           max_mutation_rate : float = 0.25,
           stagnation_limit : int = 10,
           exit_limit : int = 50,
           n: int = 128,
           fit: str = "deltaE"):
    """
    Evolutionary loop.
    Args:
        image_path (str): Path to the target image.
        pop_size (int): Population size.
        palette_size (int): Size of the color palette.
        generations (int): Number of generations.
        mutation_rate (float): Initial mutation rate.
        tournament_size (int): Number of individuals in a tournament.
        replacement (int): Number of individuals to replace in the population.
        min_mutation_rate (float): Minimum mutation rate.
        max_mutation_rate (float): Maximum mutation rate.
        stagnation_limit (int): Generations without improvement before intervention.
        n (int): Resolution of the images (n x n).
        fit (str): Fitness function to use.
    Returns:
        np.array: The best individual (evolved image).
        list: Best fitness scores over generations.
    """
    resolution = (n, n)

    data = {'iteration':[], 'fitness':[], 'im_size':[]} # Data to be saved

    target_image = load_image(image_path, resolution)

    log("Initial population initialization ...")

    population = initialize_population(pop_size, resolution, palette_size)

    log("Population initialized.")

    fitness_scores = []

    no_improvement_count = 0  # Track stagnation
    stagnation_count = 0  # Track stagnation

    for generation in range(generations):
    

        if len(fitness_scores) == 0:

            with ThreadPoolExecutor() as executor:
                fitnesses = list(executor.map(
                    lambda ind: calculate_fitness(ind, target_image, fit), 
                    population))


        best_fitness = min(fitnesses)   
        fitness_scores.append(best_fitness)
        best_individual = population[np.argmin(fitnesses)]

        selection = [tournament_selection(population, fitnesses, tournament_size) for _ in range(pop_size)]

        offspring = []
        offspring_fitnesses = []

        for _ in range(pop_size):
            parent1, parent2 = random.sample(selection, 2)
            child = crossover(parent1, parent2, method=random.choice(["half", "partial", "blending"]))
            mutated_child = mutate(child, palette_size, mutation_rate)
            offspring.append(mutated_child)
        

        with ThreadPoolExecutor() as executor:
            fitnesses_af = list(executor.map(
                lambda ind: calculate_fitness(ind, target_image, fit), 
                offspring))
        
        
        new_offspring = offspring
        offspring_fitnesses = fitnesses_af

        indices = np.argsort(offspring_fitnesses)[:replacement]
        best_offspring = [new_offspring[i] for i in indices]
        best_of_fitnesses = [offspring_fitnesses[i] for i in indices]

        worst_indices = np.argsort(fitnesses_af)[-replacement:]

        for i, idx in enumerate(worst_indices):
            population[idx] = best_offspring[i]
            fitnesses[idx] = best_of_fitnesses[i]
        

        # Check for improvement
        if len(fitness_scores)>5 and (best_fitness < fitness_scores[-5] or best_fitness > fitness_scores[-1]): ############## look at this
            no_improvement_count = 0  # Reset stagnation counter
            mutation_rate = update_mutation_rate(min_mutation_rate, rate = max_mutation_rate, decay=0.005, generation=generation)
        else:
            no_improvement_count += 1

        # Handle stagnation
        if no_improvement_count >= stagnation_limit:
            print(f"Stagnation detected at generation {generation+1}. Actual mutation rate: {mutation_rate:.3f}. Increasing mutation rate.")
            mutation_rate = min(mutation_rate * 1.2, max_mutation_rate)
            no_improvement_count = 0  # Reset stagnation counter
            stagnation_count += 1

        if (generation+1) >= 5000 and stagnation_count >= exit_limit:
            print(f"Stagnation limit reached. Exiting evolution loop.")
            break
        

        # Display progress
        if (generation+1) == 1 or (generation+1) == 2 or (generation+1) == 5 or (generation+1) == 10 or (generation+1) % 50 == 0 or generation == generations - 1:
            data['iteration'].append(generation)
            data['fitness'].append(best_fitness)
            data['im_size'].append(n)
        if (generation+1) == 1 or (generation+1) == 2 or (generation+1) == 5 or (generation+1) == 10 or (generation+1) == 100 or (generation+1) % 1000 == 0 or (generation+1) == generations - 1:
            display_image(n, generation, generations, best_fitness, mutation_rate, best_individual, image_path)
            
    if generation < generations - 1:
        data['iteration'].append(generation)
        data['fitness'].append(best_fitness)
        data['im_size'].append(n)
        display_image(n, generation, generations, best_fitness, mutation_rate, best_individual, image_path)

    export_data(data, n)

    return best_individual, fitness_scores