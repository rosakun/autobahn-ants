import random
import itertools

def initialise_uniform_pheromone_map(distance_matrix):
    n = 0
    total = 0

    # Calculate the uniform value given the distances in distance matrix
    for row in distance_matrix:
        zeroflag = False
        for item in row:
            if zeroflag:
                n += 1
                total += item
            elif not zeroflag:
                if item == 0:
                    zeroflag = True
                else:
                    pass
    
    L = total/n
    uniform_value = 1/n*L

    # Populate the pheromone matrix with the uniform value
    pheromone_matrix = [[uniform_value if i != j else 0 for j in range(len(distance_matrix))] for i in range(len(distance_matrix))]
    return pheromone_matrix


def initialise_superhighway_pheromone_map(distance_matrix, bonus):
    uniformmatrix = initialise_uniform_pheromone_map(distance_matrix)
    big_cities = [0,1,2,3]
    for i,j in itertools.product(big_cities,big_cities):
        if i!=j:
            uniformmatrix[i][j] += bonus
            uniformmatrix[j][i] += bonus
    return uniformmatrix


def generate_random_pheromone_map(distance_matrix):

    pheromone_matrix = [[random.uniform(1.0,50.0) if i != j else 0 for j in range(len(distance_matrix))] for i in range(len(distance_matrix))]
    return pheromone_matrix





    