from utils.exercise_1 import *
from utils.get_pheromonemap import *
from matplotlib import *
from pylab import *
import random
import itertools
import timeit
import numpy as np
from scipy.sparse.csgraph import dijkstra
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib import collections  as mc

class ProblemMap():

    def __init__(self,coordinatemap, distancemap, gravity):
        self.coordinates = readFileAsMatrix(coordinatemap)
        self.distances = readFileAsMatrix(distancemap)
        self.globalpheromones = initialise_superhighway_pheromone_map(self.distances, 3)
        self.gravity = readFileAsMatrix(gravity)
        self.nodes = list(range(1,len(self.distances)+1))
        self.cities = ["Berlin", "Hamburg", "Munich", "Frankfurt am Main", "Cologne", "Stuttgart", "Düsseldorf", "Hanover", "Bremen", "Wolfsburg", "Essen", "Dortmund", "Nuremburg", "Leipzig", "Ingolstadt", "Dresden", "Bonn", "Regensburg", "Halle (Salle)", "Ludwigshafen am Rhein"]

        #Parameter values 
        self.rho = 0.9
        self.q0 = 0.5
        self.alpha = 1
        self.beta = 1
        self.generations = 1

    def _runiteration(self, plot=True, evaluate=False):
        counter = 0
        
        for _ in range(self.generations):
            solutions = []
            for i, j in itertools.product(self.nodes, self.nodes):
                if i < j:
                    solution = self._generate_ant_colony(i,j,self.gravity[i-1][j-1])
                    solutions += solution
            self._update_pheromones(solutions)
            counter += 1
            if counter % 5 == 0:
                print("Iteration", counter)

        bestsolutions = []
        for i,j in itertools.product(self.nodes, self.nodes):
            if i < j: 
                colony = AntColony(self,i,j,1)
                bestsolutions.append(self._maximumpheromonepath(i,j))  
        print("Best solutions calculated.")
        print(bestsolutions)

        if evaluate:
            evaluator = Evaluator(self, bestsolutions)
            print("Total cost of best solutions:", evaluator._cost())
            print("Total efficiency of best solutions:", evaluator._efficiency())

        if plot:  
            self._plotSolution_earth(solutions=bestsolutions,title="Best Solutions")

        print("Done.")
        return bestsolutions


    def _update_pheromones(self, solutions, delete=False):
        update_matrix = [[0 for _ in range(len(self.nodes))] for _ in range(len(self.nodes))]

        for ant in solutions:
            total_path = 0
            
            # Get the total length of the path for the ant
            for index in range(len(ant)-1):
                i = ant[index]
                j = ant[index+1]
                total_path += self._get_distance(i,j)

            ant_update = 1/total_path

            # Update the update matrix
            for index in range(len(ant)-1):
                i = ant[index]
                j = ant[index+1]
                update_matrix[i-1][j-1] += ant_update
                update_matrix[j-1][i-1] += ant_update
        
        #Decay the pheromone matrix and add the update matrix value
        for i in range(len(update_matrix)):
            for j in range(len(update_matrix)):
                if i==j:
                    pass
                else:
                    self.globalpheromones[i][j] = (1-self.rho) * self._get_globalpheromones(i+1,j+1)
                    self.globalpheromones[i][j] += update_matrix[i][j]     

    # Creates a single ant colony and produces a list of candidate solutions.
    def _generate_ant_colony(self, start, end, size):
        antcolony = AntColony(self, start, end, size)
        solutions = antcolony._generate_solutions()
        return solutions   

    def _maximumpheromonepath(self, start, end):
        solution = [start]
        ant_position = start
        while ant_position != end:
            possible_nodes = [node for node in self.nodes if node not in solution]
            pheromones = [(node,self.globalpheromones[ant_position-1][node-1]) for node in possible_nodes]
            ant_position = max(pheromones, key=lambda x: x[1])[0]
            solution.append(ant_position)
        return solution    

    # Returns the distance between nodes i and j
    def _get_distance(self,i,j):
        return self.distances[i-1][j-1]

    # Returns the pheromone level between nodes i and j
    def _get_globalpheromones(self,i,j):
        return self.globalpheromones[i-1][j-1]

    def _plotSolution_earth(self, solutions, title):
        points = self.coordinates
        distances = self.distances
        cities = self.cities
        # Create the plot with Cartopy (using Mercator projection for Germany)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection=ccrs.Mercator())  # Using Mercator projection
        
        # Set the boundaries to focus on Germany
        ax.set_extent([5.87, 15.04, 47.3, 55.1], crs=ccrs.PlateCarree())  # Longitude and Latitude bounds for Germany
        
        # Add coastlines and countries
        ax.coastlines(resolution='110m')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')  # Borders of countries
        ax.add_feature(cfeature.LAND, edgecolor='black')  # Land color
        
        # Plot the cities as points on the map
        # Swap coordinates to (longitude, latitude) for Cartopy (lat, lon)
        x, y = zip(*points)  # Unpack lon/lat from points
        lon, lat = zip(*[(y, x) for x, y in points])  # Swap lon/lat to lat/lon

        # Plot city points as red dots
        ax.scatter(lon, lat, color='red', transform=ccrs.PlateCarree())  # Scatter on the map
        
        # Annotate each city with its name
        for i, p in enumerate(zip(lon, lat)):
            ax.text(p[0] + 0.1, p[1], cities[i], transform=ccrs.PlateCarree(),
                    fontsize=9, ha='left', color='black')  # Adjust text position slightly for clarity
        """
        # Draw all possible path segments (based on the distance matrix)
        lines = []
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if distances[i][j] > 0 and i > j:
                    lines.append(((p[1], p[0]), (q[1], q[0])))  # Swap lon/lat to lat/lon for lines
        
        # Create line collection for all possible paths
        #line_collection = mc.LineCollection(lines, linewidths=0.5, color='blue', transform=ccrs.PlateCarree())
        #ax.add_collection(line_collection)
        
        # Plot the solution (red path for the given solution)
        solution_lines = []
        for solution in solutions:
            for i in range(len(solution)-1):
                solution_lines.append(((points[solution[i]-1][1], points[solution[i]-1][0]), 
                                        (points[solution[i+1]-1][1], points[solution[i+1]-1][0])))  # Swap lon/lat
        
        # Create line collection for the solution path in red
        solution_line_collection = mc.LineCollection(solution_lines, linewidths=2, color='red', transform=ccrs.PlateCarree())
        ax.add_collection(solution_line_collection)
        """
        # Add title and show the plot
        ax.set_title(title)
        plt.show()


class AntColony(ProblemMap):

    def __init__(self, problemmap, start, end, size):
        self.problemmap = problemmap
        self.start = start
        self.end = end
        self.size = int(size)

    def _generate_solutions(self):
        solutions = []
        for _ in range(self.size):
            ant = Ant(self)  # Create a new Ant instance
            solution = ant._generate_solution()  # Generate a solution
            solutions.append(solution)  # Add the solution to the list
        return solutions
            

class Ant(AntColony):

    def __init__(self, colony):
        self.colony = colony
        
    """
    This function generates a candidate solution from the ant, i.e. a potential path from the start to the end node.
    """

    def _generate_solution(self):
        ant_position = self.colony.start

        ant_path = [self.colony.start]

        while ant_position != self.colony.end:
            
            # (node, pheromone) for each node that has not been visited yet
            possible_nodes = [node for node in self.colony.problemmap.nodes if node not in ant_path]
            q = random.uniform(0,1)

            #if q < q0, choose next node by edge with largest amount of pheromones. q0 is a user-defined threshold value
            if q<self.colony.problemmap.q0:
                # Check if all second values are the same
                # If all choices have same pheromone concentration, move ant to random node
                pheromones = [(node, self.colony.problemmap._get_globalpheromones(ant_position,node)) for node in possible_nodes]
                second_values = {x[1] for x in pheromones}
                if len(second_values) == 1:
                    ant_position = random.choice([node for node in self.colony.problemmap.nodes if node not in ant_path])
                    ant_path += [ant_position]
                # Otherwise, move ant to node with highest pheromone concentration
                else:
                    ant_position = max(pheromones, key=lambda x: x[1])[0]
                    ant_path += [ant_position]
            
            #if q >= q0, choose next node based on probabilistic rule.
            else:          
                ant_position = self._probabilistic_transition(ant_position, possible_nodes)
                ant_path += [ant_position]
            
        return ant_path

    """
    This is a function used in _generate_solution which implements the probabilistic transition rule given in the lecture slides.
    """

    def _probabilistic_transition(self, ant_position, possible_nodes):
        pheromones_times_distance = []
        probabilities = []
        # Calculate the product of the pheromones and distance of the edges to each node, and store in pheromones_times_distance.
        for node in possible_nodes:
            pher = self.colony.problemmap._get_globalpheromones(ant_position,node)**self.colony.problemmap.alpha
            dist = self.colony.problemmap._get_distance(ant_position,node)**self.colony.problemmap.beta
            pheromones_times_distance.append(pher*dist) 

        total = sum([t[1] for t in zip(possible_nodes,pheromones_times_distance)])

        for (node, component) in zip(possible_nodes,pheromones_times_distance):
            probabilities.append(component/total)

        return random.choices(possible_nodes, weights=probabilities, k=1)[0]


class Evaluator(ProblemMap):
    
    def __init__(self,problemmap, solutions):
        self.problemmap = problemmap
        self.solutions = solutions
        self.paths = self._get_paths()

    def _cost(self): 
        total_cost = 0
        for path in self.paths:
            total_cost += self.problemmap._get_distance(path[0],path[1])
        total_cost = total_cost/1649 # Normalise by dividing by the cost of the minimum spanning tree
        return total_cost
    
    """
    Calculate the sum of all minimum distances between all pairs of nodes in the solution.
    """
    def _efficiency(self): #TODO: normalise using minimum spanning tree
        # Creates the dijkstra weights array from the distances of the edges in the solution
        dijkstra_array = [[0 if i ==j else np.inf for j in range(len(self.problemmap.nodes))] for i in range(len(self.problemmap.nodes))]
        for (i,j) in self.paths:
            dijkstra_array[i-1][j-1] = self.problemmap._get_distance(i,j)

        # Counts the shortest distance travellable between all pairs of nodes in the solution
        total_distance = 0
        for node in self.problemmap.nodes:
            dist_matrix = dijkstra(dijkstra_array, directed=False, return_predecessors=False, indices=node-1)
            for index,distance in np.ndenumerate(dist_matrix):
                if index[0] > node-1:
                    total_distance += distance
        total_distance = total_distance/97027 # Normalise by dividing by the efficiency of the minimum spanning tree
        return total_distance
        

    def _get_paths(self):
        paths = []
        for solution in self.solutions:
            for index in range(len(solution)-1):
                i = solution[index]
                j = solution[index+1]
                if (i,j) not in paths and (j,i) not in paths:
                    paths.append((i,j))
        return paths





