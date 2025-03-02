from exercise_1 import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import itertools
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import dijkstra


cities = ["Berlin", "Hamburg", "Munich", "Frankfurt am Main", "Cologne", "Stuttgart", "Düsseldorf", "Hanover", "Bremen", "Wolfsburg", "Essen", "Dortmund", "Nuremburg", "Leipzig", "Ingolstadt", "Dresden", "Bonn", "Regensburg", "Halle (Salle)", "Ludwigshafen am Rhein"]


def get_minimum_spanning_tree():
    distances = readFileAsMatrix("datasets/cities_d.txt")
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    Tcsr = minimum_spanning_tree(distances)
    array = Tcsr.toarray().astype(int)
    solutions = []
    for i in range(array.shape[0]):  
        for j in range(array.shape[1]):  
            if array[i, j] > 0:  
                solutions.append([i, j])  
    return solutions
    


def plotSolution_earth(points, distances, solutions, title, cities):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection=ccrs.Mercator())  # Using Mercator projection
    
    ax.set_extent([5.87, 15.04, 47.3, 55.1], crs=ccrs.PlateCarree())  # Longitude and Latitude bounds for Germany
    

    ax.coastlines(resolution='110m')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black') 
    ax.add_feature(cfeature.LAND, edgecolor='black')  
    
    x, y = zip(*points) 
    lon, lat = zip(*[(y, x) for x, y in points])  # Swap lon/lat to lat/lon

    ax.scatter(lon, lat, color='red', transform=ccrs.PlateCarree()) 
    
    for i, p in enumerate(zip(lon, lat)):
        ax.text(p[0] + 0.1, p[1], cities[i], transform=ccrs.PlateCarree(),
                fontsize=9, ha='left', color='black')  

    
    solution_lines = []
    for solution in solutions:
        for i in range(len(solution)-1):
            solution_lines.append(((points[solution[i]-1][1], points[solution[i]-1][0]), 
                                    (points[solution[i+1]-1][1], points[solution[i+1]-1][0])))  # Swap lon/lat
    
    solution_line_collection = mc.LineCollection(solution_lines, linewidths=2, color='red', transform=ccrs.PlateCarree())
    ax.add_collection(solution_line_collection)
    
    # Add title and show the plot
    ax.set_title(title)
    plt.show()


def get_mst_cost(mst):
    distances = readFileAsMatrix("datasets/cities_d.txt")
    length = 0
    for edge in mst:
        length += distances[edge[0]][edge[1]]
    return length


def get_mst_efficiency(mst):
    distances = readFileAsMatrix("datasets/cities_d.txt")
    nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    length = 0

    #dijkstra_array = [[0 if i ==j else np.inf for j in range(len(self.problemmap.nodes))] for i in range(len(self.problemmap.nodes))]
    dijkstra_array = [[0 if i==j else np.inf for j in range(len(distances))] for i in range(len(distances))]
    for i, j in mst:
        dijkstra_array[i][j] = distances[i][j]
        dijkstra_array[j][i] = distances[j][i]
    for node in nodes:
        dist_matrix = dijkstra(dijkstra_array, directed=False, return_predecessors=False, indices=node)
        for index,distance in np.ndenumerate(dist_matrix):
                if index[0] > node:
                    length += distance
    return length

