import numpy as np
from geopy.distance import geodesic

def get_distancematrix(coordinatefile, outfile):
    with open(coordinatefile, "r") as file:
        coordinates = []
        for line in file:
            x, y = map(float, line.split())
            coordinates.append((x, y))
    
    # Compute GEODESIC DISTANCE
    coordinates = np.array(coordinates) 
    num_points = len(coordinates)
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        coordinates_i = coordinates[i]
        i_lat = coordinates_i[0]
        i_lon = coordinates_i[1]
        for j in range(num_points):
            j_lat = coordinates[j][0]
            j_lon = coordinates[j][1]
            distance_matrix[i, j] = geodesic((i_lat,i_lon), (j_lat,j_lon)).kilometers

    with open(outfile, "w") as f:
        for row in distance_matrix:
            row_str = "\t".join(f"{val:.6f}" for val in row)  
            f.write(row_str + "\n")  