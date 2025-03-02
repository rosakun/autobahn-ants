import csv
import re
import itertools
from exercise_1 import *
import heapq
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
#import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic

top_gdp_cities = ["Berlin", "Hamburg", "Munich", "Frankfurt am Main", "Cologne", "Hannover", "Stuttgart", "Düsseldorf", "Nürnberg", "Bremen", "Bonn", "Essen", "Dortmund", "Dresden", "Leipzig", "Mainz", "Aachen", "Karlsruhe", "Mannheim", "Braunschweig", "Wolfsburg", "Wiesbaden", "Münster", "Duisberg", "Ingolstadt", "Saarbrücken", "Bielefeld", "Augsburg", "Darmstadt", "Regensburg", "Freiburg im Breisgau", "Bochum", "Wuppertal", "Ludwigshafen am Rhein", "Kiel", "Erlangen", "Kassel", "Ulm", "Lübeck", "Krefeld", "Erfurt", "Heidelburg", "Mönchengladbach", "Chemnitz", "Magdeburg", "Osnabrück", "Gelsenkirchen", "Koblenz", "Würzburg", "Oldenburg"]
top_population_cities = ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt am Main", "Stuttgart", "Düsseldorf", "Leipzig", "Dortmund", "Essen", "Bremen", "Dresden", "Hannover", "Nürnberg", "Duisberg", "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Münster", "Mannheim", "Karlsruhe", "Augsburg", "Wiesbaden", "Mönchengladbach", "Gelsenkirchen", "Aachen", "Braunschweig", "Kiel", "Chemnitz", "Halle (Saale)", "Magdeburg", "Freiburg im Breisgau", "Krefeld", "Mainz", "Lübeck", "Erfurt", "Oberhausen", "Rostock", "Kassel", "Hagen", "Potsdam", "Saarbrücken", "Hamm", "Ludwigshafen", "Mülheim an der Ruhr", "Oldenburg", "Osnabrück", "Leverkusen", "Darmstadt"]
cities = ["Berlin", "Hamburg", "Munich", "Frankfurt am Main", "Cologne", "Stuttgart", "Düsseldorf", "Hanover", "Bremen", "Wolfsburg", "Essen", "Dortmund", "Nuremburg", "Leipzig", "Ingolstadt", "Dresden", "Bonn", "Regensburg", "Halle (Salle)", "Ludwigshafen am Rhein"]


def city_score(top_gdp_cities=top_gdp_cities, top_population_cities = top_population_cities, n=20):
    """
    This function calculates the city score of various cities and returns the n cities with the lowest city scores.
    City scores are calculated as the sum of the rank of the city in the top_gdp_cities and top_population_cities lists.
    """
    city_scores = {}
    for i in range(50):
        city = top_gdp_cities[i]
        if city in top_population_cities:
            city_scores[city] = top_gdp_cities.index(city) + 1 + top_population_cities.index(city) + 1
        else:
            city_scores[city] = top_gdp_cities.index(city) + 1
    for i in range(50):
        city = top_population_cities[i]
        if city not in city_scores:
            city_scores[city] = top_population_cities.index(city)
    sorted_city_scores = sorted(city_scores.items(), key=lambda x: x[1])
    return [city for (city, index) in sorted_city_scores[:n]]

def get_traffic_gravity(city_file="city_data/city_data.txt",outfile="city_data/gravity.txt",distances="datasets/cities_d.txt",k=1,b=1.2,weight='pop'):
    cities = []

    distancematrix = readFileAsMatrix(distances)

    with open(city_file, "r",encoding='utf-8') as file:
        for line in file:
            splitline = re.split(r"\s{2,}", line.strip())
            if len(splitline) != 4:
                print(splitline)
            cities.append(splitline)

    gravity_matrix = [[0 for _ in range(len(cities))] for _ in range(len(cities))]

    for item1, item2 in itertools.product(cities, cities):
        index1, index2 = int(item1[0]), int(item2[0])
        if index1 == index2:
            pass
        else:
            if weight=='pop':
                P_i, P_j = float(item1[2]), float(item2[2])
            elif weight=='gdp':
                P_i, P_j = float(item1[3]), float(item2[3])

            distance = (distancematrix[index1-1][index2-1]*1000000)**b

            traffic = int(k * ((P_i*P_j)/distance))
            
            gravity_matrix[index1-1][index2-1] = traffic
    
    with open(outfile, "w",encoding='utf-8') as f:
        for row in gravity_matrix:
            row_str = "\t".join(f"{val:.0f}" for val in row)  
            f.write(row_str + "\n")  

        

