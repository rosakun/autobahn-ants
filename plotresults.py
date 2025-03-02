import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

points = [(10.971, 1.382, 1), (8.454,1.220,2),(5.366,0.878,3),(4.511,0.798,4),(8.706,1.061,5),(3.829,0.771,6),(10.440,1.142,7),(7.978,1.651,8),(8.867,1.145,9),(7.586,1.902,10),(8.877,1.183,11),(8.424,1.287,12),(7.572,1.214,13),(4.618,1.042,14),(4.241,0.957,15),(8.539,1.325,16),(4.224,0.769,17),(7.902,1.239,19),(8.715,1.159,20),(9.090,1.105,21)]



def calculate_pareto_front(points=points):
    pareto_front = []
    current_points = points.copy()
    def pareto_dominates(i,j):
        i_f0, i_f1, i_name = i[0], i[1], i[2]
        j_f0, j_f1, j_name = j[0], j[1], j[2]
        return i_f0 <= j_f0 and i_f1 <= j_f1 and (i_f0 < j_f0 or i_f1 < j_f1)
    while current_points != []:
        i = current_points[0]
        pareto = True
        for j in current_points[1:]:
            if pareto_dominates(i,j): # If I dominates any other point, remove the dominated point from the list
                current_points.remove(j)
            if pareto_dominates(j,i): # If i is dominated by any point, it is not pareto-optimal; set pareto to False
                pareto = False
        if pareto:
            pareto_front.append(i)
        current_points.remove(i)
    return pareto_front


def plot_solutions(points, subset):
    x_coords, y_coords, names = zip(*points)
    red_points = subset
    blue_points = [points[i] for i in range(len(points)) if points[i] not in subset]

    if blue_points:
        blue_x, blue_y, blue_names = zip(*blue_points)
        plt.scatter(blue_x, blue_y, color='blue', marker='o', label=None)

    if red_points:
        red_x, red_y, red_names = zip(*red_points)
        plt.scatter(red_x, red_y, color='red', marker='o', label='Pareto-Optimal Solutions')

    for i, name in enumerate(names):
        plt.text(x_coords[i], y_coords[i], name, fontsize=9, ha='right', va='bottom')

    slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_coords)
    
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4g}")
    print(f"Standard Error: {std_err:.4f}")

    reg_line_x = [min(x_coords), max(x_coords)]
    reg_line_y = [slope * x + intercept for x in reg_line_x]
    plt.plot(reg_line_x, reg_line_y, color='green', linestyle='--')


    plt.xlabel('Cost')
    plt.ylabel('Efficiency')
    plt.title('Cost and Efficiency of Solutions')
    plt.legend()
    plt.show()


