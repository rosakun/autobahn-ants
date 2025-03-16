# autobahn-ants

Autobahn ants is the result of my final project for the course 'Bio-Inspired AI', held by Giovanni Iacca at University of Trento. \
The goal of the project was to use Ant Colony Optimization (ACO) to create an 'optimal' highway network for the country of Germany.

## Background

Ant Colony Optimization is an algorithm that mimics the ant's stigmergy to find optimal paths through graphs. The algorithm relies on three points:
1. Ants deposit a trail of pheromones along the path it traverses.
2. Ants decide which path to take based on pheromone concentrations along the paths, preferring those paths with more pheromones.
3. Pheromones evaporate, or 'decay', over time.


Say a colony of ants should find the shortest route from point A to point B. The ants begin by walking randomly until they find the end point. The idea is because shorter 'solutions', or paths, take less time to traverse, meaning more ants can walk along it in a given amount of time, meaning it's pheromone concentration increases over time. ACO can be used to solve shortest-path problems, Traveling Salesman Problems, and pretty much any problem that you can formulate as a graph.

## Project Idea

In this project, we wanted to create an 'optimal' highway network for Germany. The highway network is approximated as an undirected graph, where cities are nodes and edges are possible connections between cities.

The idea is to assign a colony of ants to every pair of cities A and B. The size of each colony is determined by the influence of the two cities—measured, for example, by their population or GDP. As a result, there are significantly more ants in the colony connecting Berlin and Hamburg than in the one linking Düsseldorf and Halle (Saale). Each colony searches for the most optimal path between its two cities. However, because routes between more influential cities accumulate more pheromones, colonies connecting less important cities may be drawn to these high-pheromone paths, even if it means taking a longer route.


We try to minimize **cost**, meaning total highway length, while maximizing **effiency**, meaning average distance between any two given cities. 

### Some solutions

Here are some solutions generated by the algorithm - as you can see, there is some work to be done. I've got plenty of ideas on how to improve the algorithm - for some of them, please refer to the paper!

![pareto_optimal](https://github.com/user-attachments/assets/02fbe66d-803c-44eb-8d5e-207827739043)


## The Code

There are three classes - ProblemMap, AntColony, and Ant - all of which are in the Objects.py file. There is also the Evaluator class which calculates the cost and efficiency of a given map. You can run the main code by running the file in main.py.

Given a list of tuples represeting the cost, efficiency, and index of various solutions, you can plot the solutions and calculate the Pareto-front using the functions in plotresults.py. There is an example of such a list in that file.


## Would you like to run these experiments for your own country?

If so, you're going to need:

* A file containing the city names you want to include, their population, and their GDP (example is in city_data/city_data.txt). You can use this to rank cities by city score using utils/get_cityscores.py. 
* A file containing the latitude and longitude of each city, in order of their city scores. An example is in datasets/cities_xy.txt. This can be used to create a distance matrix using utils/get_distancematrix.py.
* A gravity map, which is a matrix that tells you how many ants should traverse from city A to city B. You can calculate this using get_gravity_map() in utils/get_cityscores.py.

This should allow you to initialise the correct classes.




