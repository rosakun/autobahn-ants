# autobahn-ants

Autobahn ants is the result of my final project for the course 'Bio-Inspired AI', held by Giovanni Iacca at University of Trento. \
The goal of the project was to use Ant Colony Optimization (ACO) to create an 'optimal' highway network for the country of Germany.

## Background

Ant Colony Optimization is an algorithm that mimics the ant's stigmergy to find optimal paths through graphs. To understand the principle of the algorithm, you need to know three things:
1. Ants deposit a trail of pheromones along the path it traverses.
2. Ants decide which path to take based on pheromone concentrations along the paths, preferring those paths with more pheromones.
3. Pheromones evaporate, or 'decay', over time.


Say a colony of ants should find the shortest route from point A to point B. The ants begin by walking randomly until they find the end point. The idea is because shorter 'solutions', or paths, take less time to traverse, meaning more ants can walk along it in a given amount of time, meaning it's pheromone concentration increases over time. ACO can be used to solve shortest-path problems, Traveling Salesman Problems, and pretty much any problem that you can formulate as a graph.

## Project Idea

In this project, we wanted to create an 'optimal' highway network for Germany. We try to minimize **cost**, meaning total highway length, while maximizing **effiency**, meaning average distance between any two given cities. We approximate the 



## Show some solutions

## Explain the three classes

## Explain what you'd need to do if you want to run the experiments for your own city

## Show some different results from my paper

