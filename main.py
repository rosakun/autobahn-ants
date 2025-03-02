from Objects import Ant, AntColony, ProblemMap


if __name__ == "__main__":

    problemmap = ProblemMap("datasets/cities_xy.txt","datasets/cities_d.txt","city_data/gravity.txt")

    bestsolutions = problemmap._runiteration()

    problemmap._plotSolution_earth(bestsolutions)