import numpy as np
from random import randint
from scipy import spatial
from functools import reduce
import operator
import copy
import time

colornum = 5
cost = [3,2,6,7,5]
data = np.zeros((23,23))
population = np.zeros(shape=(50,23))
fitness = np.zeros(shape=(50,2))
rank = np.zeros(shape=(50))

def parse(filename):
    file_content = []
    with open(filename, 'r') as file:
        for line in file:
            file_content.append(line)
    nodes = []
    found_p = False
    name = ''
    node_cnt = 0
    edge_cnt = 0
    for line in file_content:
        cmd, *rest = line.split()
        if cmd == 'p' and found_p:
            print("found more than one p line: " + line)
        elif cmd == 'p' and not found_p:
            found_p = True
            name, node_cnt_s, edge_cnt_s = rest
            node_cnt = int(node_cnt_s)
            edge_cnt = int(edge_cnt_s)
            for i in range(node_cnt):
                nodes.append([])
        elif cmd == 'e' and not found_p:
            print("found edges before p")
        elif cmd == 'e' and found_p:
            [edge_from, edge_to] = rest
            data[int(edge_from)-1][int(edge_to)-1] = 1

def createpopulation(pop):
    for i in range (50):
        for j in range (23):
            pop[i][j] = randint(1, 5)
    
def printpopulation(pop):
    for i in range (50):
        for j in range (23):
            print(pop[i][j]) 

def fitnesscalculate(pop):
    f1 = 0
    f2 = 0
    node = 0
    buffer = 0
    for i in range (50):
        for j in range (23):
            node = pop[i][j]
            fitness [i][1] = cost[int(node)-1]
            for x in range (23):
                if x != j:
                    if node == pop[i][x] and data[j][x] == 1.0:
                        print("X :",j," Y :",x," == 1")
                        f1 += 1
        fitness [i][0] = f1
        f1 = 0 

def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    print(is_efficient)
    return is_efficient


parse("data.txt")
createpopulation(population)
printpopulation(population)
fitnesscalculate(population)
is_pareto(fitness)
print(fitness)






