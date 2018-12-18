import numpy as np
from random import randint
from scipy import spatial
from functools import reduce
import operator
import copy
import time
import matplotlib.pyplot as plt 

colornum = 5
cost = [3,2,6,7,5]
data = np.zeros((23,23))
population = np.zeros(shape=(50,23))
fitness = np.zeros(shape=(50,2))
rank = np.zeros(shape=(50))
parent = np.zeros(shape=(50,23))

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

def rankcalculate(fit,rank):
    for i in range(50):
        for j in range(50):
            if fit[j][0] < fit [i][0] and fit[j][1] < fit[i][1]:
                rank[i] += 1
            elif fit[j][0] < fit[i][0] and fit[j][1] <= fit[i][1]:
                rank[i] += 1
            elif fit[j][0] <= fit[i][0] and fit[j][1] < fit[i][1]:
                rank[i] += 1

def printrank(rank):
    for i in range(50):
        print(i,". populations rank is :",rank[i])

def createparent(pop,parent):
    temprank = 99999
    tempparent = []
    randomindex = 0
    for i in range(50):
        for j in range(5):
            randomindex = randint(1,50)
            if temprank > rank[randomindex]:
                temprank = rank[randomindex]
                tempparent = population[randomindex]
        parent[i] = tempparent[i]

def printparent(parent):
    for i in range (50):
        for j in range (23):
            print(parent[i][j]) 

def is_pareto(costs, maximise=False):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def iteration():
    return True

parse("data.txt")
createpopulation(population)
printpopulation(population)
fitnesscalculate(population)
dominate = is_pareto(fitness)
for i in range(len(fitness)):
    if dominate[i]:
        print(fitness[i])
print(dominate)
x, y = fitness.T
plt.scatter(x,y)
plt.show()










