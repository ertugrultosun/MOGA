import numpy as np
from random import randint

data = np.zeros((23,23))
population = np.zeros(shape=(50,23))
colornum = 5
fitness = np.zeros(shape=(50,2))

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
            raise ParseError("found more than one p line: " + line)
        elif cmd == 'p' and not found_p:
            found_p = True
            name, node_cnt_s, edge_cnt_s = rest
            node_cnt = int(node_cnt_s)
            edge_cnt = int(edge_cnt_s)
            for i in range(node_cnt):
                nodes.append([])
        elif cmd == 'e' and not found_p:
            raise ParseError("found edges before p")
        elif cmd == 'e' and found_p:
            [edge_from, edge_to] = rest
            data[int(edge_from)-1][int(edge_to)-1] = 1

def createpopulation(pop):
    for i in range (50):
        for j in range (23):
            pop[i][j] = randint(1, 5)
    
def fitnesscalculate(pop):
    temp = 0
    f1 = 0
    f2 = 0
    for i in range (50):
        for j in range (23):
            node = pop[i][j]
            if (pop[i][j] == temp):
                f1 = f1 + temp
                f2 = f2 + temp
                print(temp)
            else:
                temp = pop[i][j]
                print(temp) 
    print(f1)
    print(f2)

parse("data.txt")
createpopulation(population)
fitnesscalculate(population)


