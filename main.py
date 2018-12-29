""" 
Multi Objective Genetic Algorithm (MOGA) with Python
Ertugrul Tosun - 2018

Usage
1- Prepare the data set. By default is myciel4.col.
2- Call parse function with your data set path.
3- Call the iteration function with your desired iteration count.

"""

import numpy as np
from random import randint
from scipy import spatial
from functools import reduce
import operator
from copy import copy, deepcopy
import time
import matplotlib.pyplot as plt
import tkinter
np.set_printoptions(threshold=np.inf)

colornum = 5
cost = [3,2,6,7,5]
data = np.zeros((23,23))
population = np.zeros(shape=(50,23))
fitness = np.zeros(shape=(50,2))
firstfitness = np.zeros(shape=(50,2))
rank = np.zeros(shape=(50))
parent = np.zeros(shape=(50,23))
crossover = np.zeros(shape=(50,23))
mutation = np.zeros(shape=(50,23))
iterationcount = 0
n = 2
archive = []
archivecount = 0
firstfit = 0
lastfit = 0
totaltime = 0

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
    fitness[:] = 0
    for i in range (50):
        for j in range (23):
            node = pop[i][j]
            fitness [i][1] += cost[int(node)-1]
            for x in range (23):
                if x != j:
                    if node == pop[i][x] and data[j][x] == 1.0:
                        #print("X :",j," Y :",x," == 1")
                        f1 += 1
        fitness [i][0] = f1
        f1 = 0 

def rankcalculate(fit,rank):
    for i in range(50):
        rank[i] = 0
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
    tempparent = np.zeros(shape=(1,23))
    randomindex = 0
    for i in range(50):
        for j in range(5):
            randomindex = randint(1,50)-1
            if temprank > rank[randomindex]:
                temprank = rank[randomindex]
                for z in range(23):
                    tempparent[0][z] = population[randomindex][z]
        for x in range(23):
            parent[i][x] = tempparent[0][x]
        temprank = 9999

def printparent(parent):
    for i in range (50):
        for j in range (23):
            print(parent[i][j]) 

def createcrossover(parent,crossover):
    i,j = 1,1
    temp1,temp2 = np.zeros(shape=(1,23)),np.zeros(shape=(1,23))
    while i<50:
        for x in range(23):
            temp1[0][x] = parent[i][x]
            if i<49:
                temp2[0][x] = parent[i+1][x]
        for x in range(23):
            if x<11:
                crossover[j][x] = temp1[0][x]
            elif x>11:
                crossover[j][x] = temp2[0][x] 
        if i<49:
            j += 1
        for x in range(23):
            if x<11:
                crossover[j][x] = temp2[0][x]
            elif x>11:
                crossover[j][x] = temp1[0][x]
        i += 1

def createmutation(crossover,mutation):
    for i in range(50):
        randomindex = randint(1,23)-1
        crossover[i][randomindex] = randint(1,5)
    for i in range(50):
        for j in range(23):
            mutation[i][j] = crossover[i][j]

def is_pareto(costs, maximise=False):
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

def replacepop(pop,mut):
    for i in range(50):
        for j in range(23):
            pop[i][j] = mutation[i][j]

def addarchive():
    buffer = []
    for i in range(50):
        if rank[i] == 0:
            for j in range(23):
                buffer.append(population[i][j])
                archive.append(buffer)
            buffer.clear()

def updatearchive():
    bufrank = np.zeros(shape=(501))
    buffitness = np.zeros(shape=(501,2))
    lastarchive = np.zeros(shape=(501,23))
    f1 = 0
    f2 = 0
    node = 0
    buffer = 0
    for i in range (501):
        for j in range (23):
            node = archive[i][j]
            if node != 999:
                buffitness [i][1] = cost[int(node)-1]
                for x in range (23):
                    if x != j:
                        if node == archive[i][x] and data[j][x] == 1.0:
                            #print("X :",j," Y :",x," == 1")
                            f1 += 1
        buffitness [i][0] = f1
        f1 = 0 
    for i in range(501):
        for j in range(501):
            if buffitness[j][0] < buffitness [i][0] and buffitness[j][1] < buffitness[i][1]:
                bufrank[i] += 1
            elif buffitness[j][0] < buffitness[i][0] and buffitness[j][1] <= buffitness[i][1]:
                bufrank[i] += 1
            elif buffitness[j][0] <= buffitness[i][0] and buffitness[j][1] < buffitness[i][1]:
                bufrank[i] += 1
    for i in range(501):
        if bufrank[i] == 0:
            for j in range(23):
                lastarchive[i][j] = archive[i][j]
    return lastarchive

def performancecalc(fi,la):
    sum = 0
    for i in range(50):
        fi += (firstfitness[i][0] + firstfitness[i][1])
        la += (fitness[i][0] + fitness[i][1])
    fi = fi/50
    la = la/50
    return fi,la

def iteration(): 
    createparent(population,parent)
    createcrossover(parent,crossover)
    createmutation(crossover,mutation)
    replacepop(population,mutation)
    fitnesscalculate(population)
    rankcalculate(fitness,rank)
    addarchive()
    
def main():
    itnum=500
    start_time = time.time()
    i = 0
    parse("data.txt")
    createpopulation(population)
    fitnesscalculate(population)
    for i in range(50):
        firstfitness[i][0] = fitness[i][0]
        firstfitness[i][1] = fitness[i][1]
    rankcalculate(fitness,rank)
    """ initarchive = np.zeros(shape=[50, 2])
    for i in range(len(archive)-1):
        initarchive[i][0] = archive[i][0]
        initarchive[i][1] = archive[i][1] """
    i = 0
    while i<itnum:
        iteration()
        i += 1
    return time.time() - start_time

def additionalfn(fit):
    dominate = is_pareto(fitness)
    for i in range(len(fitness)):
        if dominate[i]:
            print(fitness[i])
    print(dominate)
    x, y = fitness.T
    plt.scatter(x,y)
    plt.show()

def RunMoga(top):
    MsgBox = tkinter.messagebox.askquestion ('Run the MOGA','Are you sure want to run the MOGA. This may take some time and the window may not appear for certain period of time. ',icon = 'warning')
    if MsgBox == 'yes':
        top.destroy()
        totaltime = main()
        ff, lf =performancecalc(firstfit,lastfit)
        userinterface(toggle = True,ff=ff,lf=lf,tt=totaltime)
    else:
        tkinter.messagebox.showinfo('Return','You will now return to the application screen')

def drawgraph(toggle = False):
    if toggle:
        x, y = firstfitness.T
        plt.scatter(x,y)
        plt.title('First Population')
        plt.xlabel('F1 - Wrong Coloring')
        plt.ylabel('F2 - Cost')
        plt.show()
    else:
        x, y = fitness.T
        plt.scatter(x,y)
        plt.title('Last Population')
        plt.xlabel('F1 - Wrong Coloring')
        plt.ylabel('F2 - Cost')
        plt.show()

def userinterface(toggle = False, ff=999,lf=999,tt=999):
    firstfit = ff
    lastfit = lf
    top = tkinter.Tk()
    top.title("MOGA - Ertugrul Tosun - 141701010")
    canvas1 = tkinter.Canvas(top, width = 600, height = 400)
    canvas1.pack()
    if toggle:
        first = "First Population's Fitness Avg   :  "+str(firstfit)
        last = "Last Population's Fitness Avg   :  "+str(lastfit)
        total = "Total time for iterations in seconds  :  "+str(round(tt, 2))
        w1 = tkinter.Label(top, text=first, font=("Helvetica", 12))
        w2 = tkinter.Label(top, text=last, font=("Helvetica", 12))
        w3 = tkinter.Label(top, text=total, font=("Helvetica", 12))
        canvas1.create_window(300,150, window=w1)
        canvas1.create_window(300,175, window=w2)
        canvas1.create_window(300,200, window=w3)
        button2 = tkinter.Button (top, text='Draw first population',font=("Helvetica", 12),command= lambda : drawgraph(toggle=True))
        canvas1.create_window(300, 250, window=button2)
        button3 = tkinter.Button (top, text='Draw last population',font=("Helvetica", 12),command= lambda : drawgraph())
        canvas1.create_window(300, 285, window=button3)
    header = tkinter.Label(top, text="Multi Objective Genetic Algorithm", font=("Courier", 18))
    header2 = tkinter.Label(top, text="Ertugrul Tosun",font=("Courier",16))
    f=tkinter.Frame(top,height=2,width=500,bg="black")
    canvas1.create_window(300, 80, window=f)
    yazi = "Available Color Number   :  "+str(colornum)
    iterationnum = tkinter.Label(top, text="Total Iteration Number   :  500", font=("Helvetica", 12))
    w = tkinter.Label(top, text=yazi, font=("Helvetica", 12))
    canvas1.create_window(300,30, window=header)
    canvas1.create_window(300,60, window=header2)
    canvas1.create_window(300,100, window=w)
    canvas1.create_window(300,125, window=iterationnum)
    button1 = tkinter.Button (top, text='Start Multi Objective Genetic Algorithm', font=("Helvetica", 12), command= lambda : RunMoga(top))
    canvas1.create_window(300, 350, window=button1)
    top.mainloop()

userinterface()