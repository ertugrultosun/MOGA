""" 
Multi Objective Genetic Algorithm (MOGA) with Python
Ertugrul Tosun - 2018

Usage
1- Prepare the data set. By default is myciel4.col.
2- Call parse function with your data set path.
3- Call the iteration function with your desired iteration count.

"""

# Imported required libraries.
import numpy as np
from random import randint
from scipy import spatial
from functools import reduce
import operator
from copy import copy, deepcopy
import time
import matplotlib.pyplot as plt
import tkinter

# Set numpy print options for easy debugging.
np.set_printoptions(threshold=np.inf)

colornum = 5
cost = [3,2,6,7,5]
data = np.zeros((23,23))
population = np.zeros(shape=(50,23))
fitness = np.zeros(shape=(50,2))
firstfitness = np.zeros(shape=(50,2))
archivefitness = []
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
itnum=0
finalarchive = np.zeros(shape=(1,23))

# File parse function. Parse the dataset and append to data line by line.
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

# Create initial random population.
def createpopulation(pop):
    for i in range (50):
        for j in range (23):
            pop[i][j] = randint(1, 5)

# Print the current population.
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
                        fitness [i][0] += 1

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

def mkcrossover(parent,crossover):
    i = 0
    randindex = randint(0,22)
    tindex = randindex
    while i<48:
        for j in range(23):
            if j>= randindex:
                crossover[i][tindex] = parent[i+1][tindex]
                crossover[i+1][tindex] = parent[i][tindex]
                tindex += 1
            else:
                crossover[i][j] = parent[i][j]
                crossover[i+1][j] = parent[i+1][j]
        tindex = randindex
        i += 2

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

finalfitness = np.zeros(shape=(len(archive),2))
finalfitness.fill(0)
finalrank = np.zeros(shape=(len(archive)))
finalrank.fill(-1)

def arsivekle():
    arsiv = []
    tesla = []
    for i in range(50):
        if rank[i] == 0:
            onbellek = []
            for j in range(23):
                onbellek.append(population[i][j])
            if onbellek not in archive:
                archive.append(onbellek)

def arsivguncele():
    finalfitness = np.zeros(shape=(len(archive),2))
    finalrank = np.zeros(shape=(len(archive)))
    for i in range(len(archive)):
        for j in range (23):
            node = archive[i][j]
            finalfitness [i][1] += cost[int(node)-1]
            for x in range (23):
                if x != j:
                    if node == archive[i][x] and data[j][x] == 1.0:
                        finalfitness [i][0] += 1

    for i in range(len(finalfitness)):
        finalrank[i] = 0
        for j in range(len(finalfitness)):
            if finalfitness[j][0] < finalfitness [i][0] and finalfitness[j][1] < finalfitness[i][1]:
                finalrank[i] += 1
            elif finalfitness[j][0] < finalfitness[i][0] and finalfitness[j][1] <= finalfitness[i][1]:
                finalrank[i] += 1
            elif finalfitness[j][0] <= finalfitness[i][0] and finalfitness[j][1] < finalfitness[i][1]:
                finalrank[i] += 1

    index=0
    archiveindex = 0
    test = []
    while index < len(archive):
        
        if finalrank[index] != 0:
            del archive[archiveindex]
            archiveindex = index-1
        else:
            archiveindex += 1
            test.append(finalfitness[index][0])
            test.append(finalfitness[index][1])
        index+=1
    arch = np.array(test)
    return arch

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
    mkcrossover(parent,crossover)
    createmutation(crossover,mutation)
    replacepop(population,mutation)
    fitnesscalculate(population)
    rankcalculate(fitness,rank)
    arsivekle()
    archivefitness = arsivguncele()
    
def main(it):
    start_time = time.time()
    i = 0
    parse("data.txt")
    createpopulation(population)
    fitnesscalculate(population)
    for i in range(50):
        firstfitness[i][0] = fitness[i][0]
        firstfitness[i][1] = fitness[i][1]
    rankcalculate(fitness,rank)
    i = 0
    itn = it.get()
    print(int(itn))
    while i<int(itn):
        iteration()
        i += 1
    additionalfn(archive)
    filenm = str(round(start_time, 0))+"_best_populations_of_"+itn+"_iterations.txt"
    hedr = " MOGA - Ertugrul Tosun - 141701010 --- Best Population Of "+itn+". Iteration"
    np.savetxt(filenm, archive, fmt='%d', delimiter=' ', newline='\n', header=hedr, footer='', comments='# ', encoding=None)
    return time.time() - start_time

def additionalfn(archive):
    dominate = is_pareto(fitness)
    for i in range(len(archive)):
        if dominate[i]:
            print(archive[i])
    finalfitness = np.zeros(shape=(len(archive),2))
    finalrank = np.zeros(shape=(len(archive)))
    for i in range(len(archive)):
        for j in range (23):
            node = archive[i][j]
            finalfitness [i][1] += cost[int(node)-1]
            for x in range (23):
                if x != j:
                    if node == archive[i][x] and data[j][x] == 1.0:
                        finalfitness [i][0] += 1

def RunMoga(top,it):
    MsgBox = tkinter.messagebox.askquestion ('Run the MOGA','Are you sure want to run the MOGA. This may take some time and the window may not appear for certain period of time. ',icon = 'warning')
    if MsgBox == 'yes':
        top.destroy()
        totaltime = main(it)
        ff, lf =performancecalc(firstfit,lastfit)
        userinterface(toggle = True,ff=ff,lf=lf,tt=totaltime,count=it.get())
    else:
        tkinter.messagebox.showinfo('Return','You will now return to the application screen')

def drawgraph(toggle = False,final = False):
    if toggle:
        x, y = firstfitness.T
        plt.scatter(x,y)
        plt.title('First Population')
        plt.xlabel('F1 - Wrong Coloring')
        plt.ylabel('F2 - Total Cost')
        plt.show()
    elif final:
        drawarchive = np.zeros(shape=(len(archive),23))
        for i in range(len(archive)):
            for j in range(23):
                drawarchive[i][j] = archive[i][j]
        dominate = is_pareto(drawarchive)
        draw2archive = np.zeros(shape=(len(dominate),23))
        for i in range(len(archive)):
            if dominate[i]:
                for j in range(23):
                    draw2archive[i][j] = archive[i][j]
        finalfitness = np.zeros(shape=(len(dominate),2))
        for i in range(len(dominate)):
            buffer = np.zeros(shape=(1,2))
            for j in range (23):
                node = draw2archive[i][j]
                finalfitness [i][1] += cost[int(node)-1]
                for x in range (23):
                    if x != j:
                        if node == draw2archive[i][x] and data[j][x] == 1.0:
                            finalfitness [i][0] += 1
        x, y = finalfitness.T
        plt.scatter(x,y)
        plt.title('Solution Population')
        plt.xlabel('F1 - Wrong Coloring')
        plt.ylabel('F2 - Total Cost')
        plt.show()
    else:
        x, y = fitness.T
        plt.scatter(x,y)
        plt.title('Last Population')
        plt.xlabel('F1 - Wrong Coloring')
        plt.ylabel('F2 - Total Cost')
        plt.show()

def userinterface(toggle = False, ff=999,lf=999,tt=999,count=0):
    firstfit = ff
    lastfit = lf
    top = tkinter.Tk()
    itn=tkinter.StringVar()
    top.title("MOGA - Ertugrul Tosun - 141701010")
    
    canvas1 = tkinter.Canvas(top, width = 600, height = 500)
    canvas1.pack()
    
    text_entry = tkinter.Entry(top, textvariable=itn)
    wtext = tkinter.Label(top, text="Please Enter Desired Iteration Number", font=("Helvetica", 12))
    canvas1.create_window(300, 450, window=text_entry)
    canvas1.create_window(300, 430, window=wtext)

    if toggle:
        first = "First Population's Fitness Avg   :  "+str(firstfit)
        last = "Last Population's Fitness Avg   :  "+str(lastfit)
        best = "Number of Best Solutions   :  "+str(len(archive))
        total = "Total time for iterations in seconds  :  "+str(round(tt, 2))
        
        w1 = tkinter.Label(top, text=first, font=("Helvetica", 12))
        w2 = tkinter.Label(top, text=last, font=("Helvetica", 12))
        w3 = tkinter.Label(top, text=total, font=("Helvetica", 12))
        w5 = tkinter.Label(top, text=best, font=("Helvetica", 12))
       
        canvas1.create_window(300,150, window=w1)
        canvas1.create_window(300,175, window=w2)
        canvas1.create_window(300,200, window=w3)
        canvas1.create_window(300,225, window=w5)
        
        button2 = tkinter.Button (top, text='Draw first population',font=("Helvetica", 12),command= lambda : drawgraph(toggle=True))
        canvas1.create_window(300, 260, window=button2)
        button3 = tkinter.Button (top, text='Draw last population',font=("Helvetica", 12),command= lambda : drawgraph())
        canvas1.create_window(300, 300, window=button3)
        button3 = tkinter.Button (top, text='Draw solution population',font=("Helvetica", 12),command= lambda : drawgraph(final=True))
        canvas1.create_window(300, 340, window=button3)
        
        yazi = "Available Color Number   :  "+str(colornum)
        it = "Total Iteration Number   :  "+ str(count)
        iterationnum = tkinter.Label(top, text=it, font=("Helvetica", 12))
        w = tkinter.Label(top, text=yazi, font=("Helvetica", 12))
        canvas1.create_window(300,100, window=w)
        canvas1.create_window(300,125, window=iterationnum)
   
    header = tkinter.Label(top, text="Multi Objective Genetic Algorithm", font=("Courier", 18))
    header2 = tkinter.Label(top, text="Ertugrul Tosun",font=("Courier",16))
    
    f=tkinter.Frame(top,height=2,width=500,bg="black")
    canvas1.create_window(300, 80, window=f)
    canvas1.create_window(300,30, window=header)
    canvas1.create_window(300,60, window=header2)
    
    f1=tkinter.Frame(top,height=1,width=500,bg="grey")
    canvas1.create_window(300, 415, window=f1)
    button1 = tkinter.Button (top, text='Start Multi Objective Genetic Algorithm', font=("Helvetica", 12), command= lambda : RunMoga(top,itn))
    canvas1.create_window(300, 480, window=button1)
    top.mainloop()

userinterface()