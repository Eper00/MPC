import numpy as np
from one_set_simulation import realsystem
import random as rn
population=9800000
# Eldönti, hogy az adott halmaz terminális-e, terminális, ha a kórházban lévő betegek szig mon csökken
# Bemenet: A halmaz melyet vizsgálunk
# Kimenet: Döntés, adott halmaz termináls vagy nem
def inspect(test_set):
    for i in range(len(test_set) - 1):
        if test_set[i] <= test_set[i + 1]:
            return 1
    return 0
# Egy adott tömbböl a k. legnagyobb elemet kiválasztja
# Bemenet: A tömb, illetve a k index meely kiválasztja k. legnyagoybb elemet
# Kimenet: k. legnagyobb elem
def kth_largest(a, k): 
    k = len(a) - 1 - k 
    return np.partition(a, k)[k] 
# Random pontokat választa kiinduló pontoknak, bizonyos megkötésekkel, hogy olyan eseteket szimuláljunk, melyek
# valóban relevánsok számunkra
# Bemenet: A pontok száma melyeket szimulálni akarunk
# Kimenet: azok az állapotok melyek terminálisak
def map (number_of_points):
    terminal_sets = np.empty((0, 6))
    for mapping in range(number_of_points):
        y0 = np.zeros(6)
        y0[5]=20000
        for i in range(5):
            if i==0:
                y0[i]=rn.uniform(1000000,7000000)
            if i==1 or i==3:
                y0[i]=rn.uniform(70000,150000)
            if i==2:
                y0[i]=rn.uniform(11000,170000)
            if i==4:
                y0[i]=rn.uniform(5000,80000)
        
        sol = realsystem(y0)
        H = sol.y[5]
        opinion = inspect(H)
        if opinion == 0:
            terminal_sets = np.vstack((terminal_sets, y0))
            
    return terminal_sets
# A függvány a "map" függvényt veszi alapul, célja a további szűrés. Ha a "map" függvény vissza ad egy
# terminális halmazt, azt jobban megvizsgálja, további random kiinduló pontokat vizsgál meg adott S halmazra
# Bemenet: None
# Kimenet: Azok a halmazok melyek átmennek a kétszeres szűrésen
def get_terminal_state():
    res=[]
    terminal_sets = np.empty((0, 6))
    while(len(terminal_sets[:,0])==0):
        res=map(1000)
        print(len(res))
        for i in range(len(res)):
            y0=np.zeros(6)
            y0[0]=kth_largest(res[:,0],i)
            y0[5]=20000
            for t in range(10):    
                for j in range(1,5):
                    if j==1 or j==3:
                        y0[j]=rn.uniform(70000,150000)
                    if j==2:
                        y0[j]=rn.uniform(11000,170000)
                    if j==4:
                        y0[j]=rn.uniform(5000,80000)
                sol = realsystem(y0)
                H = sol.y[5]
                opinion = inspect(H)
                if opinion == 1:
                    break
            if opinion==0:
                terminal_sets = np.vstack((terminal_sets, y0))
    return terminal_sets[0]/population
