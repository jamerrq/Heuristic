import signal
import pandas as pd
import numpy as np
import random as rd
import multiprocessing as mp
import os


def get_data(i):

    # Read data from sheet i
    df = pd.read_excel('Datos.xlsx', header=None, sheet_name='I' + str(i))

    # Get n, m, p, n items, m restrictions, p fobs
    n, m, p = map(int, df.loc[0][:3])

    # Create constrainsts
    consts = np.zeros((m, n + 1))
    # Read them
    for i in range(m):
        consts[i] = df.loc[i + 1].to_numpy()

    # Create fobs
    fobs = np.zeros((p, n))
    # Read them
    for i in range(p):
        fobs[i] = df.loc[i + 1 + m].to_numpy()[:-1]

    # Return all shit
    return n, m, p, consts, fobs


def greedyRan(case, alpha):

    # Get the data (from the previous method)
    n, m, p, consts, fobs = get_data(case)

    left = np.zeros((m, n))
    right = np.zeros(m)
    for i in range(m):
        left[i] = consts[i][:-1]
        right[i] = consts[i][-1]

    # Create the values to return (solution)
    items = np.zeros(n)

    # Create calification value/weight
    cals = [None]*n
    for l in range(n):
        v = 0
        w = 0
        for j in range(p):
            v += fobs[j][l]
        for k in range(m):
            if False:
                w += 1 / abs(consts[k][l])
            else:
                w += consts[k][l]

        cals[l] = [v/w, l, False] # [score/grade index taken]

    # Sort list descending
    cals_sorted = sorted(cals, reverse=True)
    # Generate initial RCL
    RCL = cals_sorted[0 : int(len(cals) * alpha)]

    # Fix negative constraints
    fix = np.zeros(m)
    fix2 = np.zeros(m)
    for i in range(m):
        if right[i] < 0:
            fix[i] = sum(left[i][j] for j in range(n))
            fix2[i] = right[i]


    valObj = np.dot(left, items)

    # Make solution
    while RCL:

        # Choose a random item from the RCL
        index = rd.randint(0, len(RCL) - 1)
        # Modify its value in the solution
        items[RCL[index][1]] = 1


        # Calculate constraints
        valObj = np.dot(left, items)


        # If we got a wrong one
        if sum([1 - (valObj[i] <= right[i] - fix[i]) for i in range(m)]):
            items[RCL[index][1]] = 0
            valObj = np.dot(left, items)
            break

        # When an item is chosen
        real_index = RCL[index][1]
        cals[real_index][2] = True


        # Update RCL
        nottaken = [cals[i] for i in range(len(cals)) if not cals[i][2]]
        cals_sorted = sorted(nottaken, reverse=True)
        RCL = cals_sorted[0:int(len(cals_sorted)*alpha)]

    #print(sum([1 - (valObj[i] <= right[i]) for i in range(m)]))

    for j in range(len(cals)):
        if not cals[j][2]:
            items[cals[j][1]] = 1
            valObj = np.dot(left, items)
            if sum([1 - (valObj[i] <= right[i]) for i in range(m)]):
                items[cals[j][1]] = 0
                valObj = np.dot(left, items)


    #print(valObj, right)
    #print(case, sum([1 - (valObj[i] <= right[i]) for i in range(m)]), m)
    return items, left, right, fobs


def neighbors1(s):
    # Separate indexes from zeros and ones
    zeros = [i for i in range(len(s)) if 1 - s[i]]
    ones  = [i for i in range(len(s)) if s[i]]
    # Create the neighborhood with the original solution
    neighbors = [s]
    #
    for i in range(len(zeros)):
        for j in range(len(ones)):
            neighbor = s.copy()
            neighbor[zeros[i]] = 1
            neighbor[ones[j]]  = 0
            neighbors.append(neighbor)

    # for neighbor in neighbors:
    #     print(neighbor)
    return neighbors


def neighbors2(s):
    neighbors = [s]
    #
    for i in range(len(s)):
        neighbor = s.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    #
    #for neighbor in neighbors:
    #    print(neighbor)
    return neighbors


def neighbors3(s):
    # Number of neighbors is the number of variables
    nn = len(s)
    # Neighbor start with the original solution
    neighborhood = [s]
    # Create a set to avoid repetitions
    check = set()
    # Add the original
    check.add(''.join([str(x) for x in s]))
    # Fill the neighborhood
    while len(neighborhood) <= nn:
        # Choose a random number of zeros and ones
        ones = rd.randint(1, len(s))
        zers = len(s) - ones
        # Create a vector of ones and zeros
        neighbor = [0] * zers + [1] * ones
        # Shuffle the array
        rd.shuffle(neighbor)
        # Create the key to check if it is repeated
        key = ''.join([str(x) for x in neighbor])
        if not key in check:
            neighborhood.append(neighbor)
            check.add(key)

    # Print that shit
    #for neighbor in neighborhood:
    #    print(neighbor)

    return neighborhood


def neighbor_value(neighbor, left, right, fobs):
    m = len(left)
    #
    valObj = np.dot(left, neighbor)
    noFill = sum([1 - (valObj[i] <= right[i]) for i in range(m)])
    #
    sumFob = sum(np.dot(fobs, neighbor))
    #
    return noFill, -sumFob


def vnd(i, alpha=0.5, file=None):
    #
    s, l, r, fobs = greedyRan(i, alpha)
    s_value = neighbor_value(s, l, r, fobs)
    #
    if file:
        file.write('First: ' + str(s_value) + '\n')
    else:
        print('First:', neighbor_value(s, l, r, fobs))
    #
    j = 0
    #
    neighborhoods = [neighbors3, neighbors2, neighbors1]
    nn = len(neighborhoods)
    #
    best_neig = s
    while j < nn:
        neighbors = neighborhoods[j](s)
        neighbors.sort(key=lambda x:neighbor_value(x, l, r, fobs))
        best_neig = neighbors[0]
        #
        best_neig_value = neighbor_value(best_neig, l, r, fobs)
        s_value = neighbor_value(s, l, r, fobs)
        if best_neig_value < s_value:
            if file:
                file.write('Got better! ' + str(s_value) + ' ' + str(best_neig_value) + '\n')
            else:
                print('Got better! ', s_value, best_neig_value)
            j = 0
            s = best_neig
        else:
            j += 1

    if file:
        file.write('Final: ' + str(s_value) + '\n')
    else:
        print('Final:', neighbor_value(s, l, r, fobs))
    return s


def check_solution(i=1):
    n, m, p, consts, fobs = get_data(i)
    left = np.zeros((m, n))
    right = np.zeros(m)
    #
    solution = []
    file = open('hamilton_solution.in')
    for line in file:
        row = list(map(int, line.split()))
        solution.extend(row)
    file.close()
    solution = np.array(solution)
    #
    for i in range(m):
        left[i] = consts[i][:-1]
        right[i] = consts[i][-1]

    valObj = np.dot(left, solution)
    return sum([1 - (valObj[i] <= right[i]) for i in range(m)])


def signal_handler(signum, frame):
    raise Exception("Timed out")


# Init pool
pool = mp.Pool(mp.cpu_count())


def print_results(i):
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(300)   # Sixty seconds
    file = None
    try:
        try:
            file = open(f'./Solutions/sol{i}.out', 'w')
        except FileNotFoundError:
            os.system('mkdir ./Solutions')
            file = open(f'./Solutions/sol{i}.out', 'w')
        vnd(i,file=file)
    except Exception as e:
        print(e + f' for case {i}')
        file.close()
    print(f'Finished case {i}!')
    file.close()


try:
    os.system('rm ./Solutions/*')
except Exception:
    pass
[pool.map(print_results, [i + 1 for i in range(20)])]

pool.close()
print('Done!')

#print(check_solution())
