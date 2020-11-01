import pandas as pd
import numpy as np
import random as rd
import constructive


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


def greedyRan(i, alpha):

    # Get the data (from the previous method)
    n, m, p, consts, fobs = get_data(i)

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
        if sum([1 - (valObj[i] <= right[i] - fix2[i]) for i in range(m)]):
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
    return (sum([1 - (valObj[i] <= right[i]) for i in range(m)]))


print('i k j m')
for i in range(20):
    print(i + 1, greedyRan(i + 1, 0.5), *constructive.constructive(i + 1))
