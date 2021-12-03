import math
from operator import index, indexOf
import sys
import time
import random
import copy
from random import randrange
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.function_base import average


def make_initial_grid():
    grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    rand1 = randrange(4)
    rand2 = randrange(4)
    rand3 = randrange(4)
    rand4 = randrange(4)
    while (rand1 == rand3 and rand2 == rand4):
        rand3 = randrange(4)
        rand4 = randrange(4)
    grid[rand1][rand2] = 2
    grid[rand3][rand4] = 2
    return grid


def moveLeft(grid):
    grid1 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(4):
        index = 0
        for j in range(4):
            if(grid[i][j] != 0):
                grid1[i][index] = grid[i][j]
                index += 1

    grid1 = merge_grid(grid1)

    grid2 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for a in range(4):
        index = 0
        for b in range(4):
            if(grid1[a][b] != 0):
                grid2[a][index] = grid1[a][b]
                index += 1

    return grid2


def add_2_or_4(grid):
    empty = []
    for i in range(4):
        for j in range(4):
            if (grid[i][j] == 0):
                empty.append((i, j))
    rand = empty[randrange(len(empty))]
    rand1 = randrange(10)
    if(rand1 >= 1):
        grid[rand[0]][rand[1]] = 2
    else:
        grid[rand[0]][rand[1]] = 4

    return grid


def moveRight(grid):
    return reverse_grid(moveLeft(reverse_grid(grid)))


def moveUp(grid):
    return transpose_grid(moveLeft(transpose_grid(grid)))


def moveDown(grid):
    return transpose_grid(moveRight(transpose_grid(grid)))


def reverse_grid(grid):
    grid1 = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            grid1[i].append(grid[i][3 - j])
    return grid1


def transpose_grid(grid):
    grid1 = [[], [], [], []]
    for i in range(4):
        for j in range(4):
            grid1[i].append(grid[j][i])
    return grid1


def merge_grid(grid):
    for i in range(4):
        for j in range(3):
            if(grid[i][j] == grid[i][j + 1] and grid[i][j] != 0):
                grid[i][j] = grid[i][j] * 2
                grid[i][j + 1] = 0
    return grid


def heuristic(grid):
    heuristic_value = 0
    weights = [
        [4096*3, 8192*3, 16384*3, 32768*3],
        [2048*3, 1024*3, 512*3, 256*3],
        [16*3, 32*3, 64*3, 128*3],
        [8*3, 4*3, 2*3, 1*3]
    ]
    for i in range(4):
        for j in range(4):
            heuristic_value += (grid[i][j] * weights[i][j])

    return heuristic_value


def minimax_search_allmins(grid):
    up = moveUp(grid)
    down = moveDown(grid)
    left = moveLeft(grid)
    right = moveRight(grid)

    upMin = []
    downMin = []
    leftMin = []
    rightMin = []

    for i in range(4):
        for j in range(4):
            if(up[i][j] == 0):
                grid1 = copy.deepcopy(up)
                grid2 = copy.deepcopy(up)
                grid1[i][j] = 2
                grid2[i][j] = 4
                upMin.append(grid1)
                upMin.append(grid2)
            if(down[i][j] == 0):
                grid1 = copy.deepcopy(down)
                grid2 = copy.deepcopy(down)
                grid1[i][j] = 2
                grid2[i][j] = 4
                downMin.append(grid1)
                downMin.append(grid2)
            if(left[i][j] == 0):
                grid1 = copy.deepcopy(left)
                grid2 = copy.deepcopy(left)
                grid1[i][j] = 2
                grid2[i][j] = 4
                leftMin.append(grid1)
                leftMin.append(grid2)
            if(right[i][j] == 0):
                grid1 = copy.deepcopy(right)
                grid2 = copy.deepcopy(right)
                grid1[i][j] = 2
                grid2[i][j] = 4
                rightMin.append(grid1)
                rightMin.append(grid2)
    



    upMax = []
    downMax = []
    leftMax = []
    rightMax = []

    for i in upMin:
        upMax.append(heuristic(moveUp(i)))
        upMax.append(heuristic(moveDown(i)))
        upMax.append(heuristic(moveLeft(i)))
        upMax.append(heuristic(moveRight(i)))


    for i in downMin:
        downMax.append(heuristic(moveUp(i)))
        downMax.append(heuristic(moveDown(i)))
        downMax.append(heuristic(moveLeft(i)))
        downMax.append(heuristic(moveRight(i)))

    for i in leftMin:
        leftMax.append(heuristic(moveUp(i)))
        leftMax.append(heuristic(moveDown(i)))
        leftMax.append(heuristic(moveLeft(i)))
        leftMax.append(heuristic(moveRight(i)))

    for i in rightMin:
        rightMax.append(heuristic(moveUp(i)))
        rightMax.append(heuristic(moveDown(i)))
        rightMax.append(heuristic(moveLeft(i)))
        rightMax.append(heuristic(moveRight(i)))

    sums = [sum(upMax), sum(downMax), sum(leftMax), sum(rightMax)]

    return sums.index(max(sums))

def minimax_search(grid):
    #find inital max
    up = moveUp(grid)
    down = moveDown(grid)
    left = moveLeft(grid)
    right = moveRight(grid)

    #find worst case for min
    minUp=copy.deepcopy(up)
    minDown=copy.deepcopy(down)
    minLeft=copy.deepcopy(left)
    minRight=copy.deepcopy(right)

    weightsIndexes = [(0,3),(0,2),(0,1),(0,0),(1,0),(1,1),(1,2),(1,3),(2,3),(2,2),(2,1),(2,0),(3,0),(3,1),(3,2),(3,3)]
    for i in weightsIndexes:
        if(up[i[0]][i[1]] == 0):
            minUp[i[0]][i[1]] = 2
            break
    for i in weightsIndexes:
        if(down[i[0]][i[1]] == 0):
            minDown[i[0]][i[1]] = 2
            break
    for i in weightsIndexes:
        if(left[i[0]][i[1]] == 0):
            minLeft[i[0]][i[1]] = 2
            break
    for i in weightsIndexes:
        if(right[i[0]][i[1]] == 0):
            minRight[i[0]][i[1]] = 2
            break
    
    #find leaves of min
    leaves1 = [heuristic(moveUp(minUp)), heuristic(moveDown(minUp)), heuristic(moveLeft(minUp)), heuristic(moveRight(minUp))]
    leaves2 = [heuristic(moveUp(minDown)), heuristic(moveDown(minDown)), heuristic(moveLeft(minDown)), heuristic(moveRight(minDown))]
    leaves3 = [heuristic(moveUp(minLeft)), heuristic(moveDown(minLeft)), heuristic(moveLeft(minLeft)), heuristic(moveRight(minLeft))]
    leaves4 = [heuristic(moveUp(minRight)), heuristic(moveDown(minRight)), heuristic(moveLeft(minRight)), heuristic(moveRight(minRight))]

    maxes = [max(leaves1),max(leaves2),max(leaves3),max(leaves4)]

    return maxes.index(max(maxes))


def NextMove(grid, step):
    if(step > 10000):
        printGrid(grid)
        return 4
    if(np.count_nonzero(np.array(grid)) == 16):
        similar = 0
        for i in range(3):
            for j in range(4):
                if grid[i][j] == grid[i+1][j]:
                    similar += 1
        for i in range(4):
            for j in range(3):
                if grid[i][j] == grid[i][j+1]:
                    similar += 1
        if(similar == 0):
            printGrid(grid)
            return 4
    movecode = minimax_search(grid)
    return movecode


def printGrid(grid):
    for i in grid:
        print(i)
    print()


def main():
    grid = make_initial_grid()
    print('Initial Grid:')
    printGrid(grid)
    step = 1
    while(True):
        print('Click Enter to play the next move')
        x = input()
        move = NextMove(grid, step)
        if(move == 0):
            print('Move was move up')
            grid = moveUp(grid)
        elif(move == 1):
            print('Move was move down')
            grid = moveDown(grid)
        elif(move == 2):
            print('Move was move left')
            grid = moveLeft(grid)
        elif(move == 3):
            print('Move was move right')
            grid = moveRight(grid)
        else:
            break
        print('Resulting Grid: ')
        printGrid(grid)
        grid = add_2_or_4(grid)
        print('New Grid: ')
        printGrid(grid)
        step += 1

main()