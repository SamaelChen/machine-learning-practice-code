import time
import random
import math


lower = [0, 1, 0, 4, 6]
upper = [2, 5, 6, 9, 20]


def costf(vec):
    return vec[0] - vec[1] ** 2 + vec[2] * 2 + vec[3] * 10 - vec[4] ** 3


def generatepop(lowerbound, upperbound, popsize=100):
    # 生成种群
    # lowerbound, upperbound很好理解，设定值域
    # popsize设定种群中个体数量
    pop = []
    for i in range(popsize):
        vec = [random.randint(lowerbound[j], upperbound[j])
               for j in range(len(lowerbound))]
        pop.append(vec)
    return pop


def crossover(gene1, gene2):
    # 基因重组
    i = random.randint(1, len(gene1) - 2)
    return gene1[0:i] + gene2[i:]


def mutate(gene, lowerbound, upperbound, mutation_prob):
    # 基因突变
    new = gene.copy()
    for i in range(len(new)):
        tmp = random.randint(lowerbound[i], upperbound[i])
        if random.random() > mutation_prob[i] and new[i] != tmp:
            new[i] = tmp
    return new


def geneticoptimize(lowerbound, upperbound, cost,
                    crossover_prob=0.3, elite=0.1, maxiter=100,
                    popsize=100, argmin=True):
    populations = generatepop(lowerbound, upperbound, popsize)
    # mutation_prob是一个list，表示每个DNA突变概率不同
    mutation_prob = [random.random() * 0.8 for i in range(len(lowerbound))]
    for i in range(maxiter):
        scores = [(cost(v), v) for v in populations]
        if argmin:
            scores.sort()
        else:
            scores.sort(reverse=True)
        ranked = [v for (s, v) in scores]
        # 适者生存
        elites_size = int(elite * popsize)
        populations = ranked[0:elites_size]
        # 开始产生后代
        while len(populations) < popsize:
            if random.random() > crossover_prob:
                c1 = random.randint(0, elites_size)
                c2 = random.randint(0, elites_size)
                populations.append(crossover(ranked[c1], ranked[c2]))
            else:
                c = random.randint(0, elites_size)
                populations.append(mutate(ranked[c], lowerbound,
                                          upperbound, mutation_prob))
        print(scores[0][0])
    return scores[0][1]


geneticoptimize(lowerbound=lower, upperbound=upper, cost=costf)
