import random

from deap import base
from deap import creator
from deap import tools

# 这里这个base.Fitness是干嘛的？？？
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 这里的list，fitness是参数，干嘛的？？？
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()  # base是个很基本的类啊！！！看来很重要

# Attribute generator: define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)  # 包含了0,1的随机整数。不明白这里是干嘛的？？？

# Structure initializers: define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,  # tools.initRepeat是干嘛的？？？
                 toolbox.attr_bool, 100)

# define the population to be a list of 'individual's
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized    注意！！！这里就定义了我们的适应度fitness函数啦！！！因为我们要解决的就是求和问题
# 只要返回一个值给我们的这个适应度函数啊！利用自带的sum函数！
# 这里取名为evalOneMax是因为这里的适应度函数就是我们后面要用来评价的依据，evaluate


def evalOneMax(individual):
    return sum(individual),

# ----------
# Operator registration
# ----------
# register the goal / fitness function
# 这里的toolbox register语句的理解：注册了一个函数evaluae依据的是后面的evalOneMax 理通了!!!


toolbox.register("evaluate", evalOneMax)

# 瞧瞧，这里的交叉函数，尼玛，crossover不用，非得用个mate，还以为是华为mate呢！你看，这里的交叉算子采用的函数，也是已经停供的，可以选择的
# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament,
                 tournsize=3)  # 这里选择的tournsize又是什么意思呢？

# ----------


def main():
    random.seed(64)
    # hash(64)is used

    # random.seed方法的作用是给随机数对象一个种子值，用于产生随机序列。
    # 对于同一个种子值的输入，之后产生的随机数序列也一样。
    # 通常是把时间秒数等变化值作为种子值，达到每次运行产生的随机系列都不一样

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)  # 定义了300个个体的种群！！！

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs   进化运行的代数！果然，运行40代之后，就停止计算了
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))  # 这时候，pop的长度还是300呢

    # Begin the evolution      开始进化了哈！！！注意注意注意！就是一个for循环里了！40次--代数
    for g in range(NGEN):
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    main()
