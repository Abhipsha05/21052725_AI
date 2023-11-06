import numpy
import imageio.v2 as imageio
import converter
import pygad
import matplotlib.pyplot

targetimage = imageio.imread('Pepsi.jpg')
targetimage = numpy.asarray(targetimage/255, dtype=numpy.float64)

targetchromosome = converter.imgtochromosome(targetimage)
print(targetchromosome)


def fitnessfunc(ga_instance, solution, solution_idx):

    fitness = numpy.sum(numpy.abs(targetchromosome-solution))
    fitness = numpy.sum(targetchromosome) - fitness
    return fitness


def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


ga_instance = pygad.GA(num_generations=20000,
                       num_parents_mating=10,
                       fitness_func=fitnessfunc,
                       sol_per_pop=20,
                       num_genes=targetimage.size,
                       init_range_low=0,
                       init_range_high=1,
                       mutation_probability=0.01,
                       parent_selection_type="sss",
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1.0,
                       on_generation=on_gen)

ga_instance.run()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(
        best_solution_generation=ga_instance.best_solution_generation))

result = converter.chromosometoimg(solution, targetimage.shape)
matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title("Using PyGAD for Reproducing Images")
matplotlib.pyplot.show()
