import itertools
from math import sqrt
import random

# Fitness functions for GS_GP_GE.py. Should be usable by any GP
# system, of course. Boolean, symbolic regression of 1-variable
# polynomials, and classification of a limited type are supplied, as
# in Moraglio et al, PPSN 2012.
#
# For each domain, we have a higher-order function -- fitness_boolean,
# fitness_sr, fitness_classifier -- which accepts a target function
# and some parameters and uses them to construct and return a fitness
# function proper. The fitness function proper accepts three arguments:
#

# individual (callable: to be evaluated)

# return_semantics (Boolean: if True, return the fn's semantics and
# its hits and its hits proportion; else just a fitness val. Default
# False)

# test_cases (Boolean: if True, evaluate on test cases; else on
# training cases. Default False)


# Boolean
#####################################################

def binlist2int(x):
    """Convert a list of binary digits to integer"""
    return int("".join(map(str, map(int, x))), 2)

def comparator(x):
    """Comparator function: input consists of two n-bit numbers. Output is
    0 if the first is larger or equal, or 1 if the second is larger."""
    n = len(x) // 2
    # no need to convert from binary. just use list comparison
    return x[:n] < x[n:]

def multiplexer(x):
    """Multiplexer: n address bits and 2^n data bits. Output the value of
    the data bit addressed by the address bits."""
    if len(x) == 3: n = 1
    elif len(x) == 6: n = 2
    elif len(x) == 11: n = 3
    elif len(x) == 20: n = 4
    else: raise ValueError(x)
    a = binlist2int(x[:n]) # get address bits, convert to int
    return x[n + a] # which data bit? offset by n

def nparity(x):
    'Parity function of any number of input variables'
    return x.count(True) % 2 == 1

def make_random_boolean_fn(n):
    """Make a random Boolean function of n variables."""
    outputs = [random.choice([False, True])
               for i in range(2**n)]
    def f(x):
        return outputs[binlist2int(x)]
    return f

def boolean_true(x):
    return True

def fitness_boolean(target, n):
    somelists = [[True,False] for i in range(n)]
    cases = list(itertools.product(*somelists)) # generate all input combinations for n variables
    target_values = [target(case) for case in cases]

    def f(individual, test_cases=False, return_semantics=False):
        'Determine the fitness of an individual. Lower is better.'
        # we ignore the values of test_cases because we use all possible
        # test cases in training

        try:
            fit = 0
            hits = []
            results = []
            for case, target_value in zip(cases, target_values):
                result = individual(case)
                results.append(result)
                hit = (result == target_value)
                hits.append(hit)
                fit += not hit
        except ValueError:
            # will be raised by invalid individuals
            fit = 2 ** n
            results = hits = [False for case in cases]
        if return_semantics:
            return fit, cases, results, hits, 100.0 * sum(hits) / float(len(cases))
        else:
            return fit
    return f

# Symbolic regression
######################################################

def quartic(x):

    """The original -- and best."""
    return x[0] + x[0]**2.0 + x[0]**3.0 + x[0]**4.0

def target_random_polynomial(degree):
    coefs = [2*(random.random() - 0.5) for i in range(degree)]
    def target(x):
        return sum(coefs[i] * (x[0] ** i) for i in range(degree))
    return target

def fitness_sr(target, train_X="random", test_X="random"):
    if train_X == "random":
        train_X = [(2*(random.random() - 0.5),) for i in range(20)]
    if test_X == "random":
        test_X = [(2*(random.random() - 0.5),) for i in range(20)]
    target_values = [target(case) for case in train_X]
    test_target_values = [target(case) for case in test_X]


    def f(individual, test_cases=False, return_semantics=False):
        'Determine the fitness of an individual. Lower is better.'
        # test_cases says whether to run on the training cases or the test cases
        # return_semantics says whether to return just the fitness (RMSE)
        # or the fitness, the values on the cases, and the error values
        results = []
        hits = []
        sumsqe = 0.0 # sum of squared errors
        if test_cases:
            cases = test_X
        else:
            cases = train_X
        try:
            for case, target_value in zip(cases, target_values):
                result = individual(case)
                results.append(result)
                sqerror = (result - target_value) ** 2
                hit = (abs(result - target_value) < 0.01)
                hits.append(hit)
                sumsqe += sqerror
            mse = sumsqe / len(cases)
            rmse = sqrt(mse)
        except (ZeroDivisionError, ValueError):
            # ZeroDivisionError will be raised by inds with a division
            # operator, but our grammar and GP operators don't create
            # division.

            # ValueError will be raised by invalid individuals
            # and by individuals where eval-ing the phenotype
            # causes a memory error
            rmse = 1.0e20
            results = [0.0 for case in cases]
            hits = [False for case in cases]
        if return_semantics:
            return rmse, cases, results, hits, 100.0 * sum(hits) / float(len(cases))
        else:
            return rmse
    return f

# Classifiers
######################################################

def target_classifier(n_vars, n_is, n_os):
    def target(x):
        return ((x[0] + x[1]) % n_os) + 1
    return target

def fitness_classifier(target, n_vars, n_is, n_os):
    somelists = [range(n_is) for i in range(n_vars)]
    cases = list(itertools.product(*somelists)) # generate all input combinations for n variables
    target_values = [target(case) for case in cases]

    def f(individual, test_cases=False, return_semantics=False):
        # we ignore the test_cases argument because in training we use
        # all possible cases
        try:
            fit = 0
            hits = []
            results = []
            for case, target_value in zip(cases, target_values):
                result = individual(case)
                results.append(result)
                hit = (result == target_value)
                hits.append(hit)
                fit += not hit
        except ValueError:
            # will be raised by invalid individuals
            fit = len(cases)
            hits = [False for case in cases]
            results = [0 for case in cases]
        if return_semantics:
            return fit, cases, results, hits, 100.0 * sum(hits) / float(len(cases))
        else:
            return fit
    return f
