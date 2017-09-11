#!/usr/bin/env python

# GS_GP_GE.py
# James McDermott <jamesmichaelmcdermott@gmail.com>
# Alberto Moraglio
#
# Geometric Semantic Grammatical Evolution, as well as
# Geometric Semantic Genetic Programming, and plain old
# Grammatical Evolution
#
# See under __main__ for example usage.
#
# This implementation uses memoisation to avoid exponential runtime.
# See https://github.com/amoraglio/GSGP for more information on this,
# and also
# http://www.cs.put.poznan.pl/kkrawiec/smgp2014/uploads/Site/Moraglio2.pdf
#
# However there is a limitation. Querying the evolved function with
# new data, eg test data, is exponentially slow, because the memoised
# function has no results cached for these data. In Python, it can
# even be impossible, because of a recursion limit.
#
# One solution to this is to call each component function as we're
# evolving on both train and test sets. This is not cheating, because
# of course we ignore the results on the test set (well, we save them
# for post-run analysis, but we don't use them in evolution). However
# it is unrealistic, because in practice we want to be able to query
# our function on new data.
#
# TODO
# GP (mutation and crossover) (all problems) -- is this worth it?


import random
from math import sqrt, log, exp
import grammar
import itertools
import fitness
import sys
import time
import numpy as np

sys.setrecursionlimit(80000)

sr_problems = ["polynomial"]
boolean_problems = ["boolean_true", "nparity", "comparator", "multiplexer", "random_boolean"]
classifier_problems = ["classifier"]
problems = sr_problems + boolean_problems + classifier_problems
algo, rep, prob, n_vars, n_is, n_os, degree, p_count, n_gens = [None] * 9
initialise, crossover, mutation, fitness_fn, gram = [None] * 5

def test_pheno_fn(ind):
    """Test that evaluating the individual's phenotype
    string leads to the same function as is executed by
    the individual."""
    if ind.pheno() is None:
        # invalid individual can arise using GE init/mut/xover. we will call this
        # a pass
        print("Passed with None (was an invalid individual)")
    else:
        fit_from_pheno = fitness_fn(eval("lambda x: " + ind.pheno()))
        fit_from_fn = fitness_fn(ind)
        if fit_from_pheno != fit_from_fn:
            print("Failed")
            print("These should have been equal:")
            print("fit_from_pheno", fit_from_pheno)
            print("fit_from_fn   ", fit_from_fn)
        else:
            print("Passed")

def test_geno_pheno(ind):
    """Test that the individual's genotype leads to the
    stated phenotype string."""
    pheno_from_geno = GEmap(ind.geno()).pheno()
    pheno = ind.pheno()
    if pheno_from_geno != pheno:
        print("Failed")
        print("These should have been equal:")
        print("pheno_from_geno", pheno_from_geno)
        print("pheno          ", pheno)
    else:
        print("Passed")

def test_size_geno_len(ind):
    """Test that the reported size value is equal to the genotype
    length -- for GSGE, it should be, by construction. For GE, we don't
    expect them to be equal (so we won't run this test)."""
    if len(ind.geno()) == ind.size:
        print("Passed")
    else:
        print("Failed: genome is of length %d but size is %d" % (
            len(ind.geno()), ind.size))

def tests():
    """Several tests."""

    degree = 5
    n_vars = 5
    n_is = 2
    n_os = 4

    for rep in ["GSGE"]:
        print(rep)
        for prob in problems:
            print(prob)
            parse_params(prob=prob, rep=rep, degree=degree, n_vars=n_vars, n_is=n_is, n_os=n_os)
            print("fitness_fn", fitness_fn)

            for i in range(10):
                print("Initialisation")
                ind_geno = initialise(2)
                ind = GEmap(ind_geno)
                test_geno_pheno(ind)
                test_pheno_fn(ind)
                if rep == "GSGE":
                    test_size_geno_len(ind)

                print("Mutation")
                mut_ind = mutation(ind)
                test_geno_pheno(mut_ind)
                test_pheno_fn(mut_ind)
                if rep == "GSGE":
                    test_size_geno_len(mut_ind)

                print("Crossover")
                xover_ind = crossover(ind, mut_ind)
                test_geno_pheno(xover_ind)
                test_pheno_fn(xover_ind)
                if rep == "GSGE":
                    test_size_geno_len(xover_ind)

def memoize(f):
    f.cache = {}
    def decorated_function(*args):
        if args in f.cache:
            return f.cache[args]
        else:
            f.cache[args] = f(*args)
            return f.cache[args]

    decorated_function.geno = f.geno
    decorated_function.pheno = f.pheno
    decorated_function.size = f.size
    return decorated_function

def GEmap(geno):
    e, size = grammar.derive_string(gram, geno)
    if e is None:
        # invalid individual. note lambda can't raise exceptions, so use def.
        def f(x):
            raise ValueError

        # an interesting alternative would be to just return 0 always
        # f = eval('lambda x: 0.0')
    else:
        try:
            f = eval('lambda x: ' + e)
        except MemoryError:
            def f(x):
                raise ValueError
    f.geno = lambda: geno
    f.pheno = lambda: e
    f.size = size
    return f

######## Initialisation for GP #####################################

def initialise_boolean_gp(depth=4):
    'Create a random Boolean expression as a string.'
    if depth==1 or random.random()<1.0/(2**depth-1):
        return "x[%d]" % random.randrange(n_vars)
    if random.random()<1.0/3:
        return 'not' + ' ' + initialise_boolean_gp(depth-1)
    else:
        return '(' + initialise_boolean_gp(depth-1) + ' ' + random.choice(['and','or']) + ' ' + initialise_boolean_gp(depth-1) + ')'

def initialise_sr_gp(depth=4):
    """Create a random arithmetic expression as a string."""
    if depth==1 or random.random()<1.0/(2**depth-1):
        if random.random() > 1.0 / (n_vars + 1):
            return "x[%d]" % random.randrange(n_vars)
        else:
            return str(random.random())
    return '(' + initialise_sr_gp(depth-1) + ' ' + random.choice(['+', '-', '*']) + ' ' + initialise_sr_gp(depth-1) + ')'

def initialise_classifier_gp(depth=4):
    def cond(curdepth, maxdepth):
        # probabilities copied mindlessly from initialise_sr_gp
        if curdepth == maxdepth or random.random() < 1.0/(2**(maxdepth - curdepth - 1)):
            return "x[%d] == %d" % (random.randrange(n_vars), n_is)
        else:
            return cond(curdepth+1, maxdepth) + " and " + cond(curdepth+1, maxdepth)
    def cf(curdepth, maxdepth):
        if curdepth == maxdepth or random.random() < 1.0/(2**(maxdepth - curdepth - 1)):
            return str(random.randrange(n_os))
        else:
            return (cf(curdepth+1, maxdepth) + " if " +
                    cond(curdepth+1, maxdepth) + " else " + cf(curdepth+1, maxdepth))
    return cf(0, depth)



######## Initialisation for GSGP ###################################

initialise_boolean_gsgp = initialise_boolean_gp
initialise_sr_gsgp = initialise_sr_gp
initialise_classifier_gsgp = initialise_classifier_gp

######## Initialisation for GE: uses a generative genotype grammar ########

def initialise_boolean_ge(depth):
    """Generate a genotype for boolean.bnf which will give a valid
    phenotype, without wrapping, without unused codons."""

    lcm = gram.production_lcm
    if depth==1 or random.random()<1.0/(2**depth-1):
        return [2] + [random.randrange(lcm)] # x1 | x2 | etc
    if random.random()<1.0/3:
        return [1] + initialise_boolean_ge(depth-1) # not <expr>
    else:
        return [0] + initialise_boolean_ge(depth-1) + [random.randrange(lcm)] + initialise_boolean_ge(depth-1) # <expr> (and | or) <expr>

def initialise_sr_ge(depth):
    """Generate a genotype for sr.bnf which will give a valid
    phenotype, without wrapping, without unused codons."""

    lcm = gram.production_lcm
    if depth==1 or random.random()<1.0/(2**depth-1):
        if random.random() < 1.0/2:
            if n_vars == 1:
                return [1] # <expr> -> <var> -> x[0] (no codon needed for latter)
            else:
                return [1, random.randrange(lcm)] # <expr> -> <var> -> x[0] | x[1] ...
        else:
            return [2, random.randrange(lcm)] # <expr> -> <const> -> const
    else:
        return [0] + initialise_sr_ge(depth-1) + [random.randrange(lcm)] + initialise_sr_ge(depth-1) # <expr> (+ | - | * | /) <expr>

def initialise_classifier_ge(depth):
    """Generate a genotype for classifier.bnf which will give a valid
    phenotype, without wrapping, without unused codons."""

    lcm = gram.production_lcm
    if depth==1 or random.random()<1.0/(2**depth-1):
        return [1] + [random.randrange(lcm)] # <cf> -> <os> -> 0 | 1 | 2 .. (output symbols)
    else:
        # Note: this allows a single clause in each conditional
        return (
            [0] # (<cf> if <cond> else <cf>)
            + initialise_classifier_ge(depth-1) # <cf> -> terminals
            + [0, random.randrange(lcm), random.randrange(lcm)] # <cond> -> <var> == <is> -> terminals
            + initialise_classifier_ge(depth-1) # <cf> -> terminals
            )



### Initialisation for GSGE #######################################

# don't create a "dumb" integer-genome initialisation operator for
# plain (non-GS) GE, because it tends to produce bad results. for fair
# comparison between GE and GSGE, use identical "smart"
# initialisation, then compare different mutation/crossover operators.
initialise_boolean_gsge = initialise_boolean_ge
initialise_sr_gsge = initialise_sr_ge
initialise_classifier_gsge = initialise_classifier_ge



### Crossover for GE (all problems) #####################################

def crossover_ge(p1, p2):
    """Perform a GE crossover (single-point variable length at genome
    level)."""
    g1, g2 = p1.geno(), p2.geno()
    u1, u2 = p1.size, p2.size
    # do xover taking account of u1 & u2.
    i1 = random.randint(1, u1)
    i2 = random.randint(1, u2)
    cg = g1[:i1] + g2[i2:]
    return GEmap(cg)

def mutation_ge(p):
    """Perform a 1-int-flip mutation. Get the genotype, then mutate
    a single codon within the size range. Note this is
    not the per-gene mutation. Doing it this way to stay compatible
    with GSGE settings (mutation is carried out once per individual
    with a certain probability, after crossover)."""
    lcm = gram.production_lcm
    used = p.size
    idx = random.randrange(used)
    g = p.geno()
    g[idx] = random.randrange(lcm)
    return GEmap(g)



########## Crossover for GSGP ##############

def crossover_boolean_gsgp(p1,p2):
    """The crossover operator is a higher order function that takes
    parent functions and return an offspring function. The definitions
    of parent functions are _not substituted_ in the definition of the
    offspring function. Instead parent functions are _called_ from the
    offspring function. This prevents exponential growth.
    """
    mask = randfunct_gsgp()
    offspring = lambda x: (p1(x) and mask(x)) or (p2(x) and not mask(x)) # define an offspring as an anonymous function
    offspring.geno = lambda: '(('+ p1.geno() + ' and ' + mask.geno() + ') or (' + p2.geno() + ' and not ' + mask.geno() + '))'
    offspring.pheno = offspring.geno # in GP, genotype == phenotype
    offspring.size = p1.size + p2.size + mask.size + 4
    return memoize(offspring) # offspring

def crossover_sr_gsgp(p1,p2):
    # this is SGXE as defined in Moraglio et al PPSN 2012
    tr1 = random.random()
    offspring = lambda x: tr1 * p1(x) + (1-tr1) * p2(x)
    offspring.geno = lambda: '(('+ str(tr1) + '*' + p1.geno() + ') + (1 - ' + str(tr1) + ' * ' + p2.geno() + '))'
    offspring.pheno = offspring.geno # in GP, genotype == phenotype
    offspring.size = p1.size + p2.size + 7 # number of nodes
    return memoize(offspring) # offspring

def crossover_classifier_gsgp(p1,p2):
    var, sym = random.randrange(n_vars), random.randrange(n_is)
    cond = lambda x: x[var] == sym
    cond_geno = "x[%d] == %d" % (var, sym)
    offspring = lambda x: p1(x) if cond(x) else p2(x)
    offspring.geno = lambda: p1.geno() + ' if ' + cond_geno + ' else ' + p2.geno()
    offspring.pheno = offspring.geno
    offspring.size = p1.size + p2.size + 4 # cond is "if x1 == 3"
    return memoize(offspring)


########## Crossover for GSGE ##############

def crossover_boolean_gsge(p1,p2):
    """
    The crossover operator is a higher order function that takes
    parent functions and return an offspring function. The definitions
    of parent functions are _not substituted_ in the definition of the
    offspring function. Instead parent functions are _called_ from the
    offspring function. This prevents exponential growth.
    """
    mask = randfunct_ge_gsge()
    offspring = lambda x: (p1(x) and mask(x)) or (p2(x) and not mask(x)) # define an offspring as an anonimous function with 1 argument of length n
    offspring.pheno = lambda: '(('+ p1.pheno() + ' and ' + mask.pheno() + ') or (' + p2.pheno() + ' and not ' + mask.pheno() + '))'
    offspring.geno = lambda: [0, 0] + p1.geno() + [0] + mask.geno() + [1, 0] +  p2.geno() + [0, 1] + mask.geno()
    offspring.size = p1.size + p2.size + 7 + 2 * mask.size
    return memoize(offspring)

def crossover_classifier_gsge(p1,p2):
    """
    O = IF Rcond THEN P1 ELSE P2

    where P1, P2 and O are parent classifiers and offspring
    classifier, respectively; Rcond is a random condition depending on
    one or more input variables.
    """
    varidx = random.randrange(n_vars)
    insym = random.randrange(n_is)
    offspring = lambda x: (p1(x) if x[varidx] == insym else p2(x))
    offspring.pheno = lambda: '('+ p1.pheno() + ' if x[' + str(varidx) + '] == ' + str(insym) + ' else ' + p2.pheno() + ')'
    offspring.geno = lambda: (
        [0] # <cf> -> (<cf> if <cond> else <cf>)
        + p1.geno() # <cf> -> p1
        + [0]       # <cond> -> <var> == <is>
                    # <var> -> x[<varidx>], no codon needed
        + [varidx]  # <varidx> -> varidx
        + [insym]   # <is> -> insym
        + p2.geno() # <cf> -> p2
        )
    offspring.size = p1.size + p2.size + 4
    return memoize(offspring)

def crossover_sr_gsge(p1,p2):
    """The crossover operator is a higher order function that takes
    parent functions and return an offspring function. The definitions
    of parent functions are _not substituted_ in the definition of the
    offspring function. Instead parent functions are _called_ from the
    offspring function. This prevents exponential growth.
    """
    # this is SGXE as defined in Moraglio et al PPSN 2012
    tr1_int = random.randint(0, 10)
    tr1 = tr1_int / 10.0
    offspring = lambda x: ((tr1 * p1(x)) + ((1-tr1) * p2(x)))
    offspring.pheno = lambda: '((' + str(tr1) + ' * ' + p1.pheno() + ') + ((1.0 - ' + str(tr1) + ') * ' + p2.pheno() + '))'
    offspring.geno = lambda: (
              [0] # <expr> -> (<expr> <biop> <expr>)
            + [0] # <expr> -> (<expr> <biop> <expr>)
            + [2, tr1_int] # <expr> -> <const> -> tr1
            + [2] # <biop> -> *
            + p1.geno() # <expr> -> p1.geno
            + [0] # <biop> -> +
            + [0] # <expr> -> (<expr> <biop> <expr>)
            + [0] # <expr> -> (<expr> <biop> <expr>)
            + [2, 10] # <expr> -> <const> -> 1.0
            + [1] # <biop> -> -
            + [2, tr1_int] # <expr> -> <const> -> tr1
            + [2] # <biop> -> *
            + p2.geno() # <expr> -> p2.geno
            )
    offspring.size = 14 + p1.size + p2.size
    return memoize(offspring)




################# Mutation for GSGE ####################

# def generate_conjunction(n, n_is):
#     """Generate the partial genotype for a random condition, of the
#     form: conjunction of n variables where each can take on n_is
#     possible values. Assume that the start symbol is <cond>"""
#     assert n >= 2
#     result = []
#     for v in range(n - 1):
#         result += [1, 0] # <cond> -> <cond> and <var> == <is> -> <var> == <is> and <var> == <is>
#         result += [n, random.rangrange(n_is)] # <var> == <is> -> x[1] == 3 (eg)
#     result += [0, random.randrange(n), random.rangrange(n_is)] # last one: <var> == <is> -> x[2] == 1 (eg)
#     return result

def generate_mintermgeno(n):
    """Generate the genotype for a random minterm of n variables. For example,
    for 3 variables, we would have:

    mintermgeno = [0] + random.choice([[],[1]]) + [2, 0] + [0, 0] + random.choice([[],[1]]) + [2, 1, 0] + random.choice([[],[1]]) + [2, 2]
    """

    assert n > 2
    result = [0] # <expr> <biop> <expr>
    for i in range(n-2):
        result += random.choice([[], [1]]) # do-nothing, or negate
        result += [2, i, 0, 0] # <var>, x[i], and, <expr> <biop> <expr>,
    result += random.choice([[], [1]]) # do-nothing, or negate
    result += [2, n-2, 0]  # <var>, x[n-2], and
    result += random.choice([[], [1]]) # do-nothing, or negate
    result += [2, n-1] # <var>, x[n-1]
    return result

def mutation_boolean_gsge(p):
    """The mutation operator is a higher order function. The parent
    function is called by the offspring function."""
    mintermgeno = generate_mintermgeno(n_vars)
    minterm = GEmap(mintermgeno) # express phenotype
    if random.random()<0.5:
        offspring = lambda x: p(x) or minterm(x) # 1 argument of length n
        offspring.pheno = lambda: '(' + p.pheno() + ' or ' + minterm.pheno() + ')'
        offspring.geno = lambda: [0] + p.geno() + [1] + minterm.geno()
        offspring.size = p.size + 2 + minterm.size
    else:
        offspring = lambda x: p(x) and not minterm(x) # 1 argument of length n
        offspring.pheno = lambda: '(' + p.pheno() + ' and not ' + minterm.pheno() + ')'
        offspring.geno = lambda: [0] + p.geno() + [0, 1] + minterm.geno()
        offspring.size = p.size + 3 + minterm.size
    return memoize(offspring)

def mutation_sr_gsge(p):
    """The mutation operator is a higher order function. The parent
    function is called by the offspring function."""
    ms = 0.1 * 0.1 * 0.1
    tr1 = randfunct_ge_gsge(depth=6)
    tr2 = randfunct_ge_gsge(depth=6)
    offspring = lambda x: (p(x) + (ms * (tr1(x) - tr2(x))))
    offspring.pheno = lambda: '(' + p.pheno() + ' + (((0.1 * 0.1) * 0.1) * (' + tr1.pheno() + ' - ' + tr2.pheno() + ')))'
    offspring.geno = lambda: ([0] # <expr> -> (<expr> <biop> <expr>)
                              + p.geno() # <expr> -> p.geno
                              + [0] # <biop> -> +
                              + [0] # <expr> -> (<expr> <biop> <expr>)
                              + [0, 0, 2, 1, 2, 2, 1, 2, 2, 1] # <expr> -> ... -> 0.1 * 0.1 * 0.1
                              + [2] # <biop> -> *
                              + [0] # <expr> -> (<expr> <biop> <expr>)
                              + tr1.geno() # <expr> -> tr1.geno
                              + [1] # <biop> -> -
                              + tr2.geno() # <expr> -> tr2.geno
                              )
    offspring.size = p.size + tr1.size + tr2.size + 16
    return memoize(offspring)

def mutation_classifier_gsge(p):
    """The mutation operator is a higher order function. The parent
    function is called by the offspring function.

    O = IF Rcond THEN ROS ELSE P

    where P and O are parent and offspring classifiers, respectively;
    Rcond is a conjunctive condition in which each input variable
    appears exactly once; ROS is a random output symbol.

    To implement this in Python, without indentation, we could write
    an expression of this "backwards" form, using Python's odd ternary
    operator "x if y else z":

    <cf> ::= (os if (x1 == is1 and x1 == is2 and x3 ...) else p)
    """

    lcm = gram.production_lcm
    rand_vals = [random.randrange(n_is) for i in range(n_vars)]
    ros_int = random.randrange(lcm)
    ros = ros_int % n_os

    offspring = lambda x: (ros if all(var == val
                                      for var, val in zip(x, rand_vals))
                           else p(x))
    cond_geno = (
        ([1] * (n_vars - 1)) # <cond> -> <cond> and <var> == <is>, many times
        + [0]) # <cond> -> <var> == <is>
    for i, val in enumerate(rand_vals):
        cond_geno += [i, val] # <var> -> x[<varidx>] (no codon needed) -> 0 | 1 ... and then <is> -> 0 | 1 ...
    cond_pheno = ' and '.join('x[%d] == %d' % (i, val) for i, val in enumerate(rand_vals))
    offspring.pheno = lambda: '(' + str(ros) + ' if ' + cond_pheno + ' else ' + p.pheno() + ')'
    offspring.geno = lambda: (
        [0] # <cf> -> (<cf> if <cond> else <cf>)
        + [1, ros_int] # <cf> -> <os> -> ros
        + cond_geno # <cond> -> Rcond
        + p.geno() # <cf> -> p
        )
    offspring.size = p.size + len(cond_geno) + 3
    return memoize(offspring)






### Mutation for GSGP ###################################

def mutation_boolean_gsgp(p):
    """The mutation operator is a higher order function. The
    parent function is called by the offspring function."""
    mintermexpr = ' and '.join([random.choice(['', 'not ']) + ("x[%d]" % i) for i in range(n_vars)]) # random minterm of n variables
    minterm = eval('lambda x: ' + mintermexpr)
    minterm.geno = lambda: mintermexpr
    if random.random()<0.5:
        offspring = lambda x: p(x) or minterm(x)
        offspring.geno = lambda: '(' + p.geno() + ' or ' + minterm.geno() + ')'
    else:
        offspring = lambda x: p(x) and not minterm(x)
        offspring.geno = lambda: '(' + p.geno() + ' and not ' + minterm.geno() + ')'
    # This counts the number of variable appearances and multplies by
    # 2 to approximate the number of x/and/or nodes in minterm, then
    # adds the number of nots, then adds 2 for the combination with
    # the original. Won't be exact but ok.
    offspring.size = p.size + n_vars * 2 + mintermexpr.count("not ") + 2
    offspring.pheno = offspring.geno # in GP, genotype == phenotype
    return memoize(offspring)

def mutation_sr_gsgp(p):
    ms = 0.001
    tr1 = randfunct_gsgp(depth=6)
    tr2 = randfunct_gsgp(depth=6)
    offspring = lambda x: (p(x) + (ms * (tr1(x) - tr2(x))))
    offspring.geno = lambda: ('(' + p.geno() + ' + (' + str(ms) + '*' +
                              '(' + tr1.pheno() + ' - ' + tr2.pheno() + ')))')
    offspring.pheno = offspring.geno
    offspring.size = p.size + tr1.size + tr2.size + 4
    return memoize(offspring)

def mutation_classifier_gsgp(p):
    ros = random.randrange(n_os)
    rand_vals = [random.randrange(n_is) for i in range(n_vars)]
    offspring = lambda x: (ros if all(var == val
                                      for var, val in zip(x, rand_vals))
                           else p(x))
    offspring.geno = lambda x: (
        '(' + str(ros) + ' if ' + ' and '.join('x[%d] == %d' % (i, val)
                                               for i, val in enumerate(rand_vals)))
    offspring.pheno = offspring.geno
    # "x[i] = n" happens n_vars times => n_vars * 3
    # "and" happens n_vars -1 times => n_vars -1
    # ros => +1
    # if => +1
    # else => +1
    offspring.size = p.size + n_vars * 3 + (n_vars - 1) + 3
    return memoize(offspring)



############### Population/algorithm stuff ####################

def randfunct_gsgp(depth=4):
    re = initialise(depth)
    f = eval("lambda x: " + re)
    f.geno = f.pheno = lambda: re # geno and pheno are functions
    # occurrences of "(" is number of internal nodes. multiply by 2 to
    # approximate total number of nodes. +1 in case tree is just a
    # const or var.
    f.size = re.count("(") * 2 + 1
    return memoize(f)

def randfunct_ge_gsge(depth=4):
    'Create a terminating function at a genotype level, without feedback from GEmap'
    y=initialise(depth) # generates a random (better distributed) genotype, cut-to-size, and terminating without using GEmap
    f=GEmap(y) # express the genotype
    return memoize(f)

def population(p_count):
    'Create a population.'
    return [ randfunct() for x in range(p_count) ]


def trunc_float(x):
    try:
        return float(x)
    except OverflowError:
        return float("inf")

def grade(pop):
    'Find stats on fitness and size in a population.'
    fitvals = np.array([fitness_fn(x) for x in pop])
    best_idx = np.argmin(fitvals)
    min_f = np.min(fitvals)
    max_f = np.max(fitvals)
    med_f = np.median(fitvals)
    sd_f = np.std(fitvals)

    # note this is the fitness on the test set of the individual which
    # is best on the training set -- not the best of all test set
    # fitnesses
    min_f_test = fitness_fn(pop[best_idx], test_cases=True)

    # we store the log of sizes to avoid overflow
    log_max_s = log(max(x.size for x in pop))
    log_best_s = log(pop[best_idx].size)
    return list(map(float, [min_f, med_f, max_f, sd_f, min_f_test, log_best_s, log_max_s]))

def evolve(pop, trunc_ratio=0.5, random_select=0.0, mutate=0.01):
    graded = [ (fitness_fn(x), x) for x in pop]
    graded = [ x[1] for x in sorted(graded, key=lambda x: x[0])]
    nparents = int(len(graded)*trunc_ratio)
    parents = graded[:nparents]

    # randomly add other individuals to promote genetic diversity
    for individual in graded[nparents:]:
        if random_select > random.random():
            parents.append(individual)

    # crossover parents to create children
    desired_length = len(pop)
    children = []
    while len(children) < desired_length:
        par = random.sample(parents, 2) # pick two random parents
        child = crossover(par[0], par[1])

        # and possibly mutate
        if mutate > random.random():
            child = mutation(child)

        children.append(child)
    return children


def parse_params(**params):

    global algo, rep, prob, n_vars, n_is, n_os, degree, p_count, n_gens
    global initialise, crossover, mutation, randfunct, fitness_fn, gram
    globals().update(params) # gives values to algo, n_vars, etc

    if rep == "GSGP":
        randfunct = randfunct_gsgp
    elif rep == "GE":
        randfunct = randfunct_ge_gsge
    elif rep == "GSGE":
        randfunct = randfunct_ge_gsge
    else:
        raise ValueError("Unexpected rep " + rep)


    if prob in ["boolean_true", "comparator", "nparity", "multiplexer", "random_boolean"]:
        if rep == "GSGP":
            initialise = initialise_boolean_gsgp
            crossover = crossover_boolean_gsgp
            mutation = mutation_boolean_gsgp
        elif rep == "GE":
            initialise = initialise_boolean_ge
            crossover = crossover_ge # no need for problem-specific xover
            mutation = mutation_ge # no need for problem-specific mutation
        elif rep == "GSGE":
            initialise = initialise_boolean_gsge
            crossover = crossover_boolean_gsge
            mutation = mutation_boolean_gsge
        else:
            raise ValueError("Unexpected rep " + rep)
        if prob == "random_boolean":
            fitness_fn = fitness.fitness_boolean(fitness.make_random_boolean_fn(n_vars), n_vars)
        else:
            fitness_fn = fitness.fitness_boolean(eval("fitness."+prob), n_vars)
        gram = grammar.Grammar("boolean.bnf", n_vars=n_vars)

    elif prob == "polynomial":
        n_vars = 1
        if rep == "GSGP":
            initialise = initialise_sr_gsgp
            crossover = crossover_sr_gsgp
            mutation = mutation_sr_gsgp
        elif rep == "GE":
            initialise = initialise_sr_ge
            crossover = crossover_ge
            mutation = mutation_ge
        elif rep == "GSGE":
            initialise = initialise_sr_gsge
            crossover = crossover_sr_gsge
            mutation = mutation_sr_gsge
        else:
            raise ValueError("Unexpected rep " + rep)
        target = fitness.target_random_polynomial(degree)
        fitness_fn = fitness.fitness_sr(target, "random")
        gram = grammar.Grammar("sr.bnf", n_vars=n_vars)

    elif prob == "classifier":
        if rep == "GSGP":
            initialise = initialise_classifier_gsgp
            crossover = crossover_classifier_gsgp
            mutation = mutation_classifier_gsgp
        elif rep == "GE":
            initialise = initialise_classifier_ge
            crossover = crossover_ge
            mutation = mutation_ge
        elif rep == "GSGE":
            initialise = initialise_classifier_gsge
            crossover = crossover_classifier_gsge
            mutation = mutation_classifier_gsge
        else:
            raise ValueError("Unexpected rep " + rep)
        target = fitness.target_classifier(n_vars, n_is, n_os)
        fitness_fn = fitness.fitness_classifier(target, n_vars, n_is, n_os)
        # can pass in values for n_is etc, eg:
        gram = grammar.Grammar("classifier.bnf",
                               n_vars=n_vars, n_is=n_is, n_os=n_os)

    else:
        raise ValueError("Unexpected problem type " + prob)

def evolution(outfile=None):
    start_time = time.time()
    p = population(p_count)

    history = [[0.0, 0.0] + grade(p)]
    for i in range(1, n_gens + 1):
        p = evolve(p, trunc_ratio=0.5, random_select=0.0, mutate=1.0)
        history.append([float(i), float(i * p_count)] + grade(p))

        if i % 100 == 0:
            print(history[-1])

    cur = min(p, key=fitness_fn)
    log_longest = max([h[-1] for h in history])
    elapsed = time.time() - start_time

    train_results = fitness_fn(cur, return_semantics=True, test_cases=False)
    test_results = fitness_fn(cur, return_semantics=True, test_cases=True)

    write_final_results(history, log_longest, elapsed, train_results, test_results, outfile)

def hillclimb(outfile=None):
    # in hillclimb, we run for p_count * ngens steps. that is, our
    # true population is 1, so p_count is ignored. but it affects the
    # number of generations we actually run. it is also used to
    # indicate how often to store the history -- in order to make
    # comparison with evolution easier.
    start_time = time.time()
    cur = randfunct()
    curfit = fitness_fn(cur)
    curfit_test = fitness_fn(cur, test_cases=True)
    log_longest = log(cur.size)
    history = [[0.0, 0.0, float(curfit), float(curfit_test), log_longest]]

    # Start from 1 because we've already evaluated 1.
    for i in range(1, n_gens * p_count + 1):
        cand = mutation(cur)
        candfit = fitness_fn(cand)
        candfit_test = fitness_fn(cand, test_cases=True)
        log_candlen = log(cur.size)
        if candfit < curfit:
            curfit = candfit
            curfit_test = candfit_test
            cur = cand
        if log_candlen > log_longest:
            log_longest = log_candlen

        # To avoid storing a huge history, and to allow easier
        # comparison/plotting versus evolution we store fitness only
        # every p_count steps.
        #
        # we store the log of the largest length ever seen
        if i % p_count == 0:
            history.append([i // p_count, i] + [float(curfit), float(curfit_test), log_longest])
            print(history[-1])

    elapsed = time.time() - start_time

    train_results = fitness_fn(cur, return_semantics=True, test_cases=False)
    test_results = fitness_fn(cur, return_semantics=True, test_cases=True)

    write_final_results(history, log_longest, elapsed, train_results, test_results, outfile)

def write_final_results(history, log_largest_ever, elapsed, train_results, test_results, outfile):
    train_fit, train_cases, train_results, train_hits, train_hits_percent = train_results
    test_fit, test_cases, test_results, test_hits, test_hits_percent = test_results
    if outfile:
        basename = outfile + "_" + sys.argv[2]
        open(basename + "_parameters.dat", "w").write(sys.argv[2])
        history = np.array(history, dtype=float)
        np.savetxt(basename + "_history.dat", history)
        f = open(basename + "_results.dat", "w")
        f.write(" ".join(map(str, train_cases)) + "\n")
        f.write(" ".join(map(str, test_cases)) + "\n")
        f.write(" ".join(map(str, train_results)) + "\n")
        f.write(" ".join(map(str, test_results)) + "\n")
        f.write(" ".join(map(str, train_hits)) + "\n")
        f.write(" ".join(map(str, test_hits)) + "\n")
        f.write("%f %f %f %f %f %f" % (
            train_fit,
            test_fit,
            train_hits_percent,
            test_hits_percent,
            log_largest_ever,
            elapsed))
        f.close()
    else:
        # for datum in history:
        #     print(datum)
        print("Best fitness (train): %.3f" % train_fit)
        print("Best fitness (test): %.3f" % test_fit)
        print("Hits (train): %.2f%%" % train_hits_percent)
        print("Hits (test): %.2f%%" % test_hits_percent)
        print("Log(largest ever): %g" % log_largest_ever)
        print("Elapsed: %fs" % elapsed)

    # Uncomment these at your peril: both grow quickly with generations
    # print(cur.geno())
    # print(cur.pheno())

def main():
    params = sys.argv[2]
    print(params)
    if len(sys.argv) >= 4:
        seed = int(sys.argv[3])
        random.seed(seed)
        print("Setting seed = %d" % seed)
    if len(sys.argv) == 5:
        outfile = sys.argv[4] + "/run_" + str(time.time()) + "_seed_" + str(seed)
    else:
        outfile = None
    # make sure to do this *after* random.seed() above
    params = parse_params(**eval("dict(" + sys.argv[2] + ")"))
    if algo == "hillclimb": hillclimb(outfile)
    elif algo == "evolution": evolution(outfile)

if __name__ == "__main__":
    """Usage:

$ python GS_GP_GE.py run "algo=('hillclimb'|'evolution'),rep=('GSGP'|'GE'|'GSGE'),prob=('polynomial'|'boolean_true'|'nparity'|'comparator'|'multiplexer'|'random_boolean'|'classifier'),n_vars=<n>,n_is=<n>,n_os=<n>,degree=<n>,p_count=<n>,n_gens=<n>" <seed> <outdir>

    For example:

$ python GS_GP_GE.py run "algo='evolution',rep='GSGE',prob='polynomial',n_vars=1,degree=6,p_count=200,n_gens=40"

$ python GS_GP_GE.py run "algo='hillclimb',rep='GSGP',prob='boolean_true',n_vars=3,p_count=10,n_gens=10"

$ python GS_GP_GE.py test

p_count is the population size. In hillclimb, we run for n_gens *
p_count steps.

degree applies only to the polynomial problem.

n_is and n_os are the number of input/output symbols in the classifier
problem.

If you supply an outdir, it will write result files there, else will
print to screen. The results consist of the generation number, the
number of fitness evaluations, the fitness and the size over the
generations (for hillclimbing) or the generation number, the number of
fitness evaluations, the best fitness, median fitness, max fitness,
stddev fitness, size of best, and largest size, over the generations
(for evolution). Will also write out the input parameters, and the
elapsed time, and the semantics (on train and test cases) and number
of hits (as a percentage of train and of test cases).

    """
    if sys.argv[1] == "run":
        main()
    elif sys.argv[1] == "test":
        tests()
