#!/usr/bin/env python
from __future__ import print_function
from math import sqrt
import sys
import os.path
import time
import subprocess
import glob
from collections import defaultdict
import pandas as pd
import numpy as np

sr_problems = ["polynomial"]
boolean_problems = ["boolean_true", "nparity", "comparator", "multiplexer", "random_boolean"]
classifier_problems = ["classifier"]
problems = sr_problems + boolean_problems + classifier_problems

n_varss = {
    "boolean_true": range(5, 9),
    "comparator": [6, 8, 10],
    "multiplexer": [6, 11],
    "nparity": range(5, 11),
    "random_boolean": range(5, 12),
    "classifier": [3, 4]
    }
n_iss = [3, 4]
n_oss = [2, 4, 8]
degrees = range(3, 11)
reps = ["GE", "GSGE", "GSGP"]
algos = ["hillclimb", "evolution"]


# over-ride settings above in order to run a partial experiment
# problems = boolean_problems
# reps = ["GSGE", "GSGP"]
# problems = sr_problems

def strify(x):
    if type(x) == str:
        return "'" + x + "'"
    else:
        return str(x)

def argstr(argd):
    return ','.join("%s=%s" % (k, strify(argd[k])) for k in sorted(argd))

# def print_stats(fit_vals):
#     print "mean", np.mean(fit_vals)
#     print "min", np.min(fit_vals)
#     print "sd", np.std(fit_vals)

def parse_results_file(filename):
    """This should be read in conjunction with
    GS_GP_GE.py:write_final_results. It just reads the last line,
    splitting it and mapping to float."""
    return map(float, open(filename).readlines()[-1].split(" "))


def ls_tool(**argd):
    args = argstr(argd)
    c = list()
    for filepath in glob.glob(outdir + "/*" + args + "_results.dat"):
        filename = os.path.basename(filepath)
        run, timestamp, seed_str, seed, rest = filename.split("_", 4) # split timestamp and seed
        c.append(int(seed))
    c.sort()
    print(args + ": " + str(c))
    assert(c == list(range(30)))

def process(**argd):
    args = argstr(argd)
    fits = []
    train_hits_percents = []
    largest_evers = []
    elapseds = []

    for filepath in glob.glob(outdir + "/*" + args + "_results.dat"):
        filename = os.path.basename(filepath)
        # print("filename", filename)
        run, timestamp, seed_str, seed, rest = filename.split("_", 4) # split timestamp and seed
        parameters = rest[:-12] # remove the "_results.out"
        # print("timestamp", timestamp)
        # print("parameters", parameters)
        # the result of this if-elif is currently unused
        if "hillclimb" in parameters:
            columns = ["gen", "evals", "min_fit", "min_fit_test", "length"]
        elif "evolution" in parameters:
            columns = ["gen", "evals", "min_fit", "mean_fit", "max_fit", "sd_fit", "min_fit_test", "best_size", "max_size"]
        train_fit, test_fit, train_hits_percent, test_hits_percent, log_largest_ever, elapsed = parse_results_file(filepath)
        fits.append(train_fit)
        train_hits_percents.append(train_hits_percent)
        largest_evers.append(log_largest_ever)
        elapseds.append(elapsed)

    # print(args)
    # print("# files of this type: ", len(fits)) # should be 30
    assert(len(fits) == 30)

    return {
        "mean": np.mean(fits),
        "stddev": np.std(fits),
        "percent_mean": np.mean(train_hits_percents),
        "percent_stddev": np.std(train_hits_percents)
        }

def run(**argd):
    seed = argd["seed"]
    del argd["seed"]
    args = argstr(argd)
    cmd = ["python", "GS_GP_GE.py", "run", args, str(seed), outdir]
    print(cmd)
    print(time.time())
    subprocess.call(cmd)

def do_all(what_to_do):
    results = {} # used when processing only
    for rep in reps:
        for algo in algos:
            for prob in problems:

                if prob in boolean_problems:
                    for n_vars in n_varss[prob]:
                        budget = 2 * n_vars * (2**n_vars)
                        p_count = max(int(sqrt(2**n_vars)), 10)
                        n_gens = budget // p_count
                        if what_to_do == "process":
                            results[rep,algo,prob,n_vars,n_gens,p_count] = process(
                                algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_gens=n_gens, p_count=p_count)
                        elif what_to_do == "ls":
                            ls_tool(algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_gens=n_gens, p_count=p_count)
                        else:
                            for iteration in iterations:
                                run(algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_gens=n_gens, p_count=p_count, seed=iteration)
                elif prob in sr_problems:
                    budget = 100000
                    if rep == "GE":
                        p_count = 1000
                    elif rep in ("GSGE", "GSGP"):
                        p_count = 20
                    else:
                        raise ValueError("Unexpected representation " + rep)
                    n_gens = budget // p_count
                    for degree in degrees:
                        if what_to_do == "process":
                            results[rep,algo,prob,degree,n_gens,p_count] = process(
                                algo=algo, prob=prob, rep=rep, degree=degree, n_gens=n_gens, p_count=p_count)
                        elif what_to_do == "ls":
                            ls_tool(algo=algo, prob=prob, rep=rep, degree=degree, n_gens=n_gens, p_count=p_count)
                        else:
                            for iteration in iterations:
                                run(algo=algo, prob=prob, rep=rep, degree=degree, n_gens=n_gens, p_count=p_count, seed=iteration)
                elif prob in classifier_problems:
                    for n_is in n_iss:
                        for n_os in n_oss:
                            for n_vars in n_varss[prob]:
                                budget = 2 * n_os * n_vars * (n_is ** n_vars)
                                p_count = max(int(sqrt(n_is ** n_vars)), 10)
                                n_gens = budget // p_count
                                if what_to_do == "process":
                                    results[rep,algo,prob,n_vars,n_is,n_os,n_gens,p_count] = process(
                                        algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_is=n_is, n_os=n_os, n_gens=n_gens, p_count=p_count)
                                elif what_to_do == "ls":
                                    ls_tool(algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_is=n_is, n_os=n_os, n_gens=n_gens, p_count=p_count)
                                else:
                                    for iteration in iterations:
                                        run(algo=algo, prob=prob, rep=rep, n_vars=n_vars, n_is=n_is, n_os=n_os, n_gens=n_gens, p_count=p_count, seed=iteration)

    if what_to_do == "process":
        print_latex(results)

def get_keys_such_that(d, t):
    result = []
    for k in d:
        if all(ti == ki or ti is None for ti, ki in zip(t, k)):
            result.append(k)
    return result

def print_latex(results):
    print(r"""
    \begin{tabular}{lc|rr|rr|rr|rr|rr|rr}\\
""")
    print(r"\multicolumn{2}{c}{} & ", end="")
    print(" & ".join(
        r"\multicolumn{2}{|c}{%s/%s}" % (rep, algo)
        for rep in reps
        for algo in ["HC", "Evo"]))

    print(r"""\\
  problem & size & avg & sd & avg & sd & avg & sd & avg & sd & avg & sd & avg & sd \\
\hline
""")

    for prob in problems:

        print(prob.replace("_", " "), end="")
        if prob in boolean_problems:
            for n_vars in n_varss[prob]:
                print(" & %d " % n_vars, end="")
                for rep in reps:
                    for algo in algos:
                        ks = get_keys_such_that(results, (rep, algo, prob, n_vars, None, None))
                        d = results[ks[0]]
                        print("& %.1f & %.1f " % (d["percent_mean"], d["percent_stddev"]), end="")
                print(r"\\")

        elif prob in sr_problems:
            for degree in degrees:
                print(" & %d " % degree, end="")
                for rep in reps:
                    for algo in algos:
                        ks = get_keys_such_that(results, (rep, algo, prob, degree, None, None))
                        d = results[ks[0]]
                        print("& %.1f & %.1f " % (d["percent_mean"], d["percent_stddev"]), end="")
                print(r"\\")

        elif prob in classifier_problems:
            for n_vars in n_varss[prob]:
                for n_is in n_iss:
                    for n_os in n_oss:
                        print(" & %d,%d,%d " % (n_vars, n_is, n_os), end="")
                        for rep in reps:
                            for algo in algos:
                                ks = get_keys_such_that(results, (rep, algo, prob, n_vars, n_is, n_os, None, None))
                                d = results[ks[0]]
                                print("& %.1f & %.1f " % (d["percent_mean"], d["percent_stddev"]), end="")
                        print(r"\\")
        print(r"\hline")
    s = r"""
\end{tabular}
"""
    print(s)


if __name__ == "__main__":
    cmd = sys.argv[1]
    outdir = sys.argv[2]
    if cmd == "run":
        iterations = eval(sys.argv[3])
    do_all(cmd)
