import numpy as np
import cocoex
import json
import argparse
import itertools
import functools
import time
import os
import sys
import pandas as pd
from src.pso import PSO
import src.mutation as Mutation

def load_settings_from_file(filelist):
    settings = {}
    for filename in filelist:
        with open(filename) as file:
            settings |= json.load(file)

    return settings

def mk_dir(alg_name, alg_params):
    if not os.path.exists("results"):
        os.mkdir("results")

    dir_results = [mk_dir_b(f"results/{alg_name}")]

    for n, (w,c), p in alg_params:
        dir_results += [f"{dir_results[0]}/{alg_name}_{n}_{w}_{p}"]
        os.mkdir(dir_results[-1])

    return dir_results

def mk_dir_b(alg_name):
    i=0
    while os.path.exists(f"{alg_name}_{i:03d}"):
        i+=1

    dir_results = f"{alg_name}_{i:03d}"
    os.mkdir(dir_results)

    return dir_results

def main(args):
    ## arguments ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='experiment' , type=str, help="json file with the experiment settings")
    parser.add_argument('-p', dest='problems'   , type=str, help="json file with the benchmark settings")
    parser.add_argument('-a', dest='algsettings', type=str, help="json file with the algorithm settings")
    args = parser.parse_args()

    ## load exp settings ##
    settings = load_settings_from_file([args.experiment, args.problems, args.algsettings])
    alg_params = list(itertools.product(settings["n"], zip(settings["w"],settings["c"]), settings["mut_p"]))
    dir_results = mk_dir(settings['name'], alg_params)
    psomutation = Mutation.mut_dict[settings["mutation"]]

    for (n, (w,c), p), directory in zip(alg_params, dir_results[1:]):
        # algorithm name
        alg_name = f"{settings['name']}_{n}_{w}"
        print(alg_name)

        # benchmark suite and observer
        suite = cocoex.Suite(settings["suite"], "", 
                             f"function_indices:{settings['functions']} dimensions:{settings['dimensions']} instance_indices:{settings['instances']}"
        )
        observer = cocoex.Observer(settings["suite"], f"result_folder: {alg_name}")

        for problem in suite:
            start = time.time()
            problem.observe_with(observer)

            history = [] # my log

            # initialize rng 
            rng = np.random.default_rng(settings["seed"])

            ## main loop ##
            maxnfe = int(settings["base_maxnfe"]*problem.dimension)
            do_restart = True
            n_restart  = 0
            # while budget or problem not solved
            while problem.evaluations<maxnfe and not problem.final_target_hit:
                if do_restart:
                    n_restart += 1

                    # initialize the algorithm
                    pso = PSO(rng, 
                              n, w, c, c,
                              problem.dimension, problem.lower_bounds, problem.upper_bounds
                    )
                    pso.mutation = functools.partial(psomutation, self=pso, prob=p)
                    # initial set of solutoins
                    X = pso.ask0()
                else:
                    # generate new candidate solutions
                    X = pso.ask()

                # evaluate
                Y = [problem(xi) for xi in X]
                pso.tell(Y)

                # my log
                #history += [f"{problem.evaluations},{n_restart},{problem.final_target_hit},{pso.gbest_f},{';'.join(map(str, pso.gbest_x))},{np.mean(X, axis=0)},{np.std(X, axis=0)}"]
                history += [[problem.evaluations, n_restart, problem.final_target_hit, pso.gbest_f, 
                             ";".join(map(str, pso.gbest_x)), 
                             ";".join(map(str, np.mean(X, axis=0))), 
                             ";".join(map(str, np.std(X, axis=0)))
                ]]

                # restart if
                ## worst and best fitness are the same
                cond_a = np.max(Y) - np.min(Y) < 1e-12 
                ## or the best fitness has been the same for 1000 iterations
                cond_b = len(set(np.array(history)[:,3][int(-1e+4):])) == 1 if len(history)>=1e+4 else False 
                do_restart = cond_a or cond_b

            # my log
            print("\t", problem.id, problem.final_target_hit, problem.evaluations, time.time()-start)

            # save history
            ## since this is not used in the paper, leave it commented to avoid writting many files
            # df = pd.DataFrame(np.array(history), columns=["nfe", "n_restart", "final_target_hit", "best_f", "best_x", "mean_x", "std_x"])
            # df.to_csv(f"{directory}/{problem.id}.csv", index=False)
            # with open(f"{directory}/{problem.id}.csv", "a") as file:
            #     file.write("\n".join(history))





if __name__ == "__main__":
   main(sys.argv[1:])
