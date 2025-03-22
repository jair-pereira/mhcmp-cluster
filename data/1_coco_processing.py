import argparse
import cocopp
import os
import glob
import pickle
import pandas as pd
import numpy as np
import sklearn.metrics
import warnings
# warnings.simplefilter('error', RuntimeWarning)

def step1_process_cocofile(path_coco):
    # cocopp: process coco files
    alg_str = " ".join([alg for alg in glob.glob(f"{path_coco}/*")])
    alg_data = cocopp.main(alg_str)

    # organize by algorithm and dimension
    data_coco = {}
    for key, data in alg_data.items():
        name = key[0]
        data_coco[name] = data.dictByDim() #organize by DIM

    return data_coco

def step2_extract_metrics(data_coco, fixbudget, targets):
    # extract the metrics of interest from the coco_objects and add them to a pandas dataframe
    # which is the area under the ecdf multiplied by the proportion of targets reached
    # curves are padded using the fixbudget so all algorithms have a data point at budget*dimension
    # https://numbbo.github.io/coco-doc/apidocs/cocopp/cocopp.pproc.DataSet.html#detERT
    # evals_with_simulated_restarts -> sort concatenate -> ECDF over all targets
    def process_algdata(alg_data, fixbudget, targets):
        # shortcut
        dimension = alg_data.dim
        algname   = alg_data.algId
        fid = alg_data.funcId
        
        fevals = alg_data.evals_with_simulated_restarts(targets) # 15 instances x 51 targets
        fevals = np.sort(np.concatenate(fevals))        
        y = np.linspace(start=0, stop=1, num=15*51) # fraction of function, target pair
        
        # consider simulated restarts up to our fixed budget
        mask = (fevals<=(fixbudget*dimension))
        if(sum(mask))>=2:
            fevals_ = fevals[mask]
            # padding
            fevals_ = np.append(fevals_, fixbudget*dimension+1) # +1 to ensure no repeated point
            y_ = np.append(y[mask], y[mask][-1]) # padding last y

            # ecdf curve
            ecdf_log10 = np.log10(fevals_/dimension)
            
            #normalize 
            minx = np.min(ecdf_log10)
            maxx = np.max(ecdf_log10)
            ecdf_nlog10 = (ecdf_log10-minx)/(maxx-minx)
            
            # ecdf auc multiplied by % targets 
            auc_log10 = sklearn.metrics.auc(x=ecdf_nlog10, y=y_)*y_[-1]
            
        # hard fail
        else:
            print(f"Hard Fail: {algname} f{fid} {dimension}d")
            auc_log10 = 0
            y_ = y
  
        return [algname, f"f{fid}d{dimension}", dimension, 
                auc_log10,
                y_,
        ]

        
    mycols = ["algorithm", "function", "dimension",
              "auc_log10", "y_",
    ]

    df = pd.DataFrame([], columns=mycols)
    for alg in data_coco.keys():
        for dimension in data_coco[alg]:
            for ds_f in data_coco[alg][dimension]:
                if ds_f.consistency_check() and ds_f.nbRuns()==15: # dont process unfinished exp
                    row = process_algdata(ds_f, fixbudget, targets)
                    df.loc[len(df.index)] = row
    return df

if __name__ == "__main__":
    ## arguments ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', dest='exp' , type=str, help="Experiment name (as in /results/<exp_name>/exdata)")
    args = parser.parse_args()

    ## globals ##
    args = {"EXP_ID"    : args.exp,
        
            "TARGETS"      : np.logspace(start=np.log10(1e+2), stop=np.log10(1e-8), num=51),
            "FIXED_BUDGET" : 2e+5, # todo: get from params/exp.json

            "PATH_DATA" : f"proc/{args.exp}",
            "PATH_RES"  : f"proc/{args.exp}/1_metrics",

            "PATH_COCO" : f"results/{args.exp}/exdata/",
    }
    ## output directories ##
    if not os.path.exists(args["PATH_DATA"]):
        os.mkdir("proc")
    if not os.path.exists(args["PATH_DATA"]):
        os.mkdir(args["PATH_DATA"])
    if not os.path.exists(args["PATH_RES"]):
        os.mkdir(args["PATH_RES"])

    ## coco processing ##
    data_coco = step1_process_cocofile(path_coco=args["PATH_COCO"])
    ## our processing ##
    df = step2_extract_metrics(data_coco, fixbudget=args["FIXED_BUDGET"], targets=args["TARGETS"])

    with open(f"{args['PATH_RES']}/bbob_metrics.pkl", 'wb') as f:
        pickle.dump(df, f)
    

