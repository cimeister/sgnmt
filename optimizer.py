import subprocess, os
import re
import threading
import string
import pickle
import torch

from cam.sgnmt import io, decode_utils
from cam.sgnmt.ui import get_parser, parse_args

import sacrebleu
from mosestokenizer import MosesDetokenizer

import ray
from ray.tune import run, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch


def detokenize(f):
    with MosesDetokenizer('fr') as detokenize:
        lines = list(open(f))
        return [detokenize(s.split()) for s in lines]

#REFS = [r.rstrip('\n') for r in list(open('data/iwslt_data/valid.detok.en'))]
#REFS = detokenize('data/wmt14.en-fr.fconv-py/valid.fr')
REFS = [r.rstrip('\n') for r in list(open('data/newstest/test.detok.fr'))]

# DEFAULT_ARGS = ("--fairseq_path /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.fconv-py/model.pt"
#     " --fairseq_lang_pair en-fr --src_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.fconv-py/wmap.bpe.en"
#     " --trg_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.fconv-py/wmap.bpe.fr --input_method file"
#     " --src_test /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.fconv-py/valid.bpe.en "
#     "--preprocessing word --postprocessing bpe@@ --decoder dijkstra_ts --beam 20 "
#     "--guido variance --outputs text")

# DEFAULT_ARGS = ("--fairseq_path /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.joined-dict.transformer/model.pt"
#     " --fairseq_lang_pair en-fr --src_wmap /home/meistecl/Documents/repositories/sgnmt/data//wmt14.en-fr.joined-dict.transformer//wmap.bpe.en"
#     " --trg_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.joined-dict.transformer/wmap.bpe.fr --input_method file"
#     " --src_test /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.fconv-py/valid.bpe.en "
#     "--preprocessing word --postprocessing bpe@@ --outputs text --decoder batch --beam 200")

# DEFAULT_ARGS = ("--fairseq_path /home/meistecl/Documents/repositories/sgnmt/data/ckpts/iwslt/cond_model.pt "
#     "--fairseq_lang_pair de-en --src_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmaps/wmap.iwslt.de "
#     "--trg_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmaps/wmap.iwslt.en --input_method file "
#     "--src_test /home/meistecl/Documents/repositories/sgnmt/data/iwslt_data/valid.de "
#     "--preprocessing word --postprocessing bpe@@ --decoder batch --outputs text") #

DEFAULT_ARGS = ("--fairseq_path /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.joined-dict.transformer/model.pt"
    " --fairseq_lang_pair en-fr --src_wmap /home/meistecl/Documents/repositories/sgnmt/data//wmt14.en-fr.joined-dict.transformer//wmap.bpe.en"
    " --trg_wmap /home/meistecl/Documents/repositories/sgnmt/data/wmt14.en-fr.joined-dict.transformer/wmap.bpe.fr --input_method file"
    " --src_test /home/meistecl/Documents/repositories/sgnmt/data/newstest/newtest.bpe.en"
    " --preprocessing word --postprocessing bpe@@ --outputs text --decoder batch")

NUM_GENERATE = 1000

def objective(config, reporter):
    

    # lv = str(round(config["local_variance"],2))
    # max1 = str(round(config["max"],2))
    # var = str(round(config["variance"],2))
    # greedy = str(round(config["greedy"],2))
    guido, lmbda = tuple(map(str, config["guido"].split(',')))
    beam = str(config["beam"])

    filename = "/home/meistecl/Documents/repositories/sgnmt/wmt_test_" +guido + "_b" + beam + "_"+ lmbda +".txt" #+ greedy +"_" + lv + "_" + max1 + "_" + var + ".txt"
    out = []
    def run(start=0):
        range_str = str(start+1) + ":" + str(NUM_GENERATE)
        parser = get_parser()
        input_args = DEFAULT_ARGS.split(' ')
        input_args.extend(["--output_path", filename])
        input_args.extend(["--guido", guido]) #"greedy,local_variance, max, variance"
        input_args.extend(["--guido_lambdas", lmbda]) # ','.join([greedy, lv, max1 , var])
        input_args.extend(["--range", range_str ])
        input_args.extend(["--beam", beam ])
        
        args = parser.parse_args(input_args)
        decode_utils.base_init(args)

        io.initialize(args)
        decoder = decode_utils.create_decoder()
        outputs = decode_utils.create_output_handlers()

        with open(args.src_test) as f:
            decode_utils.do_decode(decoder, outputs,
                                    [line.strip() for line in f])
        out.extend(detokenize(filename))

    
    if os.path.exists(filename):
        out.extend(detokenize(filename))
        filename += "2"
    if len(out) < NUM_GENERATE:        
        run(len(out))
        # Load configuration from command line arguments or configuration file
    
    score = sacrebleu.corpus_bleu(out[:NUM_GENERATE], [REFS[:NUM_GENERATE]]).score
    
    reporter(timesteps_total=1,
            mean_accuracy=(score))


if __name__ == "__main__":
    ray.init(num_gpus=torch.cuda.device_count())

    space = {"local_variance": (0, 10), "variance": (0, 12), "max": (0,10), "greedy": (0, 10)}
    

    config = {
        "num_samples": 50,
    }

    # algo = BayesOptSearch(
    #     space,
    #     max_concurrent=8,
    #     metric="mean_accuracy",
    #     mode="max",
    #     random_state=2,
    #     utility_kwargs={
    #         "kind": "ei",
    #         "kappa": 2.5,
    #         "xi": 0.0
    #     })

    # scheduler = AsyncHyperBandScheduler(metric="mean_accuracy", mode="max")

    # analysis = run(objective,
    #     name="guidos_all_200",
    #     search_alg=algo,
    #     scheduler=scheduler,
    #     resources_per_trial= {
    #             "cpu": 2,
    #             "gpu": 1},
    #     #raise_on_failed_trial=False,
    #     **config)

    analysis = run(objective,
        config={"beam": grid_search([500, 100, 50, 20, 10,5,3]), #"lambda":  grid_search([0.2,0.5,0.7,1,2,3,5,7,9,11]),
        "guido": grid_search(["max,1", "max,2", "max,3", 
            "local_variance,4","local_variance,5", "local_variance,6",
            "variance,1",  "variance,2"])},
        resources_per_trial= {
                "cpu": 2,
                "gpu": 1},
        )

    print("Best config is", analysis.get_best_config(metric="mean_accuracy"))

