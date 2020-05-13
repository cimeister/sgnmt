# -*- coding: utf-8 -*-
# coding=utf-8
# Copyright 2019 The SGNMT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the A* search strategy """


import time
import math
import copy
import numpy as np

from cam.sgnmt.decoding.MinMaxHeap import MinMaxHeap
from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class BatchDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): beam width.
            early_stopping (bool): If this is true, partial hypotheses
                                   with score worse than the current
                                   best complete scores are not
                                   expanded. This applies when nbest is
                                   larger than one and inadmissible
                                   heuristics are used
            nbest (int): If this is set to a positive value, we do not
                         stop decoding at the first complete path, but
                         continue search until we collected this many
                         complete hypothesis. With an admissible
                         heuristic, this will yield an exact n-best
                         list.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BatchDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.beam = decoder_args.beam
        self.early_stopping = decoder_args.early_stopping
        self.reward_coef = decoder_args.reward_coefficient
        self.reward_type = decoder_args.reward_type
        assert not (decoder_args.subtract_uni or decoder_args.subtract_marg) or decoder_args.ppmi

        self.epsilon = decoder_args.epsilon
        self.lmbda = decoder_args.lmbda
        self.not_monotonic = (self.reward_type or decoder_args.ppmi) and not decoder_args.heuristics
        self.use_heuristics = decoder_args.heuristics
        self.size_threshold = self.beam*decoder_args.memory_threshold_coef\
            if decoder_args.memory_threshold_coef > 0 else utils.INF

        self.length_norm = False

        self.guidos = utils.split_comma(decoder_args.guido)
        self.guido_lambdas = utils.split_comma(decoder_args.guido_lambdas, func=float)
        if any(g in ['variance', 'local_variance'] for g in self.guidos) or self.length_norm:
            self.not_monotonic = True

        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        self.initialize_ds() 
        self.total_queue_size = 0

        if self.length_norm:
            print("WARNING, USING LENGTH NORM")

        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        
        self.reward_bound(src_sentence)
        t = 0
        while True:
            t+=1
            if self.stopping_criterion():
                break

            batched_hypos = [] 
            next_hypos = MinMaxHeap(reserve=self.beam)
            for score, hypo in self.queue:

                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.insert((-score, hypo))
                    continue

                if t == self.max_len + 1:
                    next_hypos.insert((-score, hypo))
                    continue

                batched_hypos.append(hypo)

            if len(batched_hypos) == 0:
                self.queue = next_hypos
                break
            self.queue = self._expand_hypo_batch(batched_hypos, next_hypos, self.beam)
                
        for _, hypo in self.queue:
            if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis())
        
        return self.get_full_hypos_sorted(), (self.time1, self.time2, self.time3)


    def initialize_ds(self):
        self.queue = MinMaxHeap() 
        self.queue.insert((0.0, PartialHypothesis(self.get_predictor_states())))
   

    def stopping_criterion(self):
        if not self.early_stopping:

            if len(self.full_hypos) < self.nbest:
                return False
            hypo = self.queue.peekmin()[1]
            if hypo.get_last_word() == utils.EOS_ID:
                if self.not_monotonic and any([self.get_adjusted_score(hypo) > self.max_pos_score(h) for h in self.queue]):
                    return False
                return True 

        return False

    def reward_bound(self, src_sentence):
        if self.reward_type == "bounded":
            # french is 0.72
            self.l = len(src_sentence)
        elif self.reward_type == "max":
            self.l = self.max_len

    def max_pos_score(self, hypo):
        current_score = hypo.score
        if self.length_norm:
            current_score /= self.max_len
        for w,g in zip(self.guido_lambdas, self.guidos):
            if g == "variance":
                var = hypo.get_score_variance()
                lowest_var = var*len(hypo)/self.max_len
                current_score -= w*lowest_var
            elif g == "local_variance":
                var = hypo.get_local_variance()
                lowest_var = var*len(hypo)/self.max_len
                current_score -= w*lowest_var
        if self.lmbda:
            max_increase = self.lmbda*self.epsilon*(self.max_len - len(hypo))\
                if hypo.get_last_word() != utils.EOS_ID else 0
            current_score += max_increase
        elif self.reward_type:
            factor =  self.l if hypo.get_last_word() != utils.EOS_ID else 0
            current_score += self.reward_coef*factor
        return current_score
    
    def _coalesce_states(self, states):
        new_states = []

        for i, (p, _) in enumerate(self.predictors):
            p.coalesce_and_set_states([s[i] for s in states])

    def _expand_hypo_batch(self, hypos, next_hypos=None, limit=0):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """
       
        comp_func = self.max_pos_score if self.not_monotonic else self.get_adjusted_score
        all_new_hypos = next_hypos if next_hypos else MinMaxHeap(reserve=limit)
        max_batch_size = 500
        num_batches = int(math.ceil(len(hypos)/max_batch_size))  
        for i in range(num_batches):
            cur_batch = hypos[max_batch_size*i:max_batch_size*(i+1)]
            states = [copy.copy(hypo.predictor_states) for hypo in cur_batch]
            self._coalesce_states(states)
            for i, hypo in enumerate(cur_batch):
                if not hypo.word_to_consume is None: # Consume if cheap expand
                    self.consume(hypo.word_to_consume, i=i)
                    hypo.word_to_consume = None

            posteriors = self.apply_predictors_batched(cur_batch, limit)
            states = self.get_predictor_states(batch=True)
            
            
            for j, hypo in enumerate(cur_batch):
                words, posterior, score_breakdown = posteriors[j]
                max_score = utils.max(posterior)
                if len(all_new_hypos) >= limit and self.get_adjusted_score(hypo) + max_score < -all_new_hypos.peekmax()[0]:
                    continue
                hypo.predictor_states = [model_states[j] for model_states in states]
                vf = np.vectorize(lambda x: self.get_pos_score(hypo, x, max_score))
                scores = vf(posterior)
                for k, (trgt_word, pos_score) in enumerate(zip(words, scores)):
                    #pos_score = self.get_pos_score(hypo, posterior[trgt_word])
                    if len(all_new_hypos) < limit or pos_score > -all_new_hypos.peekmax()[0]:
                        
                        new_hypo = hypo.cheap_expand(
                                    trgt_word,
                                    posterior[k],
                                    score_breakdown[trgt_word], 
                                    # base_score = og_posterior[trgt_word] if self.gumbel else 0,
                                    max_score=max_score)
                        if len(all_new_hypos) < limit:
                            all_new_hypos.insert((-self.get_adjusted_score(new_hypo), new_hypo))
                        else:
                            all_new_hypos.replacemax((-self.get_adjusted_score(new_hypo), new_hypo))
                    
        
        return all_new_hypos
