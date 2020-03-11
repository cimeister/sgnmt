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


import copy
import logging
import numpy as np
from sortedcontainers import SortedDict
from collections import defaultdict
import time
import math
from cam.sgnmt.decoding.MinMaxHeap import MinMaxHeap

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class DijkstraTSDecoder(Decoder):
    
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
        super(DijkstraTSDecoder, self).__init__(decoder_args)
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
        
    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        self.initialize_order_ds() 
        self.active = self.beam
        self.count = 0
        
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0

        self.size = 0
        
        self.reward_bound(src_sentence)
        while self.queue_order:
            c,t = self.get_next()
            cur_queue = self.queues[t]
            score, hypo = cur_queue.popmin() 
            self.size -= 1
            self.time_sync[t] -= 1

            if hypo.get_last_word() == utils.EOS_ID:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if self.stop(): # if stopping criterion are met
                    break
                self.update(cur_queue, t, forward_prune=True)
                continue

            if t == self.max_len:
                self.update(cur_queue, t)
                continue

            next_queue = self.queues[t+1]
            for next_hypo in self._expand_hypo(hypo):
                self.add_hypo(next_hypo, next_queue, t+1)
                    
            self.update(cur_queue, t)
            self.update(next_queue, t+1)
        
        print(self.time1, self.time2, self.time3)
        print("Count", self.count)
        return self.get_full_hypos_sorted(), self.count

    def _expand_hypo(self, hypo):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """
        t = time.time()
        self.set_predictor_states(copy.copy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        posterior, score_breakdown = self.apply_predictors(self.beam)
        self.count += 1
        hypo.predictor_states = self.get_predictor_states()
        new_hypos = [hypo.cheap_expand(
                        trgt_word,
                        posterior[trgt_word],
                        score_breakdown[trgt_word]) for trgt_word in posterior]
    
        self.time3 += time.time() - t
        return new_hypos

    def initialize_order_ds(self):
        self.queues = [MinMaxHeap() for k in range(self.max_len+1)]
        self.queues[0].insert((0.0, PartialHypothesis(self.get_predictor_states())))
        self.queue_order = SortedDict({0.0: 0})
        self.score_by_t = [0.0]
        self.score_by_t.extend([None]*self.max_len)
        self.time_sync = defaultdict(lambda: self.beam)
        self.time_sync[0] = 1

    def get_next(self):
        return self.queue_order.popitem()
        
    def update(self, queue, t, forward_prune=False):
        ti = time.time()
        # remove current best value associated with queue
        self.queue_order.pop(self.score_by_t[t], default=None)

        # if beam used up at current time step, can prunehypotheses from older time steps
        if self.time_sync[t] <= 0:
            self.prune(t)

        #replace with next best value if anything left in queue
        elif len(queue) > 0:
            self.queue_order[-queue.peekmin()[0]] = t
            self.score_by_t[t] = -queue.peekmin()[0]

        # if previous hypothesis was complete, reduce beam in next time steps
        if forward_prune:
            i = self.max_len
            while i > t:
                self.time_sync[i] -= 1
                if self.time_sync[i] <= 0:
                    self.prune(i)
                    return
                while len(self.queues[i]) > self.time_sync[i]:
                    # remove largest element since beam is getting "smaller"
                    self.queues[i].popmax()
                i-=1
    
        self.time2 += time.time() - ti

    def prune(self, t):
        for i in range(t+1):
            self.queue_order.pop(self.score_by_t[i], default=None)
            self.queues[i] = []

    
    def add_hypo(self, hypo, queue, t):
        score = self.get_adjusted_score(hypo)
        ti = time.time()
        if len(queue) < self.time_sync[t]:
            queue.insert((-score, hypo))
            if self.size >= self.size_threshold:
                self.remove_one()
            else:
                self.size += 1
        else:
            max_val = queue.peekmax()[0]
            if score > -max_val:
                queue.popmax() #could make this faster by just replacing the max val and sifting up
                queue.insert((-score, hypo))

        self.time1 += time.time() - ti
        
    def remove_one(self):
        """ helper function for memory threshold"""
        for t, q in enumerate(self.queues):
            if len(q) > 0:
                q.popmax()
                if len(q) == 0:
                    self.queue_order.pop(self.score_by_t[t], default=None)
                return

    def stop(self):
        if self.not_monotonic:
            if not self.early_stopping and len(self.full_hypos) < self.beam:
                return False
            threshold = max(self.full_hypos) if self.early_stopping else min(self.full_hypos)
            if all([threshold.total_score > self.max_pos_score(q.peekmin()[1]) for q in self.queues if q]):
                return True
        elif self.early_stopping:
            return True 
        elif len(self.full_hypos) == self.beam:
            return True
        return False

    def reward_bound(self, src_sentence):
        if self.reward_type == "bounded":
            # french is 0.72
            self.l = len(src_sentence)
        elif self.reward_type == "max":
            self.l = self.max_len

    def get_adjusted_score(self, hypo):
        """Combines hypo score with future cost estimates.""" 
        current_score =  hypo.score
        if self.reward_type: 
            factor =  min(self.l, len(hypo))
            current_score += self.reward_coef*factor
            print(self.heuristics)
            if self.heuristics:
                if hypo.get_last_word() != utils.EOS_ID:
                        potential = max(self.l - len(hypo.trgt_sentence),0) 
                        current_score += self.reward_coef*potential
        elif self.heuristics:
            if hypo.get_last_word() != utils.EOS_ID:
                remaining = self.max_len - len(hypo.trgt_sentence) 
                current_score += self.lmbda*self.epsilon*remaining

        return current_score 

    def max_pos_score(self, hypo):
        current_score = hypo.score
        if self.lmbda:
            max_increase = self.lmbda*self.epsilon*(self.max_len - len(hypo))\
                if hypo.get_last_word() != utils.EOS_ID else 0
            current_score += max_increase
        elif self.reward_type:
            factor =  self.l if hypo.get_last_word() != utils.EOS_ID else 0
            current_score += self.reward_coef*factor
        return current_score
        
