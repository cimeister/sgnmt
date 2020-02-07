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
from heapq import heappush, heappop
import heapq
import logging
import numpy as np
from sortedcontainers import SortedDict
from collections import defaultdict
import time
import math
from cam.sgnmt.decoding.MinMaxHeap import MinMaxHeap

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
import os
import gc
import ctypes

class PyObject(ctypes.Structure):
    _fields_ = [("refcnt", ctypes.c_long)]


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
        self.not_monotonic = self.reward_type or decoder_args.ppmi

    def decode(self, src_sentence):
        self.initialize_predictors(src_sentence)
        self.initialize_order_ds() 
        self.active = self.beam
        self.count = 0
        
        self.time1 = 0
        self.time2 = 0
        self.time3 = 0
        
        self.reward_bound(src_sentence)
        while self.queue_order:
            c,t = self.get_next()
            cur_queue = self.queues[t]
            score, hypo = cur_queue.popmin() 
            assert -c == score
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

            for x,i in enumerate(self.score_by_t):
                if self.queues[x]:
                    assert i == -self.queues[x].peekmin()[0]
                else:
                    assert all([j[1] != x for j in self.queue_order.items()])

        
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
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
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
        self.queue_order.pop(self.score_by_t[t], default=None)
        if len(queue) > 0:
            self.queue_order[-queue.peekmin()[0]] = t
            self.score_by_t[t] = -queue.peekmin()[0]
        elif forward_prune:
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
        if self.time_sync[t] <= 0:
            self.prune(t)
    
        self.time2 += time.time() - ti

    def prune(self, t):
        for i in range(t+1):
            if not self.queue_order.pop(self.score_by_t[i], default=None):
                #assert len(self.queues[i]) == 0
                #continue
                pass
            self.queues[i] = []

    
    def add_hypo(self, hypo, queue, t):
        score = self.get_adjusted_score(hypo)
        ti = time.time()
        if len(queue) < self.time_sync[t]:
            queue.insert((-score, hypo))
        else:
            max_val = queue.peekmax()[0]
            if score > -max_val:
                queue.popmax() #could make this faster by just replacing the max val and sifting up
                queue.insert((-score, hypo))

        self.time1 += time.time() - ti
        

    def stop(self):
        if self.not_monotonic:
            if not self.early_stopping and len(self.full_hypos) < self.beam:
                return False
            threshold = min(self.full_hypos) if self.early_stopping else max(self.full_hypos)
            if all([threshold.total_score > self.max_pos_score(q.peekmin()[1]) for q in self.queues if q]):
                return True
        elif self.early_stopping:
            return True 
        elif len(self.full_hypos) == self.beam:
            return True
        return False

    def reward_bound(self, src_sentence):
        if self.reward_type == "bounded":
            self.l = len(src_sentence)
        elif self.reward_type == "max":
            self.l = self.max_len

    def get_adjusted_score(self, hypo):
        """Combines hypo score with future cost estimates.""" 
        current_score =  hypo.score
        if self.reward_type:
            factor =  min(self.l, len(hypo))
            current_score += self.reward_coef*factor
        return current_score 

    def max_pos_score(self, hypo):
        current_score = hypo.score
        if self.lmbda:
            max_increase = self.lmbda*self.epsilon*(self.max_len - len(hypo))\
                if hypo.get_last_word() != utils.EOS_ID else 0
            current_score += max_increase
        elif self.reward_type:
            factor =  min(0,self.l - len(hypo)) #if hypo.get_last_word() == utils.EOS_ID else self.l
            current_score += self.reward_coef*factor
        return current_score
        
