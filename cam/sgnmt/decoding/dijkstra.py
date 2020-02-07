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
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


class DijkstraDecoder(Decoder):
    
    def __init__(self, decoder_args):
        """Creates a new A* decoder instance. The following values are
        fetched from `decoder_args`:
        
            beam (int): Maximum number of active hypotheses.
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
        super(DijkstraDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.capacity = decoder_args.beam
        self.early_stopping = decoder_args.early_stopping    

    def decode(self, src_sentence):
        """Decodes a single source sentence using A* search. """
        self.initialize_predictors(src_sentence)
        open_set = []
        best_score = self.get_lower_score_bound()
        print("Bound:", best_score)
        heappush(open_set, (0.0,
                            PartialHypothesis(self.get_predictor_states())))
        count = 0
        while open_set:
            c,hypo = heappop(open_set)
            count += 1
            if self.early_stopping and hypo.score < best_score:
                continue
            logging.debug("Expand (est=%f score=%f exp=%d best=%f): sentence: %s"
                          % (-c, 
                             hypo.score, 
                             self.apply_predictors_count, 
                             best_score, 
                             hypo.trgt_sentence))
            if hypo.get_last_word() == utils.EOS_ID: # Found best hypothesis
                if hypo.score > best_score:
                    logging.debug("New best hypo (score=%f exp=%d): %s" % (
                            hypo.score,
                            self.apply_predictors_count,
                            ' '.join([str(w) for w in hypo.trgt_sentence])))
                    best_score = hypo.score
                self.add_full_hypo(hypo.generate_full_hypothesis())
                if len(self.full_hypos) >= self.nbest: # if we have enough hypos
                    return self.get_full_hypos_sorted(), count
                continue
            self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
            if not hypo.word_to_consume is None: # Consume if cheap expand
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
            posterior,score_breakdown = self.apply_predictors()
            hypo.predictor_states = self.get_predictor_states()
            for trgt_word in posterior: # Estimate future cost, add to heap
                next_hypo = hypo.cheap_expand(trgt_word, posterior[trgt_word],
                                                  score_breakdown[trgt_word])
                score = next_hypo.score
                if score > best_score:
                  # only push if hypothesis can beat lower bound. Saves memory...
                  heappush(open_set, (-score, next_hypo))
                  
            # Limit heap capacity
            if self.capacity > 0 and len(open_set) > self.capacity:
                new_open_set = []
                for _ in range(self.capacity):
                    heappush(new_open_set, heappop(open_set))
                open_set = new_open_set
        
        return self.get_full_hypos_sorted(), count
