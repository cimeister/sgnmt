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

"""Contains all the basic interfaces and abstract classes for decoders.
The ``Decoder`` class provides common functionality for all decoders.
The ``Hypothesis`` class represents complete hypotheses, which are 
returned by decoders. ``PartialHypothesis`` is a helper class which can
be used by predictors to represent translation prefixes.
"""

from abc import abstractmethod
import copy
import math
from runstats import Statistics

from cam.sgnmt import utils
from cam.sgnmt.decoding.MinMaxHeap import MinMaxHeap
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor
from cam.sgnmt.decoding.interpolation import FixedInterpolationStrategy, \
                                             EntropyInterpolationStrategy, \
                                             MoEInterpolationStrategy
from cam.sgnmt.utils import Observable, Observer, MESSAGE_TYPE_DEFAULT, \
    MESSAGE_TYPE_POSTERIOR, MESSAGE_TYPE_FULL_HYPO, NEG_INF, EPS_P
import numpy as np
from operator import mul
import logging
from functools import reduce


class Hypothesis:
    """Complete translation hypotheses are represented by an instance
    of this class. We store the produced sentence, the combined score,
    and a score breakdown to the separate predictor scores.
    """
    
    def __init__(self, trgt_sentence, total_score, score_breakdown = [], base_score=0):
        """Creates a new full hypothesis.
        
        Args:
            trgt_sentence (list): List of target word ids without <S> 
                                  or </S> which make up the target 
                                  sentence
            total_score (float): combined total score of this hypo
            score_breakdown (list): Predictor score breakdown for each
                                    target token in ``trgt_sentence``
        """
        self.trgt_sentence = trgt_sentence
        self.total_score = total_score
        self.score_breakdown = score_breakdown
        if base_score:
            self.base_score = base_score

    def __repr__(self):
        """Returns a string representation of this hypothesis."""
        return "%s (%f)" % (' '.join(str(w) for w in self.trgt_sentence),
                            self.total_score)
    def __len__(self):
        return len(self.trgt_sentence)

    def __lt__(self, other):
        return self.total_score < other.total_score
    


class PartialHypothesis(object):
    """Represents a partial hypothesis in various decoders. """
    
    def __init__(self, initial_states = None):
        """Creates a new partial hypothesis with zero score and empty
        translation prefix.
        
        Args:
            initial_states: Initial predictor states
        """
        self.predictor_states = initial_states
        self.trgt_sentence = []
        self.score, self.base_score = 0.0, 0.0
        self.score_breakdown = []
        self.statistics = Statistics()
        self.word_to_consume = None
        self.max_score = 0.0


    def __lt__(self, other):
        return len(self.trgt_sentence) < len(other.trgt_sentence)

    def __len__(self):
        return len(self.trgt_sentence)
    
    def get_last_word(self):
        """Get the last word in the translation prefix. """
        if not self.trgt_sentence:
            return None
        return self.trgt_sentence[-1]
        
    def cur_length(self):
        return len(self.trgt_sentence)
    
    def generate_full_hypothesis(self):
        """Create a ``Hypothesis`` instance from this hypothesis. """
        return Hypothesis(self.trgt_sentence, self.score, self.score_breakdown, self.base_score)
    
    def _new_partial_hypo(self, states, word, score, score_breakdown, max_score=None, base_score=0.0):
        """Create a new partial hypothesis, setting its state, score
        translation prefix and score breakdown.
        Args:
            states (object): Predictor states for new hypo. May be state 
                             after consuming word or current state, depending
                             whether full or cheap expansion is used
            word (int): New word to add to prefix
            score (float): Word log probability to be added to score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        new_hypo = PartialHypothesis(states)
        new_hypo.score = self.score + score if not base_score else score
        new_hypo.base_score = base_score
        new_hypo.score_breakdown = copy.copy(self.score_breakdown)
        new_hypo.score_breakdown.append(score_breakdown)
        new_hypo.statistics = self.statistics.copy()
        new_hypo.statistics.push(score_breakdown[0][0], cur_max=max_score)
        new_hypo.trgt_sentence = self.trgt_sentence + [word]
        
        return new_hypo

    def expand(self, word, new_states, score, score_breakdown):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability and updates the
        stored predictor states.
        
        Args:
            word (int): New word to add to the translation prefix
            new_states (object): Predictor states after consuming
                                 ``word``
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        return self._new_partial_hypo(new_states, word, score, score_breakdown)
    
    def cheap_expand(self, word, score, score_breakdown, max_score=None, base_score=0.0):
        """Creates a new partial hypothesis adding a new word to the
        translation prefix with given probability. Does NOT update the
        predictor states but adds a flag which signals that the last 
        word in this hypothesis has not been consumed yet by the 
        predictors. This can save memory because we can reuse the 
        current state for many hypothesis. It also saves computation
        as we do not consume words which are then discarded anyway by
        the search procedure.
        
        Args:
            word (int): New word to add to the translation prefix
            score (float): Word log probability which is to be added
                           to the total hypothesis score
            score_breakdown (list): Predictor score breakdown for
                                    the new word
        """
        hypo = self._new_partial_hypo(self.predictor_states,
                                     int(word), float(score), score_breakdown,
                                     max_score=max_score, base_score=base_score)
        hypo.word_to_consume = int(word)
        return hypo

    # GUIDOs
    def get_score_variance(self, val=None):
        if len(self.score_breakdown) < 2:
            return 1
        if val is not None:
            return self.statistics.pos_variance(val,ddof=0) 
        return self.statistics.variance(ddof=0)


    def get_score_max(self, val=None):
        if val is not None:
            return -self.statistics.pos_minimum(val) 
        return -self.statistics.minimum()

    def get_score_max2(self, tau):
        if self.get_score_max() <= tau:
            return 0
        return utils.INF

    def get_local_variance(self, val=None):
        if val is not None and len(self.score_breakdown) > 0:
            return self.statistics.pos_local_variance(val,ddof=0) 
        if len(self.score_breakdown) < 2:
            return 1
        return self.statistics.local_variance(ddof=0) 
        
    def get_score_greedy(self, val=None):
        if val is not None:
            val, cur_max = val
            return self.statistics.pos_max_offset(val, cur_max) 
        return self.statistics.max_offset()
        

"""The ``CLOSED_VOCAB_SCORE_NORM_*`` constants define the normalization
behavior for closed vocabulary predictor scores. Closed vocabulary 
predictors (e.g. NMT) have a predefined (and normally very limited) 
vocabulary. In contrast, open vocabulary predictors (see 
``UnboundedPredictor``) are defined over a much larger vocabulary 
(e.g. FST) s.t. it is easier to consider them as having an open 
vocabulary. When combining open and closed vocabulary predictors, we use
the UNK probability of closed vocabulary predictors for words outside 
their vocabulary. The following flags decide (as argument to 
``Decoder``) what to do with the closed vocabulary predictor scores
when combining them with open vocabulary predictors in that way. This
can be changed with the --closed_vocab_norm argument """


CLOSED_VOCAB_SCORE_NORM_NONE = 1
"""None: Do not apply any normalization. """


CLOSED_VOCAB_SCORE_NORM_EXACT = 2
"""Exact: Normalize by 1 plus the number of words outside the 
vocabulary to make it a valid distribution again"""


CLOSED_VOCAB_SCORE_NORM_REDUCED = 3
"""Reduced: Always normalize the closed vocabulary scores to the 
vocabulary which is defined by the open vocabulary predictors at each
time step. """

CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK = 4
"""Rescale UNK: Divide the UNK scores by the number of words outside the 
vocabulary. Results in a valid distribution if predictor scores are
stochastic. """

CLOSED_VOCAB_SCORE_NORM_NON_ZERO = 5
"""Apply no normalization, but ensure posterior contains only tokens with scores
strictly < 0.0. """


class Heuristic(Observer):
    """A ``Heuristic`` instance can be used to estimate the future 
    costs for a given word in a given state. See the ``heuristics``
    module for implementations."""
    
    def __init__(self):
        """Creates a heuristic without predictors. """
        super(Heuristic, self).__init__()
        self.predictors = []

    def set_predictors(self, predictors):
        """Set the predictors used by this heuristic. 
        
        Args:
            predictors (list):  Predictors and their weights to be
                                used with this heuristic. Should be in
                                the same form as ``Decoder.predictors``,
                                i.e. a list of (predictor, weight)
                                tuples
        """
        self.predictors = predictors
    
    def initialize(self, src_sentence):
        """Initialize the heuristic with the given source sentence.
        This is not passed through to the heuristic predictors
        automatically but handles initialization outside the
        predictors.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        pass

    @abstractmethod
    def estimate_future_cost(self, hypo):
        """Estimate the future cost (i.e. negative score) given the 
        states of the predictors set by ``set_predictors`` for a
        partial hypothesis ``hypo``. Note that this function is not 
        supposed to change predictor states. If (e.g. for the greedy 
        heuristic) this is not possible, the predictor states must be
        changed back after execution by the implementing method.
        
        Args:
            hypo (PartialHypo): Hypothesis for which to estimate the
                                future cost
        
        Returns:
            float. The future cost estimate for this heuristic
        """
        raise NotImplementedError
    
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """This is the notification method from the ``Observer``
        super class. We implement it with an empty method here, but
        implementing sub classes can override this method to get
        notifications from the decoder instance about generated
        posterior distributions.
        
        Args:
            message (object): The posterior sent by the decoder
        """
        pass
    
class Decoder(Observable):    
    """A ``Decoder`` instance represents a particular search strategy
    such as A*, beam search, greedy search etc. Decisions are made 
    based on the outputs of one or many predictors, which are 
    maintained by the ``Decoder`` instance.
    
    Decoders are observable. They fire notifications after 
    apply_predictors has been called. All heuristics
    are observing the decoder by default.
    """
    
    def __init__(self, decoder_args):
        """Initializes the decoder instance with no predictors or 
        heuristics.
        """
        super(Decoder, self).__init__()
        self.max_len_factor = decoder_args.max_len_factor
        self.predictors = [] # Tuples (predictor, weight)
        self.heuristics = []
        self.heuristic_predictors = []
        self.predictor_names = []
        self.gumbel = decoder_args.gumbel
        self.allow_unk_in_output = decoder_args.allow_unk_in_output
        self.nbest = 1 # length of n-best list
        self.combi_predictor_method = Decoder.combi_arithmetic_unnormalized
        self.combine_posteriors = self._combine_posteriors_norm_none
        self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NONE

        if decoder_args.closed_vocabulary_normalization == 'exact':
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_EXACT
            self.combine_posteriors = self._combine_posteriors_norm_exact
        elif decoder_args.closed_vocabulary_normalization == 'reduced':
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_REDUCED
            self.combine_posteriors = self._combine_posteriors_norm_reduced
        elif decoder_args.closed_vocabulary_normalization == 'rescale_unk':
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK
            self.combine_posteriors = self._combine_posteriors_norm_rescale_unk
        elif decoder_args.closed_vocabulary_normalization == 'non_zero':
            self.closed_vocab_norm = CLOSED_VOCAB_SCORE_NORM_NON_ZERO
            self.combine_posteriors = self._combine_posteriors_norm_non_zero
        self.current_sen_id = -1
        self.apply_predictors_count = 0
        self.lower_bounds = []
        if decoder_args.score_lower_bounds_file:
            with open(decoder_args.score_lower_bounds_file) as f:
                for line in f:
                    self.lower_bounds.append(float(line.strip()))
        self.interpolation_strategies = []
        self.interpolation_smoothing = decoder_args.interpolation_smoothing
        if decoder_args.interpolation_strategy:
            self.interpolation_mean = decoder_args.interpolation_weights_mean
            pred_strat_names = decoder_args.interpolation_strategy.split(',')
            all_strat_names = set([])
            for s in pred_strat_names:
                all_strat_names |= set(s.split("|"))
            for name in set(all_strat_names):
                pred_indices = [idx for idx, strat in enumerate(pred_strat_names)
                                    if name in strat]
                if name == 'fixed':
                    strat = FixedInterpolationStrategy()
                elif name == 'entropy':
                    strat = EntropyInterpolationStrategy(
                             decoder_args.pred_trg_vocab_size,
                             cross_entropy=False)
                elif name == 'crossentropy':
                    strat = EntropyInterpolationStrategy(
                             decoder_args.pred_trg_vocab_size,
                             cross_entropy=True)
                elif name == 'moe':
                    strat = MoEInterpolationStrategy(len(pred_indices), 
                                                     decoder_args)
                else:
                    logging.error("Unknown interpolation strategy '%s'. "
                                  "Ignoring..." % name)
                    continue
                self.interpolation_strategies.append((strat, pred_indices))
    
    def add_predictor(self, name, predictor, weight=1.0):
        """Adds a predictor to the decoder. This means that this 
        predictor is going to be used to predict the next target word
        (see ``predict_next``)
        
        Args:
            name (string): Predictor name like 'nmt' or 'fst'
            predictor (Predictor): Predictor instance
            weight (float): Predictor weight
        """
        self.predictors.append((predictor, weight))
        self.predictor_names.append(name)
    
    def remove_predictors(self):
        """Removes all predictors of this decoder. """
        self.predictors = []
        self.predictor_names = []
        
    def change_predictor_weights(self, new_weights):
        new_preds_and_weights = []
        for w,  (p, _) in zip(new_weights, self.predictors):
            new_preds_and_weights.append((p, w))
        self.predictors = new_preds_and_weights
        logging.debug('Changed predictor weights: {}'.format(
            [w for (_, w) in self.predictors]))

    def set_heuristic_predictors(self, heuristic_predictors):
        """Define the list of predictors used by heuristics. This needs
        to be called before adding heuristics with ``add_heuristic()``

        Args:
            heuristic_predictors (list):  Predictors and their weights 
                                          to be used with heuristics. 
                                          Should be in the same form 
                                          as ``Decoder.predictors``,
                                          i.e. a list of 
                                          (predictor, weight) tuples
        """
        self.heuristic_predictors = heuristic_predictors
    
    def add_heuristic(self, heuristic):
        """Add a heuristic to the decoder. For future cost estimates,
        the sum of the estimates from all heuristics added so far will
        be used. The predictors used in this heuristic have to be set
        before via ``set_heuristic_predictors()``
        
        Args:
            heuristic (Heuristic): A heuristic to use for future cost
                                   estimates
        """
        heuristic.set_predictors(self.heuristic_predictors)
        self.add_observer(heuristic)
        self.heuristics.append(heuristic)
    
    def estimate_future_cost(self, hypo):
        """Uses all heuristics which have been added with 
        ``add_heuristic`` to estimate the future cost for a given
        partial hypothesis. The estimates are used in heuristic based
        searches like A*. This function returns the future log *cost* 
        (i.e. the lower the better), assuming that the last word in the
        partial hypothesis ``hypo`` is consumed next.
        
        Args:
            hypo (PartialHypothesis): Hypothesis for which to estimate
                                      the future cost given the current
                                      predictor state
        
        Returns
            float. Future cost
        """
        return sum([h.estimate_future_cost(hypo) for h in  self.heuristics])
    
    def has_predictors(self):
        """Returns true if predictors have been added to the decoder. """
        return len(self.predictors) > 0
    
    def consume(self, word, i=None):
        """Calls ``consume()`` on all predictors. """
        for (p, _) in self.predictors:
            p.consume(word) if i is None else p.consume(word, i) # May change predictor state
    
    def _get_non_zero_words(self, bounded_predictors, posteriors):
        """Get the set of words from the predictor posteriors which 
        have non-zero probability. This set of words is then passed
        through to the open vocabulary predictors.

        This method assumes that both arguments are not empty.

        Args:
            bounded_predictors (list): Tuples of (Predictor, weight)
            bounded_posteriors (list): Corresponding posteriors.

        Returns:
            Iterable with all words with non-zero probability.
        """
        restricted, unrestricted = self._split_restricted_posteriors(
            bounded_predictors, posteriors)
        if not restricted: # No restrictions: use union of keys
            key_sets = []
            max_arr_length = 0
            for posterior in unrestricted:
                if isinstance(posterior, dict):
                    key_sets.append(posterior.keys())
                else:
                    max_arr_length = max(max_arr_length, len(posterior))
            if max_arr_length:
                if all(all(el < max_arr_length for el in k) for k in key_sets):
                    return range(max_arr_length)
                key_sets.append(range(max_arr_length))
            if len(key_sets) == 1:
                return key_sets[0]
            return set().union(*key_sets)
        # Calculate the common subset of restricting posteriors
        arr_lengths = []
        dict_words = None
        for posterior in restricted:
            if isinstance(posterior, dict):
                posterior_words = set(posterior.keys())
                if not dict_words:
                    dict_words = posterior_words
                else:
                    dict_words = dict_words & posterior_words
                if not dict_words: 
                    return None
            else: # We record min and max lengths for array posteriors.
                arr_lengths.append(len(posterior))
        if dict_words: # Dictionary restrictions
            if not arr_lengths:
                return dict_words
            min_arr_length = min(arr_lengths)
            return [w for w in dict_words if w < min_arr_length]
        # Array restrictions
        return range(min(arr_lengths))

    def _split_restricted_posteriors(self, predictors, posteriors):
        """Helper method for _get_non_zero_words(). Splits the
        given list of posteriors into unrestricting and restricting
        ones. Restricting posteriors have UNK scores of -inf.
        """
        restricted = []
        unrestricted = []
        for idx, posterior in enumerate(posteriors):
            (p, _) = predictors[idx]
            if p.get_unk_probability(posterior) == NEG_INF:
                restricted.append(posterior)
            else:
                unrestricted.append(posterior)
        return restricted, unrestricted

    def apply_interpolation_strategy(
            self, pred_weights, non_zero_words, posteriors, unk_probs):
        """Applies the interpolation strategies to find the predictor 
        weights for this apply_predictors() call.
    
        Args:
            pred_weights (list): a priori predictor weights
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``

        Returns:
          A list of predictor weights.
        """
        if self.interpolation_strategies:
            predictions = [[] for _ in pred_weights]
            not_fixed_indices = set()
            for strat, pred_indices in self.interpolation_strategies:
                if not strat.is_fixed():
                  not_fixed_indices |= set(pred_indices)
                new_pred_weights = strat.find_weights(
                        [pred_weights[idx] for idx in pred_indices],
                        non_zero_words,
                        [posteriors[idx] for idx in pred_indices],
                        [unk_probs[idx] for idx in pred_indices])
                for idx, weight in zip(pred_indices, new_pred_weights):
                    predictions[idx].append(weight)
            for idx, preds in enumerate(predictions):
                if preds:
                    if self.interpolation_mean == 'arith':
                        pred_weights[idx] = sum(preds) / float(len(preds))
                    else:
                        pred_weights[idx] = reduce(mul, preds, 1)
                    if self.interpolation_mean == 'geo':
                        pred_weights[idx] = pred_weights[idx]**(1.0/len(preds))
            if self.interpolation_mean == 'prob':
                partition = sum(pred_weights[idx] for idx in not_fixed_indices)
                for idx in not_fixed_indices:
                    pred_weights[idx] /= partition
            if self.interpolation_smoothing != 0.0:
                uni = 1.0 / len(not_fixed_indices)
                s = self.interpolation_smoothing
                for idx in not_fixed_indices:
                    pred_weights[idx] = (1.0 - s) * pred_weights[idx] + s * uni
        return pred_weights
    
    
    def apply_predictors(self, top_n=0):
        """Get the distribution over the next word by combining the
        predictor scores.

        Args:
            top_n (int): If positive, return only the best n words.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        self.apply_predictors_count += 1
        bounded_predictors = [el for el in self.predictors 
                        if not isinstance(el[0], UnboundedVocabularyPredictor)]
        # Get bounded posteriors
        bounded_posteriors = [p.predict_next() for (p, _) in bounded_predictors]
        non_zero_words = self._get_non_zero_words(bounded_predictors,
                                                  bounded_posteriors)
        
        if not non_zero_words: # Special case: no word is possible
            non_zero_words = set([utils.EOS_ID])
        # Add unbounded predictors and unk probabilities
        posteriors = []
        unk_probs = []
        pred_weights = []
        bounded_idx = 0
        
        for (p, w) in self.predictors:
            if isinstance(p, UnboundedVocabularyPredictor):
                posterior = p.predict_next(non_zero_words)
            else: # Take it from the bounded_* variables
                posterior = bounded_posteriors[bounded_idx]
                bounded_idx += 1
            posteriors.append(posterior)
            unk_probs.append(p.get_unk_probability(posterior))
            pred_weights.append(w)
        pred_weights = self.apply_interpolation_strategy(
                pred_weights, non_zero_words, posteriors, unk_probs)


        ret = self.combine_posteriors(
            non_zero_words, posteriors, unk_probs, pred_weights, top_n) 
                
        if not self.allow_unk_in_output and utils.UNK_ID in ret[0]:
            del ret[0][utils.UNK_ID]
            del ret[1][utils.UNK_ID]
        if top_n > 0 and len(ret[0]) > top_n:
            top = utils.argmax_n(ret[0], top_n)
            ret = ({w: ret[0][w] for w in top},
                   {w: ret[1][w] for w in top})
        
        self.notify_observers(ret, message_type = MESSAGE_TYPE_POSTERIOR)
        return ret

    def apply_predictors_batched(self, hypos, top_n=0):
        """Get the distribution over the next word by combining the
        predictor scores.

        Args:
            top_n (int): If positive, return only the best n words.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        self.apply_predictors_count += 1
        bounded_predictors = [el for el in self.predictors 
                        if not isinstance(el[0], UnboundedVocabularyPredictor)]
        # Get bounded posteriors
        all_bounded_posteriors = [p.predict_next_batch(hypos) for (p, _) in bounded_predictors]
        all_rets = []
        assert len(hypos) == len(all_bounded_posteriors[0])

        for i in range(len(hypos)):
            bounded_posteriors = [posteriors[i] for posteriors in all_bounded_posteriors]
            non_zero_words = self._get_non_zero_words(bounded_predictors,
                                                      bounded_posteriors)
            
            if not non_zero_words: # Special case: no word is possible
                non_zero_words = set([utils.EOS_ID])
            # Add unbounded predictors and unk probabilities
            posteriors = []
            unk_probs = []
            pred_weights = []
            bounded_idx = 0
            
            for (p, w) in self.predictors:
                if isinstance(p, UnboundedVocabularyPredictor):
                    posterior = p.predict_next(non_zero_words)
                else: # Take it from the bounded_* variables
                    posterior = bounded_posteriors[bounded_idx]
                    bounded_idx += 1
                posteriors.append(posterior)
                unk_probs.append(p.get_unk_probability(posterior))
                pred_weights.append(w)
            pred_weights = self.apply_interpolation_strategy(
                    pred_weights, non_zero_words, posteriors, unk_probs)
            ret = self.combine_posteriors(
                non_zero_words, posteriors, unk_probs, pred_weights, top_n) 
                    
            if not self.allow_unk_in_output and utils.UNK_ID in ret[0]:
                del ret[0][utils.UNK_ID]
                del ret[1][utils.UNK_ID]
            if top_n > 0 and len(ret[0]) > top_n:
                top = utils.argmax_n(ret[0], top_n)
                ret = ({w: ret[0][w] for w in top},
                       {w: ret[1][w] for w in top})
            
            self.notify_observers(ret, message_type = MESSAGE_TYPE_POSTERIOR)
            all_rets.append(ret)
        return all_rets


    def apply_predictors_with_gumbel(self, hypo, top_n=0):
        """Get the distribution over the next word by combining the
        predictor scores.

        Args:
            top_n (int): If positive, return only the best n words.
        
        Returns:
            combined,score_breakdown: Two dicts. ``combined`` maps 
            target word ids to the combined score, ``score_breakdown``
            contains the scores for each predictor separately 
            represented as tuples (unweighted_score, predictor_weight)
        """
        assert len(self.predictors) == 1
        predictor = self.predictors[0][0]
        self.apply_predictors_count += 1
        # Get bounded posteriors
        bounded_posterior = predictor.predict_next() 
        non_zero_words = self._get_non_zero_words(self.predictors,
                                                  [bounded_posterior])

        max_score = utils.max(bounded_posterior)
        vf = np.vectorize(lambda x: self.get_pos_score(hypo, x, max_score) - self.get_adjusted_score(hypo))
        shifted_posterior = vf(bounded_posterior)
        shifted_posterior = np.log(utils.softmax(shifted_posterior))
        # print()
        # print('\n')

        np.random.seed(seed=0)
        gumbels = np.random.gumbel(loc=0, scale=1, size=shifted_posterior.shape)
        gumbel_posterior = shifted_posterior + gumbels + hypo.base_score
        Z = np.max(gumbel_posterior)

        v = hypo.score - gumbel_posterior + utils.logmexp(gumbel_posterior - Z)
        gumbel_full_posterior = hypo.score - np.maximum(0, v) - utils.logpexp(-np.abs(v))

        # make sure invalid tokens still have neg inf log probability
        gumbel_full_posterior[(bounded_posterior == utils.NEG_INF).nonzero()] == utils.NEG_INF
        
        if not non_zero_words: # Special case: no word is possible
            non_zero_words = set([utils.EOS_ID])

        posteriors = [gumbel_full_posterior]
        unk_probs = [predictor.get_unk_probability(gumbel_full_posterior)]
        pred_weights = self.apply_interpolation_strategy(
                [self.predictors[0][1]], non_zero_words, posteriors, unk_probs)
        ids, posteriors, breakdown = self.combine_posteriors(
            non_zero_words, posteriors, unk_probs, pred_weights, top_n, bounded_posterior) 

        shifted_posterior = {id_: utils.common_get(shifted_posterior,
                                       id_, unk_probs[0]) for id_ in ids}
        
        if not self.allow_unk_in_output and utils.UNK_ID in posteriors:
            del posteriors[utils.UNK_ID]
            del breakdown[utils.UNK_ID]

        self.notify_observers((posteriors, breakdown), message_type = MESSAGE_TYPE_POSTERIOR)
        return ids, posteriors, breakdown, shifted_posterior

    
    def _expand_hypo(self, hypo, limit=0):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """

        self.set_predictor_states(copy.copy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        if self.gumbel: 
            ret = self.apply_predictors_with_gumbel(hypo, limit)
            ids, posterior, score_breakdown, shifted_posterior = ret
        else:
            ids, posterior, score_breakdown = self.apply_predictors(limit)

 
        max_score = utils.max(posterior)
        hypo.predictor_states = self.get_predictor_states()
        new_hypos = [hypo.cheap_expand(
                        trgt_word,
                        score,
                        score_breakdown[trgt_word], 
                        max_score=max_score,
                        base_score=hypo.base_score+shifted_posterior[trgt_word] if self.gumbel else 0.0
                        ) for trgt_word, score in zip(ids,posterior)]
    
        return new_hypos


    def get_pos_score(self, hypo, val, max_=None):
        """Combines hypo score with future cost estimates.""" 
        current_score =  hypo.score + val
        if hasattr(self, 'guidos'):
            for w, g in zip(self.guido_lambdas, self.guidos):
                if g == "variance":
                    current_score -= w*hypo.get_score_variance(val)

                elif g == "max":
                    current_score -= w*hypo.get_score_max(val)

                elif g == "local_variance":
                    current_score -= w*hypo.get_local_variance(val)

                elif g == "greedy":
                    current_score -= w*hypo.get_score_greedy((val, max_))

        return current_score

    def get_adjusted_score(self, hypo):
        """Combines hypo score with penalties/rewards.""" 
        current_score =  hypo.score
        if self.gumbel:
            return current_score
        if getattr(self, 'reward_type', False): 
            factor =  min(self.l, len(hypo))
            current_score += self.reward_coef*factor
            if self.heuristics:
                if hypo.get_last_word() != utils.EOS_ID:
                        potential = max(self.l - len(hypo.trgt_sentence),0) 
                        current_score += self.reward_coef*potential
        elif getattr(self, 'heuristics', False):
            if hypo.get_last_word() != utils.EOS_ID:
                remaining = self.max_len - len(hypo.trgt_sentence) 
                current_score += self.lmbda*self.epsilon*remaining

        elif getattr(self, 'length_norm', False): 
            current_score /= len(hypo)

        elif hasattr(self, 'guidos'):
            for w, g in zip(self.guido_lambdas, self.guidos):
                if g == "variance":
                    current_score -= w*hypo.get_score_variance()

                elif g == "max":
                    current_score -= w*hypo.get_score_max()

                elif g == "max2":
                    current_score -= hypo.get_score_max2(w)

                elif g == "local_variance":
                    current_score -= w*hypo.get_local_variance()

                elif g == "greedy":
                    current_score -= w*hypo.get_score_greedy()

        return current_score

    
    def _combine_posteriors_simple(self,
                                      non_zero_words,
                                      posteriors,
                                      unk_probs,
                                      pred_weights,
                                      top_n=0,
                                      original_posterior=None):
        """        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): If positive, return only top n words
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if isinstance(non_zero_words, range) and top_n > 0:
            non_zero_words = Decoder._scale_combine_non_zero_scores(
                len(non_zero_words),
                posteriors,
                unk_probs,
                pred_weights,
                top_n=top_n)

        scores_func = np.vectorize(lambda x: utils.common_get(posteriors[0], x, unk_probs[0]))
        scores = scores_func(non_zero_words)

        if original_posterior is not None:
            scores_func = np.vectorize(lambda x: utils.common_get(original_posterior, x, unk_probs[0]))
            ret = scores_func(non_zero_words)
        else:
            ret = scores

        score_breakdown = {t: [(k, 1.0)] for t,k in zip(non_zero_words, ret)} 
        return non_zero_words, scores, score_breakdown


    def _combine_posteriors_norm_none(self,
                                      non_zero_words,
                                      posteriors,
                                      unk_probs,
                                      pred_weights,
                                      top_n=0,
                                      original_posterior=None):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_NONE``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): If positive, return only top n words
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if pred_weights[0] == 1.0:
            # faster way of doing this... necessary for exact decoding
            return self._combine_posteriors_simple(non_zero_words, 
                posteriors, unk_probs, pred_weights, top_n, original_posterior)
        if isinstance(non_zero_words, range) and top_n > 0:
            non_zero_words = Decoder._scale_combine_non_zero_scores(
                len(non_zero_words),
                posteriors,
                unk_probs,
                pred_weights,
                top_n=top_n)

        combined = {}
        score_breakdown = {}
        for trgt_word in non_zero_words:
            preds = [(utils.common_get(posteriors[idx],
                                       trgt_word, unk_probs[idx]), w)
                        for idx, w in enumerate(pred_weights)]
            combined[trgt_word] = self.combi_predictor_method(preds) 
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown

    def _combine_posteriors_norm_rescale_unk(self,
                                             non_zero_words,
                                             posteriors,
                                             unk_probs,
                                             pred_weights,
                                             top_n=0):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_RESCALE_UNK``. For more 
        information on closed vocabulary predictor score normalization 
        see the documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): If positive, return only top n words
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        unk_counts = [0.0] * n_predictors
        for idx, w in enumerate(pred_weights):
            if unk_probs[idx] >= EPS_P or unk_probs[idx] == NEG_INF:
                continue
            for trgt_word in non_zero_words:
                if not utils.common_contains(posteriors[idx], trgt_word):
                    unk_counts[idx] += 1.0
        return self._combine_posteriors_norm_none(
                          non_zero_words,
                          posteriors,
                          [unk_probs[idx] - np.log(max(1.0, unk_counts[idx]))
                               for idx in range(n_predictors)],
                          top_n)
    
    def _combine_posteriors_norm_exact(self,
                                       non_zero_words,
                                       posteriors,
                                       unk_probs,
                                       pred_weights,
                                       top_n=0):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_EXACT``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): Not implemented!
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        unk_counts = [0] * n_predictors
        for trgt_word in non_zero_words:
            preds = []
            for idx, w in enumerate(pred_weights):
                if utils.common_contains(posteriors[idx], trgt_word):
                    preds.append((posteriors[idx][trgt_word], w))
                else:
                    preds.append((unk_probs[idx], w))
                    unk_counts[idx] += 1
            score_breakdown_raw[trgt_word] = preds
        renorm_factors = [0.0] * n_predictors
        for idx in range(n_predictors):
            if unk_counts[idx] > 1:
                renorm_factors[idx] = np.log(
                            1.0 
                            + (unk_counts[idx] - 1.0) * np.exp(unk_probs[idx]))  
        return self._combine_posteriors_with_renorm(score_breakdown_raw,
                                                    renorm_factors)
    
    def _combine_posteriors_norm_reduced(self,
                                         non_zero_words,
                                         posteriors,
                                         unk_probs,
                                         pred_weights,
                                         top_n=0):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_REDUCED``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): Not implemented!
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        score_breakdown_raw = {}
        for trgt_word in non_zero_words: 
            score_breakdown_raw[trgt_word] = [(utils.common_get(
                                                posteriors[idx],
                                                trgt_word, unk_probs[idx]), w)
                        for idx, w in enumerate(pred_weights)]
        sums = []
        for idx in range(n_predictors):
            sums.append(utils.log_sum([preds[idx][0] 
                            for preds in score_breakdown_raw.values()]))
        return self._combine_posteriors_with_renorm(score_breakdown_raw, sums)
    
    @staticmethod
    def _scale_combine_non_zero_scores(non_zero_word_count,
                                       posteriors,
                                       unk_probs,
                                       pred_weights,
                                       top_n=0):
      scaled_posteriors = []
      for posterior, unk_prob, weight in zip(
              posteriors, unk_probs, pred_weights):
          if isinstance(posterior, dict):
              arr = np.full(non_zero_word_count, unk_prob)
              for word, score in posterior.items():
                  if word < non_zero_word_count:
                      arr[word] = score
              scaled_posteriors.append(arr * weight)
          else:
              n_unks = non_zero_word_count - len(posterior)
              if n_unks > 0:
                  posterior = np.concatenate((
                      posterior, np.full(n_unks, unk_prob)))
              elif n_unks < 0:
                  posterior = posterior[:n_unks]
              scaled_posteriors.append(posterior * weight)
      combined_scores = np.sum(scaled_posteriors, axis=0)
      return utils.argmax_n(combined_scores, top_n)

    def _combine_posteriors_norm_non_zero(self,
                                          non_zero_words,
                                          posteriors,
                                          unk_probs,
                                          pred_weights,
                                          top_n=0):
        """Combine predictor posteriors according the normalization
        scheme ``CLOSED_VOCAB_SCORE_NORM_NON_ZERO``. For more information
        on closed vocabulary predictor score normalization see the 
        documentation on the ``CLOSED_VOCAB_SCORE_NORM_*`` vars.
        
        Args:
            non_zero_words (set): All words with positive probability
            posteriors: Predictor posterior distributions calculated
                        with ``predict_next()``
            unk_probs: UNK probabilities of the predictors, calculated
                       with ``get_unk_probability``
            pred_weights (list): Predictor weights
            top_n (int): If positive, return only top n words

        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        if isinstance(non_zero_words, range) and top_n > 0:
          non_zero_words = Decoder._scale_combine_non_zero_scores(len(non_zero_words), 
                                                                  posteriors,
                                                                  unk_probs,
                                                                  pred_weights,
                                                                  top_n)
        combined = {}
        score_breakdown = {}
        for trgt_word in non_zero_words:
            preds = [(utils.common_get(posteriors[idx],
                                       trgt_word, unk_probs[idx]), w)
                        for idx, w in enumerate(pred_weights)]
            combi_score = self.combi_predictor_method(preds)
            if abs(combi_score) <= EPS_P:
                continue
            combined[trgt_word] = combi_score  
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown

    def _combine_posteriors_with_renorm(self,
                                        score_breakdown_raw,
                                        renorm_factors):
        """Helper function for ``_combine_posteriors_norm_*`` functions
        to renormalize score breakdowns by predictor specific factors.
        
        Returns:
            combined,score_breakdown: like in ``apply_predictors()``
        """
        n_predictors = len(self.predictors)
        combined = {}
        score_breakdown = {}
        for trgt_word,preds_raw in score_breakdown_raw.items():
            preds = [(preds_raw[idx][0] - renorm_factors[idx],
                      preds_raw[idx][1]) for idx in range(n_predictors)]
            combined[trgt_word] = self.combi_predictor_method(preds) 
            score_breakdown[trgt_word] = preds
        return combined, score_breakdown
    
    def set_current_sen_id(self, sen_id):
        self.current_sen_id = sen_id - 1  # -1 because incremented in init()
            
    def initialize_predictors(self, src_sentence):
        """First, increases the sentence id counter and calls
        ``initialize()`` on all predictors. Then, ``initialize()`` is
        called for all heuristics.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        """
        self.max_len = int(np.ceil(self.max_len_factor * len(src_sentence)))
        self.full_hypos = []
        self.current_sen_id += 1
        for idx, (p, _) in enumerate(self.predictors):
            p.set_current_sen_id(self.current_sen_id)
            p.initialize(src_sentence)
        for h in self.heuristics:
            h.initialize(src_sentence)
    
    def add_full_hypo(self, hypo):
        """Adds a new full hypothesis to ``full_hypos``. This can be
        used by implementing subclasses to add a new hypothesis to the
        result set. This method also notifies observers.
        
        Args:
            hypo (Hypothesis): New complete hypothesis
        """
        if len(self.full_hypos) == 0 or hypo.total_score > self.cur_best.total_score:
            self.cur_best = hypo
        self.full_hypos.append(hypo)
        self.notify_observers(hypo, message_type = MESSAGE_TYPE_FULL_HYPO)
    
    def get_full_hypos_sorted(self):
        """Returns ``full_hypos`` sorted by the total score. Can be 
        used by implementing subclasses as return value of
        ``decode``
        
        Returns:
            list. ``full_hypos`` sorted by ``total_score``.
        """
        return sorted(self.full_hypos,
                      key=lambda hypo: hypo.total_score,
                      reverse=True)
    
    def get_lower_score_bound(self):
        """Intended to be called by implementing subclasses. Returns a
        lower bound on the best score of the current sentence. This is
        either read from the lower bounds file (if provided) or set to
        negative infinity.
        
        Returns:
            float. Lower bound on the best score for current sentence
        """ 
        if self.current_sen_id < len(self.lower_bounds):
            return self.lower_bounds[self.current_sen_id] - EPS_P
        return NEG_INF


    def get_empty_hypo(self, src_sentence):
        self.initialize_predictors(src_sentence)
        hypo = PartialHypothesis(self.get_predictor_states())
        ind = utils.EOS_ID
        ids, posterior, score_breakdown = self.apply_predictors()

        max_score = utils.max(posterior)

        hypo.score += posterior[ind] 
        hypo.score_breakdown.append(score_breakdown[ind])
        hypo.statistics.push(posterior[ind], cur_max=max_score)
        hypo.trgt_sentence += [ind]

        hypo.score = self.get_adjusted_score(hypo)
        return hypo

    
    def get_max_expansions(self, max_expansions_param, src_sentence):
        """This is a helper for decoders which support the 
        ``max_node_expansions`` parameter. It returns the maximum
        number of node expansions for the given sentence.
        
        Args:
            max_expansions_param (int): max_node_expansions parameter
                                        passed through from the config
            src_sentence (list): Current source sentence
        
        Returns:
            int. Maximum number of node expansions for this decoding
            task.
        """
        if max_expansions_param > 0:
            return max_expansions_param
        if max_expansions_param < 0:
            return -len(src_sentence) * max_expansions_param
        return 100000000  
    
    def set_predictor_states(self, states):
        """Calls ``set_state()`` on all predictors. """
        i = 0
        for (p, _) in self.predictors:
            p.set_state(states[i])
            i = i + 1
    
    def get_predictor_states(self, batch=False):
        """Calls ``get_state()`` on all predictors. """
        return [p.get_state() if not batch else p.get_states() for (p, _) in self.predictors]
    
    @staticmethod
    def combi_arithmetic_unnormalized(x):
        """Calculates the weighted sum (or geometric mean of log 
        values). Do not use with empty lists.
        
        Args:
            x (list): List of tuples [(out1, weight1), ...]
        
        Returns:
            float. Weighted sum out1*weight1+out2*weight2...
        """
        #return sum(f*w for f, w in x)
        (fAcc, _) = reduce(lambda x1, x2: (x1[0]*x1[1] + x2[0]*x2[1], 1.0),
                           x,
                           (0.0, 1.0))
        return fAcc
    
    @abstractmethod
    def decode(self, src_sentence):
        """Decodes a single source sentence. This method has to be 
        implemented by subclasses. It contains the core of the 
        implemented search strategy ``src_sentence`` is a list of
        source word ids representing the source sentence without
        <S> or </S> symbols. This method returns a list of hypotheses,
        order descending by score such that the first entry is the best
        decoding result. Implementations should delegate the scoring of
        hypotheses to the predictors via ``apply_predictors()``, and
        organize predictor states with the methods ``consume()``,
        ``get_predictor_states()`` and ``set_predictor_states()``. In
        this way, the decoder is decoupled from the scoring modules.
        
        Args:
            src_sentence (list): List of source word ids without <S> or
                                 </S> which make up the source sentence
        
        Returns:
            list. A list of ``Hypothesis`` instances ordered by their
            score.
        
        Raises:
            ``NotImplementedError``: if the method is not implemented
        """
        raise NotImplementedError

    def are_equal_predictor_states(self, states1, states2):
        """This method applies ``is_equal`` on all predictors. It 
        returns true if all predictor states are equal.
        
        Args:
            states1 (list): First predictor states as returned by
                            ``get_predictor_states``
            states2 (list): Second predictor states as returned by
                            ``get_predictor_states``
        
        Returns:
            boolean. True if all predictor states are equal, False
            otherwise 
        """
        i = 0
        for (p, _) in self.predictors:
            if not p.is_equal(states1[i], states2[i]):
                return False
            i = i + 1
        return True
    
