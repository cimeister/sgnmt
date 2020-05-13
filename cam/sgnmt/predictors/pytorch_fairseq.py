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

"""This is the interface to the fairseq library.

https://github.com/pytorch/fairseq

The fairseq predictor can read any model trained with fairseq.
"""

import logging
import os

from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor

from fairseq import checkpoint_utils, options, tasks
from fairseq import utils as fairseq_utils
from fairseq.sequence_generator import EnsembleModel
import torch
import numpy as np
import copy



FAIRSEQ_INITIALIZED = False
"""Set to true by _initialize_fairseq() after first constructor call."""


def _initialize_fairseq(user_dir):
    global FAIRSEQ_INITIALIZED
    if not FAIRSEQ_INITIALIZED:
        logging.info("Setting up fairseq library...")
        if user_dir:
            args = type("", (), {"user_dir": user_dir})()
            fairseq_utils.import_user_module(args)
        FAIRSEQ_INITIALIZED = True

def get_fairseq_args(model_path, lang_pair):
    parser = options.get_generation_parser()
    input_args = ["--path", model_path, os.path.dirname(model_path)]
    if lang_pair:
        src, trg = lang_pair.split("-")
        input_args.extend(["--source-lang", src, "--target-lang", trg])
    return options.parse_args_and_arch(parser, input_args)


class FairseqPredictor(Predictor):
    """Predictor for using fairseq models."""

    def __init__(self, model_path, user_dir, lang_pair, n_cpu_threads=-1, 
        subtract_uni=False, subtract_marg=False, marg_path=None, lmbda=1.0, 
        ppmi=False, epsilon=0, temperature=1.
        ):
        """Initializes a fairseq predictor.

        Args:
            model_path (string): Path to the fairseq model (*.pt). Like
                                 --path in fairseq-interactive.
            lang_pair (string): Language pair string (e.g. 'en-fr').
            user_dir (string): Path to fairseq user directory.
            n_cpu_threads (int): Number of CPU threads. If negative,
                                 use GPU.
        """
        super(FairseqPredictor, self).__init__()
        _initialize_fairseq(user_dir)
        self.use_cuda = torch.cuda.is_available() and n_cpu_threads < 0

        args = get_fairseq_args(model_path, lang_pair)

        # Setup task, e.g., translation
        task = tasks.setup_task(args)
        source_dict = task.source_dictionary
        target_dict = task.target_dictionary
        self.src_vocab_size = len(source_dict) + 1
        self.trg_vocab_size = len(target_dict) + 1
        self.pad_id = target_dict.pad()
        self.eos_id = target_dict.eos()
        self.bos_id = target_dict.bos()
         # Load ensemble
        self.models = self.load_models(model_path, task)
        self.model = EnsembleModel(self.models)
        self.model.eval()

        self.temperature = temperature


    def load_models(self, model_path, task):
        logging.info('Loading fairseq model(s) from {}'.format(model_path))
        models, _ = checkpoint_utils.load_model_ensemble(
            model_path.split(':'),
            task=task,
        )

        # Optimize ensemble for generation
        for model in models:
            model.make_generation_fast_(
                beamable_mm_beam_size=1,
                need_attn=False,
            )
            if self.use_cuda:
                model.cuda()
        return models

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
                
    def predict_next(self):
        """Call the fairseq model."""
        inputs = torch.LongTensor([self.consumed])
        
        if self.use_cuda:
            inputs = inputs.cuda()

        lprobs, _  = self.model.forward_decoder(
            inputs, self.encoder_outs, temperature=self.temperature
        )
        lprobs[:, self.pad_id] = utils.NEG_INF
        return np.array(lprobs[0].cpu() if self.use_cuda else lprobs[0])

    def predict_next_batch(self, hypos):
        """Call the fairseq model."""
        
        inputs = torch.LongTensor(self.consumed)
        if self.use_cuda:
            inputs = inputs.cuda()
        self.encoder_outs = self.model.reorder_encoder_out(self.encoder_outs, 
            torch.zeros(inputs.size(0), device=inputs.device, dtype=torch.long))

        lprobs, _  = self.model.forward_decoder(
            inputs, self.encoder_outs, temperature=self.temperature
        )
        lprobs[:, self.pad_id] = utils.NEG_INF
        if self.use_cuda:
            lprobs = lprobs.cpu()
        lprobs_per_hypo = []
        for i, hypo in enumerate(hypos):
            lprobs_per_hypo.append(np.array(lprobs[i]))
        return lprobs_per_hypo
    
    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""
        self.consumed = []
        src_tokens = torch.LongTensor([
            utils.oov_to_unk(src_sentence + [utils.EOS_ID],
                             self.src_vocab_size)])
        src_lengths = torch.LongTensor([len(src_sentence) + 1])
        if self.use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
        self.encoder_outs = self.model.forward_encoder({
            'src_tokens': src_tokens,
            'src_lengths': src_lengths})
        self.consumed = [utils.GO_ID or utils.EOS_ID]

        # Reset incremental states

        for model in self.models:
            self.model.incremental_states[model] = {}
   
    def consume(self, word, i=None):
        """Append ``word`` to the current history."""
        self.consumed.append(word) if i is None else self.consumed[i].append(word)
    
    def get_empty_str_prob(self):
        inputs = torch.LongTensor([[utils.GO_ID or utils.EOS_ID]])
        if self.use_cuda:
            inputs = inputs.cuda()
        
        lprobs, _ = self.model.forward_decoder(
            inputs, self.encoder_outs
        )
        return lprobs[0,self.eos_id].item()


    def get_states(self):
        ret = []
        for i in range(len(self.consumed)):
            ind = torch.LongTensor([i])
            if self.use_cuda:
                ind = ind.cuda()
            
            ret.append((self.delete_padding(self.consumed[i]), [self.separate_incremental_state(model, ind)\
                for model in self.models]))

        return ret

    def delete_padding(self, seq):
        try:
            del seq[seq.index(self.pad_id):]
        except ValueError:
            pass
        return seq

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [self.model.incremental_states[m] 
                               for m in self.models]
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        consumed, inc_states = state
        self.consumed = copy.copy(consumed)
        for model, inc_state in zip(self.models, inc_states):
            self.model.incremental_states[model] = inc_state

    def coalesce_and_set_states(self, states):
        self.consumed = []
        all_inc_states = []
        for state in states:
            consumed, inc_states = state
            self.consumed.append(copy.copy(consumed) )
            all_inc_states.append(inc_states)
        
        all_inc_states = self._coalesce(len(states), all_inc_states)
        assert len(all_inc_states) == len(self.models)
        for model, inc_state in zip(self.models, all_inc_states):
            self.model.incremental_states[model] = inc_state
           

    def _coalesce(self, bs, all_inc_states):
        assert len(all_inc_states) == bs
        ret = []
        for i in range(len(self.models)):
            coalesced_inc_states = {}
            if all_inc_states[0][i]:
                for key, state in all_inc_states[0][i].items():
                    states = {k: torch.cat([s[i][key][k] for s in all_inc_states], 0)\
                        if state[k] is not None else None for k in state.keys()}
                    coalesced_inc_states[key] = states
            ret.append(coalesced_inc_states)
        return ret

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]

    def separate_incremental_state(
        self, model, ind,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        ret_dict = {}
        incremental_states = self.model.incremental_states[model]
        if incremental_states is None:
            return None
        for key, input_buffer in incremental_states.items():
            states = {}
            if input_buffer is not None:
                for k in input_buffer.keys():
                    input_buffer_k = input_buffer[k]
                    if input_buffer_k is not None:
                        states[k] = input_buffer_k.index_select(0, ind)
                ret_dict[key] = states
        return ret_dict


    