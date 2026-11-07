"""Opt-in frequency-side sampling, with checkpoint-bound private RNG and source IDs."""
import hashlib,json
from pathlib import Path
import torch
from marshawn_frequency_sampling_proposal import FrequencyProposal

class FocusRuntime:
    def __init__(self, pipe, main_adapter, configuration):
        expected=dict(budget=2048,core_count=1229,band_count=410,uniform_count=409,private_seed=4201024)
        if configuration!=expected:raise ValueError('Exact focused2048 proposal required')
        if pipe.config.grid_update_batch_size!=2048 or pipe.config.grid_update_interval!=1024 or pipe.config.frequency_projection_mode!='legacy' or pipe.config.independent_rng_streams or pipe.config.datamanager.cpu_fas_prefetch:raise ValueError('Frequency/main RNG policy differs')
        if hasattr(pipe,'dec5_frequency_sampler'):raise ValueError('Frequency hook already owned')
        self.configuration=dict(configuration);self.pipe=pipe;self.last_step=None
        self.source_sha256=hashlib.sha256(json.dumps(main_adapter.source_records,sort_keys=True).encode()).hexdigest()
        self.proposal=FrequencyProposal([r['core'].shape for r in main_adapter.rows],[r['pools']for r in main_adapter.rows],seed=configuration['private_seed'])
        self.original=pipe._update_frequency_grid;pipe.dec5_frequency_sampler=self.proposal.hook;pipe._update_frequency_grid=self.update
    def update(self,step):
        if not self.pipe.cached_freq_maps:return
        self.proposal.prepare(list(self.pipe.cached_freq_maps),self.configuration['budget'],torch.get_rng_state())
        self.original(step)
        if self.proposal.pending is not None:raise ValueError('Actual frequency update did not consume proposal')
        self.last_step=int(step)
    def state_dict(self):
        return dict(schema=1,configuration=self.configuration,source_sha256=self.source_sha256,last_step=self.last_step,proposal=self.proposal.state_dict())
    def load_state_dict(self,state):
        if state.get('schema')!=1 or state['configuration']!=self.configuration or state['source_sha256']!=self.source_sha256:raise ValueError('Private frequency source/policy mismatch')
        self.proposal.load_state_dict(state['proposal']);self.last_step=state['last_step']
    def digest(self):return hashlib.sha256(json.dumps(self.state_dict(),sort_keys=True).encode()).hexdigest()
