"""Core speculative sampling implementation."""
from .speculative_sampling import speculative_sampling, speculative_sampling_original
from .autoregressive_sampling import autoregressive_sampling
from .utils import sample_from_draft_model, get_distribution, sample

__all__ = [
    'speculative_sampling',
    'speculative_sampling_original',
    'autoregressive_sampling',
    'sample_from_draft_model',
    'get_distribution',
    'sample'
]
