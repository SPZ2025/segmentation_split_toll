"""Automatic large-granularity segmentation of continuous input text."""
from .algorithm import InfeasibleSegmentationError
from .backend import ModelUnavailableError
from .segmenter import Fragment, LargeTextSegmenter, SegmentationResult, profile

__version__ = '1.0.0'
__all__ = ['Fragment', 'LargeTextSegmenter', 'SegmentationResult',
           'InfeasibleSegmentationError', 'ModelUnavailableError', 'profile']
