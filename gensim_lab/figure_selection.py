"""All finite comparisons may be plotted; spread never gates presentation."""
from __future__ import annotations

import math



DISPLAY_THRESHOLDS = {
    'correlation': 0.0,
    'classification': 0.0,
    'analogy': 0.0,
    'coverage': 0.0,
}


def display_summary(values, threshold):
    values = [float(v) for v in values]
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('Display comparison needs at least two finite values')
    low, high = min(values), max(values)
    span = high - low
    return {'minimum': low, 'maximum': high, 'spread': span,
            'threshold': float(threshold), 'plot': True,
            'status': 'constant' if span <= 1e-12 else
                      'varying'}

