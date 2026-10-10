"""Project-wide configurable parallel runtime; no serial-training mandate."""
import os
from pathlib import Path
import yaml


def settings():
    result = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/runtime.yaml').read_text(encoding='utf-8'))
    if any(not isinstance(value, int) or value < 1 for value in result.values()):
        raise ValueError('Runtime thread/process counts must be positive integers')
    return result


def training_threads():
    return settings()['training_threads']


def numerical_threads():
    return settings()['numerical_threads']


def required_environment():
    count = str(numerical_threads())
    return {'PYTHONHASHSEED': '0', 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': count,
            'OPENBLAS_NUM_THREADS': count, 'MKL_NUM_THREADS': count, 'NUMEXPR_NUM_THREADS': count}


def worker_environment():
    result = os.environ.copy()
    result.update(required_environment())
    return result
