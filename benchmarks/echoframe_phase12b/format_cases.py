import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../echoframe_phase12a'))
from placement_cases import get_benchmark_cases

def get_format_cases():
    return get_benchmark_cases()
