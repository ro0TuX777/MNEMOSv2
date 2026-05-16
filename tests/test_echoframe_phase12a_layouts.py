import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12a'))
from placement_cases import get_benchmark_cases
from placement_layouts import EchoFramePacket, render_layout

def test_layouts_determinism():
    cases = get_benchmark_cases()
    for case in cases:
        packet = EchoFramePacket.from_dict(case['packet'])
        for layout in ['A', 'B', 'C', 'D', 'E']:
            res1 = render_layout(packet, layout)
            res2 = render_layout(packet, layout)
            assert res1 == res2
            assert len(res1) > 0

def test_preserves_facts():
    cases = get_benchmark_cases()
    case = cases[0]
    packet = EchoFramePacket.from_dict(case['packet'])
    res = render_layout(packet, 'D')
    assert "require manager approval" in res
    assert "$500" in res

def test_preserves_sources():
    cases = get_benchmark_cases()
    case = cases[0]
    packet = EchoFramePacket.from_dict(case['packet'])
    res = render_layout(packet, 'E')
    assert "S1" in res
