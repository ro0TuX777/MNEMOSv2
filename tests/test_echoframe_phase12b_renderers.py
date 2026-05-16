import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12b'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../benchmarks/echoframe_phase12a'))
from format_renderers import render_format, assert_format_preserves_packet
from format_cases import get_format_cases
import placement_layouts as layouts

def test_renderers_determinism():
    cases = get_format_cases()
    formats = [
        '1_baseline', '2_layout_d', '3_ultra_compact', '4_yaml_lite', 
        '5_minified_json', '6_markdown_table', '7_toon_rows', '8_source_table_facts'
    ]
    for case in cases:
        packet = layouts.EchoFramePacket.from_dict(case['packet'])
        for fmt in formats:
            res1 = render_format(packet, fmt)
            res2 = render_format(packet, fmt)
            assert res1 == res2
            assert len(res1) > 0

def test_assert_format_preserves_packet():
    cases = get_format_cases()
    formats = [
        '1_baseline', '2_layout_d', '3_ultra_compact', '4_yaml_lite', 
        '5_minified_json', '6_markdown_table', '7_toon_rows', '8_source_table_facts'
    ]
    for case in cases:
        packet = layouts.EchoFramePacket.from_dict(case['packet'])
        for fmt in formats:
            res = render_format(packet, fmt)
            # This should not raise AssertionError
            assert_format_preserves_packet(packet, res)
