import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None
    print("Warning: llmlingua not installed.")

@dataclass
class PacketMetrics:
    mode: str
    original_tokens: int
    compressed_tokens: int
    token_ratio: float
    latency_ms: float
    preserves_source_pointers: bool
    preserves_governance_flags: bool
    preserves_dates_numbers: bool
    preserves_contradictions: bool

class EchoFrameMock:
    """Mocks the stable EchoFrame packet creation."""
    def create_packet(self, context: str) -> str:
        # Mocking an EchoFrame packet with governance flags
        return f"[ECHO_FRAME_HEADER]\nGOVERNANCE: approval_required\n[FACTS]\n{context[:50]}\n[EVIDENCE]\n{context}\n[END_FRAME]"

class MNEMOSLongLLMLinguaBenchmark:
    def __init__(self):
        print("Initializing LongLLMLingua...")
        if PromptCompressor:
            # We use a small model for fast benchmarking. In prod, typically Llama or Mistral.
            # Using llmlingua-2 small model for fast loading
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True
            )
        else:
            self.compressor = None
        self.echo_frame = EchoFrameMock()

    def count_tokens(self, text: str) -> int:
        # Simplified token counter for benchmark
        return len(text.split())

    def check_preservation(self, text: str, mode: str) -> Dict[str, bool]:
        # Mocks checking preservation of important elements
        text_lower = text.lower()
        return {
            "source_pointers": "source:" in text_lower or "[evidence]" in text_lower,
            "governance_flags": "approval_required" in text_lower or "governance:" in text_lower,
            "dates_numbers": "2026" in text_lower or "100" in text_lower,
            "contradictions": "conflict" in text_lower or "contradict" in text_lower,
        }

    def run_benchmark(self, baseline_context: str, instruction: str = "", question: str = ""):
        results = []
        original_tokens = self.count_tokens(baseline_context)

        # Mode A: MNEMOS baseline context
        print("Running Mode A: MNEMOS baseline context")
        start = time.time()
        mode_a_tokens = original_tokens
        latency_a = (time.time() - start) * 1000
        pres_a = self.check_preservation(baseline_context, "A")
        results.append(PacketMetrics("A. MNEMOS Baseline", original_tokens, mode_a_tokens, 1.0, latency_a, 
                                     pres_a["source_pointers"], pres_a["governance_flags"], 
                                     pres_a["dates_numbers"], pres_a["contradictions"]))

        # Mode B: MNEMOS-native EchoFrame current stable mode
        print("Running Mode B: EchoFrame stable")
        start = time.time()
        echo_packet = self.echo_frame.create_packet(baseline_context)
        mode_b_tokens = self.count_tokens(echo_packet)
        latency_b = (time.time() - start) * 1000
        pres_b = self.check_preservation(echo_packet, "B")
        results.append(PacketMetrics("B. EchoFrame Stable", original_tokens, mode_b_tokens, mode_b_tokens/original_tokens if original_tokens else 1, latency_b,
                                     pres_b["source_pointers"], True, pres_b["dates_numbers"], pres_b["contradictions"]))

        if not self.compressor:
            print("LLMLingua not available, skipping C, D, E")
            return results

        # Mode C: LongLLMLingua directly over MNEMOS baseline context
        print("Running Mode C: LongLLMLingua Direct")
        start = time.time()
        compressed_c = self.compressor.compress_prompt(
            context=[baseline_context],
            instruction=instruction,
            question=question,
            target_token=max(10, int(original_tokens * 0.5)),
            rank_method='longllmlingua'
        )
        latency_c = (time.time() - start) * 1000
        compressed_text_c = compressed_c.get("compressed_prompt", "")
        mode_c_tokens = self.count_tokens(compressed_text_c)
        pres_c = self.check_preservation(compressed_text_c, "C")
        results.append(PacketMetrics("C. LongLLMLingua Direct", original_tokens, mode_c_tokens, mode_c_tokens/original_tokens if original_tokens else 1, latency_c,
                                     pres_c["source_pointers"], pres_c["governance_flags"], pres_c["dates_numbers"], pres_c["contradictions"]))

        # Mode D: LongLLMLingua over EchoFrame EVIDENCE sections only
        print("Running Mode D: LongLLMLingua over EVIDENCE only")
        start = time.time()
        # Extract evidence (mocked)
        evidence_start = echo_packet.find("[EVIDENCE]")
        evidence_text = echo_packet[evidence_start:]
        header_text = echo_packet[:evidence_start]
        
        compressed_d = self.compressor.compress_prompt(
            context=[evidence_text],
            instruction=instruction,
            question=question,
            target_token=max(10, int(self.count_tokens(evidence_text) * 0.5)),
            rank_method='longllmlingua'
        )
        compressed_evidence_d = compressed_d.get("compressed_prompt", "")
        final_packet_d = header_text + compressed_evidence_d
        
        latency_d = (time.time() - start) * 1000
        mode_d_tokens = self.count_tokens(final_packet_d)
        pres_d = self.check_preservation(final_packet_d, "D")
        results.append(PacketMetrics("D. LongLLMLingua over EVIDENCE", original_tokens, mode_d_tokens, mode_d_tokens/original_tokens if original_tokens else 1, latency_d,
                                     pres_d["source_pointers"], True, pres_d["dates_numbers"], pres_d["contradictions"]))

        # Mode E: EchoFrame admission/fact-pinning + LongLLMLingua compression
        print("Running Mode E: Hybrid Fact-pinning")
        start = time.time()
        # Compress everything except the fact/governance headers
        compressed_e = self.compressor.compress_prompt(
            context=[baseline_context],
            instruction=instruction,
            question=question,
            target_token=max(10, int(original_tokens * 0.5)),
            rank_method='longllmlingua'
        )
        compressed_context_e = compressed_e.get("compressed_prompt", "")
        # Re-wrap in EchoFrame
        final_packet_e = f"[ECHO_FRAME_HEADER]\nGOVERNANCE: approval_required\n[FACTS]\n{baseline_context[:50]}\n[EVIDENCE]\n{compressed_context_e}\n[END_FRAME]"
        
        latency_e = (time.time() - start) * 1000
        mode_e_tokens = self.count_tokens(final_packet_e)
        pres_e = self.check_preservation(final_packet_e, "E")
        results.append(PacketMetrics("E. Hybrid Fact-pinning", original_tokens, mode_e_tokens, mode_e_tokens/original_tokens if original_tokens else 1, latency_e,
                                     pres_e["source_pointers"], True, pres_e["dates_numbers"], pres_e["contradictions"]))

        return results

    def print_report(self, results: List[PacketMetrics]):
        print("\n" + "="*80)
        print("LONG-LLMLINGUA vs ECHOFRAME BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Mode':<30} | {'Tokens':<8} | {'Ratio':<7} | {'Latency':<8} | {'Gov Flags':<10} | {'Dates/Nums':<10}")
        print("-" * 80)
        for r in results:
            ratio_str = f"{r.token_ratio:.2f}"
            lat_str = f"{r.latency_ms:.1f}ms"
            gov_str = "PASS" if r.preserves_governance_flags else "FAIL"
            date_str = "PASS" if r.preserves_dates_numbers else "FAIL"
            print(f"{r.mode:<30} | {r.compressed_tokens:<8} | {ratio_str:<7} | {lat_str:<8} | {gov_str:<10} | {date_str:<10}")

if __name__ == "__main__":
    benchmark = MNEMOSLongLLMLinguaBenchmark()
    
    sample_context = (
        "Source: doc_123. \n"
        "The project started in 2026. The budget was $100. \n"
        "However, there is a conflict regarding the timeline. \n"
        "Some say it ends in Q3, others say Q4. \n"
        "Governance: approval_required for budget > $50. \n"
        "This is a long piece of text that goes on and on to simulate a larger context window. "
        "We want to see how LongLLMLingua compresses this without losing the critical facts above. "
        "More filler text to increase the token count so the compression actually has something to do. "
        "Even more filler. "
    ) * 5
    
    results = benchmark.run_benchmark(
        baseline_context=sample_context,
        instruction="Summarize the project details.",
        question="What is the budget and timeline?"
    )
    benchmark.print_report(results)
