import os
import glob
import time
import json
import re
import math
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

SIGINT_CORPUS_DIR = "benchmarks/echoframe_eval/phase10/datasets/sigint_corpus/*.md"

PROTECTED_PATTERNS = {
    "source_pointer": r"(?i)source:\s*\[.*?\]|source\s+\d+|Source:",
    "governance_flag": r"(?i)governance:|approval_required|risk_label|kill switch",
    "evidence_gap": r"(?i)\[EVIDENCE_GAP\]|unknown|insufficient context",
    "contradiction": r"(?i)\[CONTRADICTION\]|conflict|disagree",
    "date": r"\b\d{4}-\d{2}-\d{2}\b",
    "number": r"\b\d+(?:\.\d+)?(?:%|M|K|B)?\b",
    "config_key": r"(?i)compact_semantic_minEvidence_hysteresis_v0|config_[a-z_]+",
    "exception_clause": r"(?i)\bexcept\b|\bunless\b|\bhowever\b|\bbut\b",
    "negation": r"(?i)\bnot\b|\bnever\b|\bno\b|\bcannot\b",
}

class TokenizerZipfMetrics:
    def compute(self, text: str):
        tokens = text.split()
        count = len(tokens)
        if count == 0:
            return {"COMPRESSION": 0, "CARDINALITY": 0, "AUC": 0, "SLOPE": 0, "POWER_LAW": 0}
        
        freqs = Counter(tokens)
        cardinality = len(freqs)
        sorted_freqs = sorted(freqs.values(), reverse=True)
        ranks, fs = [], []
        for i, f in enumerate(sorted_freqs):
            rank = i + 1
            log_rank = math.log(rank)
            if log_rank <= 6.0:
                ranks.append(log_rank)
                fs.append(math.log(f))
            else:
                break
                
        auc, slope, power_law_mae = 0.0, 0.0, 0.0
        if len(ranks) > 1:
            x = np.array(ranks)
            y = np.array(fs)
            auc = np.trapz(y, x)
            x_reshaped = x.reshape(-1, 1)
            model = LinearRegression().fit(x_reshaped, y)
            slope = model.coef_[0]
            power_law_mae = mean_absolute_error(y, model.predict(x_reshaped))

        return {"COMPRESSION": count, "CARDINALITY": cardinality, "AUC": auc, "SLOPE": slope, "POWER_LAW": power_law_mae}

def validate_spans(original, compressed):
    results = {}
    for key, pattern in PROTECTED_PATTERNS.items():
        orig_matches = re.findall(pattern, original)
        if not orig_matches:
            continue
        comp_matches = re.findall(pattern, compressed)
        orig_set = set(m.lower() for m in orig_matches)
        comp_set = set(m.lower() for m in comp_matches)
        results[key] = orig_set.issubset(comp_set)
    return results

def create_stable_packet(text):
    return f"[ECHO_FRAME_HEADER]\nGOVERNANCE: approval_required\n[FACTS]\nExtracted facts from SIGINT.\n[EVIDENCE]\n{text}\n[END_FRAME]"

def run_sigint_benchmark():
    files = glob.glob(SIGINT_CORPUS_DIR)
    print(f"Loaded {len(files)} SIGINT documents.")
    if not files: return
    
    if not PromptCompressor:
        print("LLMLingua not installed.")
        return
        
    print("Loading LLMLingua-2...")
    compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True)
    zipf = TokenizerZipfMetrics()
    
    # We will test Mode C (Direct) vs Mode E (Hybrid)
    modes = ["C", "E"]
    results = {m: [] for m in modes}
    
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            baseline = f.read()
        
        tokens_base = len(baseline.split())
        print(f"Processing {os.path.basename(fpath)} ({tokens_base} tokens)")
        
        contents = {}
        
        # Mode C: Direct
        t0 = time.time()
        c_res = compressor.compress_prompt(context=[baseline], target_token=max(10, int(tokens_base*0.5)), rank_method='longllmlingua')
        c_lat = (time.time() - t0) * 1000
        contents["C"] = c_res.get("compressed_prompt", baseline)
        
        # Mode E: Hybrid
        t0 = time.time()
        packet = create_stable_packet(baseline)
        ev_start = packet.find("[EVIDENCE]") + len("[EVIDENCE]\n")
        ev_end = packet.find("[END_FRAME]")
        evidence = packet[ev_start:ev_end]
        e_res = compressor.compress_prompt(context=[evidence], target_token=max(10, int(len(evidence.split())*0.5)), rank_method='longllmlingua')
        comp_ev = e_res.get("compressed_prompt", evidence)
        contents["E"] = packet[:ev_start] + comp_ev + packet[ev_end:]
        e_lat = (time.time() - t0) * 1000
        
        latencies = {"C": c_lat, "E": e_lat}
        
        for m in modes:
            out_text = contents[m]
            toks = len(out_text.split())
            preservation = validate_spans(baseline, out_text)
            z_metrics = zipf.compute(out_text)
            
            results[m].append({
                "file": os.path.basename(fpath),
                "tokens": toks,
                "latency_ms": latencies[m],
                "preservation": preservation,
                "zipf": z_metrics
            })
            
    print("\nSIGINT BENCHMARK RESULTS")
    print("="*60)
    for m in modes:
        mode_data = results[m]
        avg_toks = np.mean([r["tokens"] for r in mode_data])
        avg_lat = np.mean([r["latency_ms"] for r in mode_data])
        
        failed_docs = sum(1 for r in mode_data if not all(r["preservation"].values()))
        avg_power_law = np.mean([r["zipf"]["POWER_LAW"] for r in mode_data])
        
        print(f"Mode {m}:")
        print(f"  Avg Tokens: {avg_toks:.0f}")
        print(f"  Avg Latency: {avg_lat:.0f}ms")
        print(f"  Documents with Protected Span Failures: {failed_docs} / {len(mode_data)}")
        print(f"  Avg Zipf Power Law Deviation: {avg_power_law:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    run_sigint_benchmark()
