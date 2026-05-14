import json
from typing import Dict, Any

try:
    from mnemos_baseline_adapter import MnemosBaselineAdapter
except ImportError:
    MnemosBaselineAdapter = None

class EchoFrameCandidateAdapter:
    def __init__(self):
        if MnemosBaselineAdapter is None:
            raise RuntimeError("MnemosBaselineAdapter not found.")
        self.baseline_adapter = MnemosBaselineAdapter()

    def determine_decode_level(self, category: str) -> str:
        """
        Deterministic mapping from query category to EchoFrame decode level.
        """
        cat = category.lower()
        if "low-risk general" in cat:
            return "D1"
        elif "exact fact" in cat:
            return "D2"
        elif "code/api" in cat:
            return "D3"
        elif "policy/obligation" in cat:
            return "D3"
        elif "contradiction" in cat or "high-risk" in cat:
            return "D4"
        else:
            return "D3"

    def render_json_packet(self, baseline_results: Dict[str, Any], decode_level: str) -> str:
        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        packet = {
            "echoframe_version": "v0.1",
            "decode_level": decode_level,
            "evidence": []
        }
        
        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        for idx, text in enumerate(contents):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            
            ev_entry = {"id": ev_id, "source": src_id}
            
            if decode_level == "D0":
                pass # ID and source only
            elif decode_level == "D1":
                ev_entry["summary"] = text[:100] + "..." if len(text) > 100 else text
            elif decode_level == "D2":
                ev_entry["summary"] = text[:150] + "..." if len(text) > 150 else text
                ev_entry["provenance_key"] = "verified"
            elif decode_level == "D3":
                ev_entry["excerpt"] = text[:400] + "..." if len(text) > 400 else text
                ev_entry["provenance_key"] = "verified"
            elif decode_level == "D4":
                ev_entry["raw_content"] = text
                ev_entry["provenance_key"] = "verified"
                
            packet["evidence"].append(ev_entry)
            
        return json.dumps(packet, indent=2)

    def render_compact_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str) -> str:
        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1

        lines = [
            "[EF:v0.1 mode=compact]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines)

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        lines.append("EVIDENCE:")
        for idx, text in enumerate(contents):
            if decode_level == "D0":
                lines.append(f"- [S{idx+1}] [ID_ONLY]")
            elif decode_level == "D1":
                excerpt = text[:100].replace("\n", " ") + "..." if len(text) > 100 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D2":
                excerpt = text[:150].replace("\n", " ") + "..." if len(text) > 150 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D3":
                excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D4":
                lines.append(f"- [S{idx+1}] {text}")

        return "\n".join(lines)

    def render_compact_safe_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str) -> str:
        import re
        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1

        lines = [
            "[EF:v0.1 mode=compact_safe]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines)

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        facts = set()
        for text in contents:
            # Check numbers
            for n in re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', text):
                if len(n) <= 15:
                    facts.add(f"threshold: {n}")
            # Check dates
            for d in re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', text):
                facts.add(f"date: {d}")
            # Config keys
            for k in re.findall(r'\b[A-Z_][A-Z0-9_]{3,}\b|\b[a-z_]+_[a-z0-9_]+\b', text):
                if len(k) <= 40:
                    facts.add(f"key: {k}")
            # Exception clauses
            for c in re.findall(r'\b(?:except|unless|must|shall|required|prohibited|may not|not)\b[^\.\n;]{0,50}', text, re.IGNORECASE):
                facts.add(f"clause: {c.strip()}")

        if facts:
            lines.append("FACTS:")
            for f in sorted(list(facts)):
                lines.append(f"- {f}")

        lines.append("EVIDENCE:")
        for idx, text in enumerate(contents):
            if decode_level == "D0":
                lines.append(f"- [S{idx+1}] [ID_ONLY]")
            elif decode_level == "D1":
                excerpt = text[:100].replace("\n", " ") + "..." if len(text) > 100 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D2":
                excerpt = text[:150].replace("\n", " ") + "..." if len(text) > 150 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D3":
                excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D4":
                lines.append(f"- [S{idx+1}] {text}")

        return "\n".join(lines)

    def search(self, query: str, category: str, top_k: int = 10, mode: str = "json_v0") -> Dict[str, Any]:
        """
        Executes baseline MNEMOS search, then applies the candidate EchoFrame rendering.
        """
        baseline_res = self.baseline_adapter.search(query, top_k)
        
        if baseline_res.get("errors"):
             return baseline_res # Propagate errors directly without EchoFrame rendering
             
        decode_level = self.determine_decode_level(category)
        
        if mode == "compact_v0":
            packet_str = self.render_compact_packet(baseline_res, decode_level, category, query)
    def render_compact_selective_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str) -> str:
        import re
        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1

        lines = [
            "[EF:v0.1 mode=compact_selective]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines)

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        all_facts = []
        for text in contents:
            for n in re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', text):
                if len(n) <= 15: all_facts.append(("threshold", f"threshold: {n}"))
            for d in re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', text):
                all_facts.append(("date", f"date: {d}"))
            for k in re.findall(r'\b[A-Z_][A-Z0-9_]{3,}\b|\b[a-z_]+_[a-z0-9_]+\b', text):
                if len(k) <= 40: all_facts.append(("key", f"key: {k}"))
            for c in re.findall(r'\b(?:except|unless|must|shall|required|prohibited|may not|not)\b[^\.\n;]{0,50}', text, re.IGNORECASE):
                all_facts.append(("clause", f"clause: {c.strip()}"))

        # Selective filtering logic
        pinned_facts = set()
        cat_lower = category.lower()
        
        # High risk, contradiction, stale vs current, unknown all trigger fallback behavior essentially (keep everything)
        ambiguous = False
        if any(x in cat_lower for x in ["high-risk", "contradiction", "stale", "unknown", "insufficient"]):
            ambiguous = True
        elif "numeric" in cat_lower or "threshold" in cat_lower:
            for typ, f in all_facts:
                if typ == "threshold": pinned_facts.add(f)
        elif "date" in cat_lower:
            for typ, f in all_facts:
                if typ == "date": pinned_facts.add(f)
        elif "config" in cat_lower or "api" in cat_lower:
            for typ, f in all_facts:
                if typ == "key": pinned_facts.add(f)
        elif "policy" in cat_lower or "obligation" in cat_lower or "exception" in cat_lower:
            for typ, f in all_facts:
                if typ == "clause": pinned_facts.add(f)
        else:
            ambiguous = True # Unrecognized specific category, keep all facts

        if ambiguous:
            for typ, f in all_facts:
                pinned_facts.add(f)
            lines[0] = "[EF:v0.1 mode=compact_selective fallback=compact_safe_v0]"

        if pinned_facts:
            lines.append("FACTS:")
            for f in sorted(list(pinned_facts)):
                lines.append(f"- {f}")

        lines.append("EVIDENCE:")
        for idx, text in enumerate(contents):
            if decode_level == "D0":
                lines.append(f"- [S{idx+1}] [ID_ONLY]")
            elif decode_level == "D1":
                excerpt = text[:100].replace("\n", " ") + "..." if len(text) > 100 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D2":
                excerpt = text[:150].replace("\n", " ") + "..." if len(text) > 150 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D3":
                excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D4":
                lines.append(f"- [S{idx+1}] {text}")

        return "\n".join(lines)

    def render_compact_semantic_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str, semantic_keep: float = 0.70, mixed_keep: float = 0.55, budget_aware: bool = False, category_aware: bool = False) -> str:
        import re
        import numpy as np
        
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "scorer_model"):
                self.scorer_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self.scorer_model = None

        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1
            
        cat_lower = category.lower()
        ambiguous_risk = any(x in cat_lower for x in ["high-risk", "contradiction", "stale", "unknown", "insufficient"])

        lines = [
            "[EF:v0.1 mode=compact_semantic]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines)

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        all_facts = []
        for text in contents:
            for n in re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', text):
                if len(n) <= 15: all_facts.append(("threshold", f"threshold: {n}"))
            for d in re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', text):
                all_facts.append(("date", f"date: {d}"))
            for k in re.findall(r'\b[A-Z_][A-Z0-9_]{3,}\b|\b[a-z_]+_[a-z0-9_]+\b', text):
                if len(k) <= 40: all_facts.append(("key", f"key: {k}"))
            for c in re.findall(r'\b(?:except|unless|must|shall|required|prohibited|may not|not)\b[^\.\n;]{0,50}', text, re.IGNORECASE):
                all_facts.append(("clause", f"clause: {c.strip()}"))

        pinned_facts = set()
        query_lower = query.lower()

        # Fallback condition check
        if self.scorer_model is None:
            # Fallback to safe mode
            for typ, f in all_facts: pinned_facts.add(f)
            lines[0] = "[EF:v0.1 mode=compact_semantic fallback=compact_safe_v0 reason=embedding_unavailable]"
        else:
            try:
                q_emb = self.scorer_model.encode([query])
                
                # Deduplicate before scoring
                unique_facts = list(set(all_facts))
                scored_unpinned_facts = []
                
                # If category_aware is True, adjust thresholds
                if category_aware:
                    if any(x in cat_lower for x in ["numeric", "threshold", "date", "config", "api", "policy", "obligation", "exception", "high-risk"]):
                        current_semantic_keep = 0.70
                        current_mixed_keep = 0.55
                    else:
                        current_semantic_keep = 0.60
                        current_mixed_keep = 0.45
                else:
                    current_semantic_keep = semantic_keep
                    current_mixed_keep = mixed_keep
                
                for typ, f in unique_facts:
                    # Safety Pinned rules
                    is_pinned = False
                    
                    if ambiguous_risk:
                        is_pinned = True
                    elif typ == "threshold" and "numeric" in cat_lower:
                        is_pinned = True
                    elif typ == "date" and "date" in cat_lower:
                        is_pinned = True
                    elif typ == "key" and "config" in cat_lower:
                        is_pinned = True
                    elif typ == "clause" and any(x in cat_lower for x in ["policy", "obligation", "exception"]):
                        is_pinned = True
                    
                    if is_pinned:
                        pinned_facts.add(f)
                        continue
                        
                    # Semantic scoring for unpinned facts
                    f_emb = self.scorer_model.encode([f])
                    score = np.dot(q_emb[0], f_emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(f_emb[0]))
                    
                    scored_unpinned_facts.append((f, score))
                    
                if budget_aware:
                    # Sort by score descending
                    scored_unpinned_facts.sort(key=lambda x: x[1], reverse=True)
                    # For budget aware, just keep top N that might fit in a budget, 
                    # but actually we can just take the top 2 non-pinned facts if they exist
                    # Or keep anything > 0.65. The budget logic will be handled outside,
                    # but here we can just use the provided semantic_keep threshold that will be swept.
                    for f, score in scored_unpinned_facts:
                        if score >= semantic_keep:
                            pinned_facts.add(f)
                        elif score >= mixed_keep:
                            words = set(re.findall(r'\w+', f.lower()))
                            q_words = set(re.findall(r'\w+', query_lower))
                            if len(words.intersection(q_words)) > 0:
                                pinned_facts.add(f)
                else:
                    for f, score in scored_unpinned_facts:
                        if score >= current_semantic_keep:
                            pinned_facts.add(f)
                        elif score >= current_mixed_keep:
                            # lexical overlap
                            words = set(re.findall(r'\w+', f.lower()))
                            q_words = set(re.findall(r'\w+', query_lower))
                            if len(words.intersection(q_words)) > 0:
                                pinned_facts.add(f)
            except Exception as e:
                # Scorer error fallback
                for typ, f in all_facts: pinned_facts.add(f)
                lines[0] = "[EF:v0.1 mode=compact_semantic fallback=compact_safe_v0 reason=scorer_error]"

        if pinned_facts:
            lines.append("FACTS:")
            for f in sorted(list(pinned_facts)):
                lines.append(f"- {f}")

        lines.append("EVIDENCE:")
        for idx, text in enumerate(contents):
            if decode_level == "D0":
                lines.append(f"- [S{idx+1}] [ID_ONLY]")
            elif decode_level == "D1":
                excerpt = text[:100].replace("\n", " ") + "..." if len(text) > 100 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D2":
                excerpt = text[:150].replace("\n", " ") + "..." if len(text) > 150 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D3":
                excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                lines.append(f"- [S{idx+1}] {excerpt}")
            elif decode_level == "D4":
                lines.append(f"- [S{idx+1}] {text}")

        return "\n".join(lines)

    def render_compact_semantic_minEvidence_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str) -> str:
        import re
        import numpy as np
        
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "scorer_model"):
                self.scorer_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self.scorer_model = None

        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1
            
        cat_lower = category.lower()
        ambiguous_risk = any(x in cat_lower for x in ["high-risk", "contradiction", "stale", "unknown", "insufficient"])

        lines = [
            "[EF:v0.1 mode=compact_semantic_minEvidence]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines)

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        all_facts = []
        for text in contents:
            for n in re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', text):
                if len(n) <= 15: all_facts.append(("threshold", f"threshold: {n}"))
            for d in re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', text):
                all_facts.append(("date", f"date: {d}"))
            for k in re.findall(r'\b[A-Z_][A-Z0-9_]{3,}\b|\b[a-z_]+_[a-z0-9_]+\b', text):
                if len(k) <= 40: all_facts.append(("key", f"key: {k}"))
            for c in re.findall(r'\b(?:except|unless|must|shall|required|prohibited|may not|not)\b[^\.\n;]{0,50}', text, re.IGNORECASE):
                all_facts.append(("clause", f"clause: {c.strip()}"))

        pinned_facts = set()
        query_lower = query.lower()

        if self.scorer_model is None:
            for typ, f in all_facts: pinned_facts.add(f)
            lines[0] = "[EF:v0.1 mode=compact_semantic_minEvidence fallback=compact_safe_v0 reason=embedding_unavailable]"
        else:
            try:
                q_emb = self.scorer_model.encode([query])
                unique_facts = list(set(all_facts))
                
                for typ, f in unique_facts:
                    is_pinned = False
                    if ambiguous_risk: is_pinned = True
                    elif typ == "threshold" and "numeric" in cat_lower: is_pinned = True
                    elif typ == "date" and "date" in cat_lower: is_pinned = True
                    elif typ == "key" and "config" in cat_lower: is_pinned = True
                    elif typ == "clause" and any(x in cat_lower for x in ["policy", "obligation", "exception"]): is_pinned = True
                    
                    if is_pinned:
                        pinned_facts.add(f)
                        continue
                        
                    f_emb = self.scorer_model.encode([f])
                    score = np.dot(q_emb[0], f_emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(f_emb[0]))
                    
                    if score >= 0.70:
                        pinned_facts.add(f)
                    elif score >= 0.55:
                        words = set(re.findall(r'\w+', f.lower()))
                        q_words = set(re.findall(r'\w+', query_lower))
                        if len(words.intersection(q_words)) > 0:
                            pinned_facts.add(f)
            except Exception as e:
                for typ, f in all_facts: pinned_facts.add(f)
                lines[0] = "[EF:v0.1 mode=compact_semantic_minEvidence fallback=compact_safe_v0 reason=scorer_error]"

        if pinned_facts:
            lines.append("FACTS:")
            for f in sorted(list(pinned_facts)):
                lines.append(f"- {f}")

        lines.append("EVIDENCE:")
        for idx, text in enumerate(contents):
            source_pinned_facts = []
            for f in pinned_facts:
                f_val = f.split(": ", 1)[1] if ": " in f else f
                if f_val in text:
                    source_pinned_facts.append(f_val)
                    
            if not source_pinned_facts:
                if ambiguous_risk:
                    excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                    lines.append(f"- [S{idx+1}] [E3] {excerpt}")
                else:
                    lines.append(f"- [S{idx+1}] [E0] [ID_ONLY]")
            else:
                windows = []
                for f_val in source_pinned_facts:
                    for m in re.finditer(re.escape(f_val), text):
                        start = max(0, m.start() - 80)
                        end = min(len(text), m.end() + 120)
                        windows.append([start, end])
                        
                windows.sort(key=lambda x: x[0])
                merged = []
                for w in windows:
                    if not merged: merged.append(w)
                    else:
                        prev = merged[-1]
                        if w[0] <= prev[1] + 20: # merge if very close
                            prev[1] = max(prev[1], w[1])
                        else:
                            merged.append(w)
                            
                snippets = []
                for w in merged:
                    snip = text[w[0]:w[1]].replace("\n", " ")
                    if w[0] > 0: snip = "..." + snip
                    if w[1] < len(text): snip = snip + "..."
                    snippets.append(snip)
                    
                joined = " | ".join(snippets)
                lines.append(f"- [S{idx+1}] [E2] {joined}")

        return "\n".join(lines)

    def render_compact_semantic_minEvidence_hysteresis_packet(self, baseline_results: Dict[str, Any], decode_level: str, category: str, query: str, state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        # This will use the base logic but apply hysteresis overrides
        # We need to return the packet string and the updated state
        import re
        import numpy as np
        
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, "scorer_model"):
                self.scorer_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            self.scorer_model = None

        raw_context_str = baseline_results.get("rendered_context", "[]")
        try:
            contents = json.loads(raw_context_str)
            if not isinstance(contents, list):
                contents = [raw_context_str]
        except json.JSONDecodeError:
            contents = [raw_context_str]

        evidence_ids = baseline_results.get("selected_evidence_ids", [])
        source_ids = baseline_results.get("selected_source_ids", [])

        query_id = baseline_results.get("query_id", "Q-???")
        gap = 1 if len(baseline_results.get("evidence_gaps", [])) > 0 else 0
        contra = 1 if len(baseline_results.get("contradiction_flags", [])) > 0 else 0
        approval = 1 if baseline_results.get("approval_required", False) else 0
        
        unknown = 0
        if "insufficient evidence" in category.lower() and not evidence_ids:
            unknown = 1
        elif baseline_results.get("unknown_preserved", False):
            unknown = 1
            
        cat_lower = category.lower()
        ambiguous_risk = any(x in cat_lower for x in ["high-risk", "contradiction", "stale", "unknown", "insufficient"])

        lines = [
            "[EF:v0.1 mode=compact_semantic_minEvidence_hysteresis]",
            f"Q: {query_id} / {category}",
            f"D: {decode_level}",
            f"GOV: gap={gap} contra={contra} approval={approval} unknown={unknown}"
        ]

        if not contents or not evidence_ids:
            lines.append("EVIDENCE:\n- [NO_EVIDENCE_FOUND]")
            return "\n".join(lines), state

        lines.append("SRC:")
        for idx in range(len(contents)):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            ev_id = evidence_ids[idx] if idx < len(evidence_ids) else "unknown"
            lines.append(f"- S{idx+1}:{src_id}#{ev_id}")

        all_facts = []
        for text in contents:
            for n in re.findall(r'\b\d+(?:\.\d+)?(?:[a-zA-Z%]+)?\b', text):
                if len(n) <= 15: all_facts.append(("threshold", f"threshold: {n}"))
            for d in re.findall(r'\b20\d{2}(?:-\d{2}-\d{2})?\b', text):
                all_facts.append(("date", f"date: {d}"))
            for k in re.findall(r'\b[A-Z_][A-Z0-9_]{3,}\b|\b[a-z_]+_[a-z0-9_]+\b', text):
                if len(k) <= 40: all_facts.append(("key", f"key: {k}"))
            for c in re.findall(r'\b(?:except|unless|must|shall|required|prohibited|may not|not)\b[^\.\n;]{0,50}', text, re.IGNORECASE):
                all_facts.append(("clause", f"clause: {c.strip()}"))

        pinned_facts = set()
        query_lower = query.lower()

        if self.scorer_model is None:
            for typ, f in all_facts: pinned_facts.add(f)
            lines[0] = "[EF:v0.1 mode=compact_semantic_minEvidence_hysteresis fallback=compact_safe_v0 reason=embedding_unavailable]"
        else:
            try:
                q_emb = self.scorer_model.encode([query])
                unique_facts = list(set(all_facts))
                
                for typ, f in unique_facts:
                    is_pinned = False
                    if ambiguous_risk: is_pinned = True
                    elif typ == "threshold" and "numeric" in cat_lower: is_pinned = True
                    elif typ == "date" and "date" in cat_lower: is_pinned = True
                    elif typ == "key" and "config" in cat_lower: is_pinned = True
                    elif typ == "clause" and any(x in cat_lower for x in ["policy", "obligation", "exception"]): is_pinned = True
                    
                    if is_pinned:
                        pinned_facts.add(f)
                        continue
                        
                    f_emb = self.scorer_model.encode([f])
                    score = np.dot(q_emb[0], f_emb[0]) / (np.linalg.norm(q_emb[0]) * np.linalg.norm(f_emb[0]))
                    
                    if score >= 0.70:
                        pinned_facts.add(f)
                    elif score >= 0.55:
                        words = set(re.findall(r'\w+', f.lower()))
                        q_words = set(re.findall(r'\w+', query_lower))
                        if len(words.intersection(q_words)) > 0:
                            pinned_facts.add(f)
            except Exception as e:
                for typ, f in all_facts: pinned_facts.add(f)
                lines[0] = "[EF:v0.1 mode=compact_semantic_minEvidence_hysteresis fallback=compact_safe_v0 reason=scorer_error]"

        if pinned_facts:
            lines.append("FACTS:")
            for f in sorted(list(pinned_facts)):
                lines.append(f"- {f}")

        new_levels = {}
        decisions = []
        lines.append("EVIDENCE:")
        
        prev_levels = state.get("evidence_render_levels", {})
        
        for idx, text in enumerate(contents):
            src_id = source_ids[idx] if idx < len(source_ids) else "unknown"
            source_pinned_facts = []
            for f in pinned_facts:
                f_val = f.split(": ", 1)[1] if ": " in f else f
                if f_val in text:
                    source_pinned_facts.append(f_val)
                    
            proposed_level = "E0"
            proposed_excerpt = ""
            
            if not source_pinned_facts:
                if ambiguous_risk:
                    excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                    proposed_level = "E3"
                    proposed_excerpt = excerpt
                else:
                    proposed_level = "E0"
                    proposed_excerpt = "[ID_ONLY]"
            else:
                windows = []
                for f_val in source_pinned_facts:
                    for m in re.finditer(re.escape(f_val), text):
                        start = max(0, m.start() - 80)
                        end = min(len(text), m.end() + 120)
                        windows.append([start, end])
                        
                windows.sort(key=lambda x: x[0])
                merged = []
                for w in windows:
                    if not merged: merged.append(w)
                    else:
                        prev = merged[-1]
                        if w[0] <= prev[1] + 20: 
                            prev[1] = max(prev[1], w[1])
                        else:
                            merged.append(w)
                            
                snippets = []
                for w in merged:
                    snip = text[w[0]:w[1]].replace("\n", " ")
                    if w[0] > 0: snip = "..." + snip
                    if w[1] < len(text): snip = snip + "..."
                    snippets.append(snip)
                    
                joined = " | ".join(snippets)
                proposed_level = "E2"
                proposed_excerpt = joined

            # HYSTERESIS LOGIC
            prev_lvl = prev_levels.get(src_id, None)
            final_level = proposed_level
            final_excerpt = proposed_excerpt
            
            # Helper to compare levels
            lvl_rank = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
            
            if prev_lvl is not None:
                # If risk increases, expand quickly (already handled by proposed level generally)
                # If risk decreases, compress slowly
                # If prev was high (E3) and proposed is E0, don't drop to E0 immediately unless irrelevant
                
                # If source carries governance/contradiction
                has_gov = ambiguous_risk
                
                if has_gov:
                    # Retain longer
                    if lvl_rank[proposed_level] < lvl_rank[prev_lvl]:
                        # prevent unsafe compression
                        final_level = prev_lvl
                        # Rebuild excerpt for the retained level if it's E3
                        if final_level == "E3":
                            final_excerpt = text[:400].replace("\n", " ") + "..." if len(text) > 400 else text.replace("\n", " ")
                        decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "retain", "reason": "governance_signal"})
                    else:
                        decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "expand", "reason": "risk_increase"})
                else:
                    if lvl_rank[proposed_level] < lvl_rank[prev_lvl]:
                        if proposed_level == "E0":
                            decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "drop", "reason": "irrelevant"})
                        else:
                            decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "compress", "reason": "risk_decrease"})
                    elif lvl_rank[proposed_level] > lvl_rank[prev_lvl]:
                        decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "expand", "reason": "high_relevance"})
                    else:
                        decisions.append({"source_id": src_id, "previous_level": prev_lvl, "new_level": final_level, "decision": "retain", "reason": "high_relevance"})
            
            new_levels[src_id] = final_level
            lines.append(f"- [S{idx+1}] [{final_level}] {final_excerpt}")

        new_state = {
            "evidence_render_levels": new_levels,
            "hysteresis_decisions": decisions
        }
        return "\n".join(lines), new_state

    def search(self, query: str, category: str, top_k: int = 10, mode: str = "json_v0", semantic_keep: float = 0.70, mixed_keep: float = 0.55, budget_aware: bool = False, category_aware: bool = False, hysteresis_state: Dict[str, Any] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executes baseline MNEMOS search, then applies the candidate EchoFrame rendering.
        Returns a tuple of (packet_result, updated_hysteresis_state) if state is provided, else returns just packet_result.
        """
        baseline_res = self.baseline_adapter.search(query, top_k)
        
        if baseline_res.get("errors"):
             if hysteresis_state is not None:
                 return baseline_res, hysteresis_state
             return baseline_res # Propagate errors directly without EchoFrame rendering
             
        decode_level = self.determine_decode_level(category)
        
        new_state = hysteresis_state or {}
        
        if mode == "compact_v0":
            packet_str = self.render_compact_packet(baseline_res, decode_level, category, query)
        elif mode == "compact_safe_v0":
            packet_str = self.render_compact_safe_packet(baseline_res, decode_level, category, query)
        elif mode == "compact_selective_v0":
            packet_str = self.render_compact_selective_packet(baseline_res, decode_level, category, query)
        elif mode == "compact_semantic_v0":
            packet_str = self.render_compact_semantic_packet(baseline_res, decode_level, category, query, semantic_keep, mixed_keep, budget_aware, category_aware)
        elif mode == "compact_semantic_minEvidence_v0":
            packet_str = self.render_compact_semantic_minEvidence_packet(baseline_res, decode_level, category, query)
        elif mode == "compact_semantic_minEvidence_hysteresis_v0":
            packet_str, new_state = self.render_compact_semantic_minEvidence_hysteresis_packet(baseline_res, decode_level, category, query, new_state)
        else:
            packet_str = self.render_json_packet(baseline_res, decode_level)
            
        candidate_tokens = int(len(packet_str) / 4)
        baseline_tokens = baseline_res.get("context_token_count", 1)
        
        ratio = round(candidate_tokens / max(1, baseline_tokens), 4)

        # For unknown/insufficient evidence queries, preserve the flag
        unknown_preserved = False
        if "insufficient evidence" in category.lower():
            # In a real system, the gap detection logic sets this.
            # Here we carry forward the baseline's state, or set it if no evidence found.
            if not baseline_res.get("selected_evidence_ids"):
                unknown_preserved = True

        res_dict = {
            "selected_evidence_ids": baseline_res.get("selected_evidence_ids", []),
            "selected_source_ids": baseline_res.get("selected_source_ids", []),
            "baseline_context_token_count": baseline_tokens,
            "candidate_context_token_count": candidate_tokens,
            "token_reduction_ratio": ratio,
            "decode_level": decode_level,
            "rendered_echoframe_packet": packet_str,
            "provenance_present": baseline_res.get("provenance_present", False),
            "evidence_gaps": baseline_res.get("evidence_gaps", []),
            "contradiction_flags": baseline_res.get("contradiction_flags", []),
            "governance_flags": baseline_res.get("governance_flags", []),
            "approval_required": baseline_res.get("approval_required", False),
            "unknown_preserved": unknown_preserved or baseline_res.get("unknown_preserved", False),
            "errors": baseline_res.get("errors", None),
            "notes": "EchoFrame candidate packet successfully rendered."
        }
        
        if hysteresis_state is not None:
            return res_dict, new_state
        return res_dict
