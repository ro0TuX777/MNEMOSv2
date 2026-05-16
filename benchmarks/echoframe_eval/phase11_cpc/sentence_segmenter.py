import re

class SentenceSegmenter:
    """
    Splits text into sentences, bullet clauses, numbered policy statements,
    section headers, and table rows to preserve military/policy boundaries.
    """
    def __init__(self):
        # Matches military numbering (e.g. 1.2.3), clauses (e.g. (a)), and standard sentence endings
        self.split_pattern = r"(?:\n\s*\d+\.\d+(?:\.\d+)*\s+)|(?:\n\s*\([a-z]\)\s+)|(?:(?<=[.!?])\s+(?=[A-Z]))|(?:\n{2,})"

    def segment(self, text: str):
        # Heuristic segmentation
        segments = re.split(self.split_pattern, text)
        cleaned = [seg.strip() for seg in segments if seg.strip()]
        
        # Further refine by looking for explicit obligation lines if they got mashed
        final_segments = []
        for c in cleaned:
            if "\n" in c and len(c) > 200:
                # If chunk is too large, split by newlines as a fallback for tables/lists
                sub_segs = c.split("\n")
                final_segments.extend([s.strip() for s in sub_segs if s.strip()])
            else:
                final_segments.append(c)
                
        return final_segments
