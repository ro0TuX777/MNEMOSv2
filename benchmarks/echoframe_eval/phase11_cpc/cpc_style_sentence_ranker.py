import math

class CPCStyleSentenceRanker:
    """
    Mock implementation of a Context-Aware Sentence Encoder.
    In production, this would use a cross-encoder or contrastively trained sentence-transformer.
    For this benchmark prototype, we simulate relevance scoring.
    """
    def __init__(self, use_random=False):
        self.use_random = use_random

    def rank(self, question: str, sentences: list, target_ratio: float) -> list:
        # Dummy TF-IDF style overlap for prototype ranking
        q_words = set(question.lower().split())
        
        scored = []
        for i, s in enumerate(sentences):
            s_words = set(s.lower().split())
            overlap = len(q_words.intersection(s_words))
            score = overlap / (len(s_words) + 1e-5)
            scored.append((i, s, score))
            
        # Sort by relevance
        scored.sort(key=lambda x: x[2], reverse=True)
        
        # Take top N
        num_keep = max(1, int(len(sentences) * target_ratio))
        selected = scored[:num_keep]
        
        # Restore original order
        selected.sort(key=lambda x: x[0])
        return [s[1] for s in selected]
