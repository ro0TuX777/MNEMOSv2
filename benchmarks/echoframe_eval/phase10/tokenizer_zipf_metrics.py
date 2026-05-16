import json
import math
from collections import Counter
from typing import List, Dict, Tuple

try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
except ImportError:
    np = None
    LinearRegression = None
    mean_absolute_error = None
    print("Warning: numpy or sklearn not installed. Please install them for accurate metrics.")

class TokenizerZipfMetrics:
    """
    Implements metrics defined in 'Beyond Text Compression: Evaluating Tokenizers Across Scales'.
    Metrics are computed on log-log frequency-rank distributions, restricted to log(rank) <= 6.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_metrics(self, text: str) -> Dict[str, float]:
        # Tokenize the text
        if hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, "__call__"):
            tokens = self.tokenizer(text)
        else:
            # Fallback for simple testing
            tokens = text.split()
            
        token_count = len(tokens)
        
        # 1. CARDINALITY: number of unique tokens
        frequencies = Counter(tokens)
        cardinality = len(frequencies)
        
        # Sort by frequency descending
        sorted_freqs = sorted(frequencies.values(), reverse=True)
        
        # Metrics defined for log(rank) <= 6
        # log is natural log in the paper usually, but rank starts at 1
        ranks = []
        freqs = []
        for i, f in enumerate(sorted_freqs):
            rank = i + 1
            log_rank = math.log(rank)
            if log_rank <= 6.0:
                ranks.append(log_rank)
                freqs.append(math.log(f))
            else:
                break
                
        # Calculate metrics if we have enough points and modules
        auc = 0.0
        slope = 0.0
        power_law_mae = 0.0
        
        if np and LinearRegression and len(ranks) > 1:
            x = np.array(ranks)
            y = np.array(freqs)
            
            # 2. AUC using Simpson's rule (approximated with trapezoidal here for simplicity, or scipy.integrate.simpson)
            try:
                from scipy.integrate import simpson
                auc = simpson(y=y, x=x)
            except ImportError:
                auc = np.trapz(y, x)
                
            # Fit linear function f(x) = beta_0 + beta_1 * x
            x_reshaped = x.reshape(-1, 1)
            model = LinearRegression().fit(x_reshaped, y)
            
            # 3. SLOPE
            slope = model.coef_[0]
            
            # 4. POWER LAW deviation (Mean Absolute Error)
            y_pred = model.predict(x_reshaped)
            power_law_mae = mean_absolute_error(y, y_pred)

        return {
            "COMPRESSION": token_count,
            "CARDINALITY": cardinality,
            "AUC": auc,
            "SLOPE": slope,
            "POWER_LAW": power_law_mae
        }

if __name__ == "__main__":
    print("Testing Zipf Metrics...")
    
    # Simple whitespace tokenizer mock for testing
    class MockTokenizer:
        def encode(self, text):
            # A simple deterministic subword-like split
            import re
            return re.findall(r'[a-zA-Z]{1,3}|[0-9]+|\s+|[^\w\s]', text)
            
    mock_tokenizer = MockTokenizer()
    metrics_eval = TokenizerZipfMetrics(mock_tokenizer)
    
    sample_text = ("This is a sample document to test the Zipf distribution metrics. "
                  "It needs to be relatively long to get a good distribution curve, "
                  "so we will repeat some words many times. The the the the a a a a "
                  "is is is to to to Zipf Zipf metrics metrics test test test.") * 20
                  
    results = metrics_eval.compute_metrics(sample_text)
    
    print("\nZipf Metrics Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
