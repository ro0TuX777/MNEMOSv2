# commands/search.py
from mnemos_sdk.client import MnemosClient
from mnemos_sdk.config import MnemosConfig

def execute(args=None, **kwargs) -> dict:
    """
    Executes a search against the MNEMOS universal library.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parsed, _ = parser.parse_known_args(args or [])
    
    query = parsed.query
    top_k = parsed.top_k
    
    # Load configuration from environment for the standalone library
    config = MnemosConfig.from_env()
    client = MnemosClient(config)
    
    # Connect and search using the standalone core logic
    if client.wait_until_ready():
        try:
            results = client.search(query, top_k=top_k)
            # Format outputs to match ForgeRoot's expected dictionary schema
            formatted_results = [
                {
                    "score": hit.score,
                    "tier": hit.tier,
                    "content": hit.engram.get("content", "")
                }
                for hit in results
            ]
            
            result = {
                "status": "SUCCESS",
                "message": f"Successfully ran search for '{query}'. Retrieved {len(results)} results.",
                "data": formatted_results
            }
            print(result)
            return result
        except Exception as e:
            result = {
                "status": "ERROR",
                "message": f"Search failed: {str(e)}"
            }
            print(result)
            return result
    else:
        result = {
            "status": "ERROR",
            "message": "MNEMOS service is not ready or unreachable."
        }
        print(result)
        return result
