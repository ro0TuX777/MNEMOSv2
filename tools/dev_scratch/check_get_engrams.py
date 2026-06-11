import os
import sys

from mnemos.retrieval.lancedb_tier import LanceDBTier

tier = LanceDBTier(db_dir="data/pit11a/lance", table_name="mnemos_engrams")
tier._initialize()

# Let's get an ID first
res1 = tier._table.search().limit(1).to_list()
if res1:
    test_id = res1[0]["id"]
    print("Testing get_engrams with ID:", test_id)
    res2 = tier.get_engrams([test_id])
    print("Result:", len(res2))
    if res2:
        print("Success:", res2[0]["id"])
else:
    print("No records found in DB.")
