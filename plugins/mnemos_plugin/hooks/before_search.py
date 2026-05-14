# hooks/before_search.py
from forgeroot.adapters.forgeledger_adapter import ForgeLedgerAdapter

def evaluate(context: dict, args: dict) -> dict:
    """
    Evaluates the search intent against MNEMOS safety logic.
    - context: Contains active profile and environment data.
    - args: The arguments provided to the hook.
    """
    query = args.get("query", "")

    # Basic business logic example
    if "restricted" in query.lower() or "classified" in query.lower():
        decision = "REQUIRE_APPROVAL"
        reason = f"Query contains sensitive terms: '{query}'"
    else:
        decision = "ALLOW"
        reason = "Query is safe."
        
    try:
        # Emit evidence to the append-only ledger
        ForgeLedgerAdapter.emit_evidence(
            event_type="governance_decision",
            actor=args.get("actor", "unknown"),
            context={
                "plugin": "mnemos_plugin",
                "target": query,
                "decision": decision
            }
        )
    except Exception:
        pass # Optional dependency

    return {
        "decision": decision,
        "reason": reason
    }
