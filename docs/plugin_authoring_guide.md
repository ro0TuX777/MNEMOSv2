# ForgeRoot Plugin Authoring Guide for AI Developers

This guide provides step-by-step instructions for converting an existing application, script, or policy engine into a native **ForgeRoot Plugin**.

In ForgeRoot, plugins are isolated modules that define **Commands** (executable actions) and **Hooks** (governance interceptors). By converting your application into a plugin, it automatically inherits ForgeRoot's cryptographic ledgering, MCP compatibility, profile-based access control, and hash-pinned integrity.

---

## 1. Directory Structure
A ForgeRoot plugin is a self-contained directory containing a manifest and Python scripts. Create a new directory under the `plugins/` folder (e.g., `plugins/my_custom_plugin/`).

Your directory should look like this:
```text
plugins/my_custom_plugin/
├── plugin.yaml          # The manifest defining the plugin
├── __init__.py          # (Optional) Makes it a Python module
├── hooks/               # Governance evaluation logic
│   └── before_action.py
└── commands/            # Executable tasks
    └── run_action.py
```

---

## 2. Universal Design (The Adapter Pattern)
**CRITICAL:** Do NOT write your core business logic directly inside the ForgeRoot plugin files. If you do, your application will be tightly coupled to ForgeRoot and will fail in systems that do not have ForgeRoot installed.

To build a universal plugin that avoids vendor lock-in, use the **Adapter Pattern** (Clean Architecture):
1. **The Core Library:** Build your application (e.g., MNEMOS) as a standalone Python package (`pip install mnemos`). It should have zero knowledge of ForgeRoot or MCP.
2. **The ForgeRoot Wrapper:** The files in your `hooks/` and `commands/` directories should act purely as lightweight wrappers. They import your standalone library, execute it, and format the output to match ForgeRoot's expected dictionary schema.

*This guarantees your application remains usable via REST APIs, raw MCP servers, or standard Python imports outside of the ForgeRoot ecosystem.*

---

## 3. Defining the Manifest (`plugin.yaml`)
The `plugin.yaml` file is the strict contract for your plugin. It tells ForgeRoot what your plugin does, what parameters it requires, and what integrity protections it enforces.

Create `plugin.yaml` with the following structure:
```yaml
plugin:
  id: "my_custom_plugin"
  version: "1.0.0"
  description: "A description of what this plugin governs or executes."
  author: "Your Name/Team"
  integrity:
    mode: "hash_pinned" # Required for production plugins to prevent tampering

  hooks:
    - name: "before_action"
      description: "Intercepts an action before it occurs to determine if it is allowed."
      entrypoint: "hooks.before_action"
      parameters:
        type: "object"
        properties:
          target: { type: "string" }
        required: ["target"]

  commands:
    - name: "run_action"
      description: "Executes an action via CLI or MCP."
      entrypoint: "commands.run_action"
      parameters:
        type: "object"
        properties:
          target: { type: "string" }
        required: ["target"]
```

---

## 4. Implementing Hooks (Governance)
Hooks are used to intercept an agent's intent *before* execution. They are evaluated by ForgeGate.

Create `hooks/before_action.py`. Your file **must** contain an `evaluate` function that returns a dictionary containing a `decision` (`ALLOW`, `DENY`, `REQUIRE_APPROVAL`, `BLOCK`) and a `reason`.

```python
# hooks/before_action.py
from my_universal_library import core_logic # Import your standalone logic

def evaluate(context: dict, args: dict) -> dict:
    """
    Evaluates the intent against the plugin's internal logic.
    - context: Contains active profile and environment data.
    - args: The arguments provided to the hook.
    """
    target = args.get("target")

    # Call your standalone core library
    is_safe, reason = core_logic.check_safety(target)
    
    if not is_safe:
        return {
            "decision": "REQUIRE_APPROVAL",
            "reason": reason
        }
    
    return {
        "decision": "ALLOW",
        "reason": "Target is safe to process."
    }
```

---

## 5. Implementing Commands (Execution)
Commands allow your plugin to be executed directly via the ForgeRoot CLI (`forgeroot plugin run ...`) or exposed to AI agents via the MCP server.

Create `commands/run_action.py`. Your file **must** contain an `execute` function.

```python
# commands/run_action.py
from my_universal_library import core_logic # Import your standalone logic

def execute(args: dict) -> dict:
    """
    Executes the command logic.
    - args: The arguments provided to the command.
    """
    target = args.get("target")
    
    # Call your standalone core library
    result = core_logic.run(target)
    
    return {
        "status": "SUCCESS",
        "message": f"Successfully ran action against {target}. Result: {result}"
    }
```

---

## 6. Integrating with the ForgeLedger (Optional but Recommended)
If your application makes security decisions or mutates state, you should emit cryptographic evidence to the ForgeLedger.

You can import the ledger adapter directly into your hook or command:
```python
from forgeroot.adapters.forgeledger_adapter import ForgeLedgerAdapter

def evaluate(context: dict, args: dict) -> dict:
    # Logic...
    decision = "ALLOW"
    
    # Emit evidence to the append-only ledger
    ForgeLedgerAdapter.emit_evidence(
        event_type="governance_decision",
        actor=args.get("actor", "unknown"),
        context={
            "plugin": "my_custom_plugin",
            "target": args.get("target"),
            "decision": decision
        }
    )
    
    return {"decision": decision, "reason": "Authorized."}
```

---

## 7. Testing Your Plugin
Once your plugin is authored, you can test it directly using the ForgeRoot CLI.

**Test the Command:**
```bash
forgeroot plugin run my_custom_plugin run_action --target "test_file"
```

**Test the Hook (Evaluation):**
```bash
forgeroot plugin evaluate my_custom_plugin before_action --target "sensitive_file"
```

If your plugin works via the CLI, it is automatically available to MCP-compatible AI agents (like Claude Desktop or Cursor) assuming they are running under an active Profile.

## 8. Generating Integrity Hashes
If your `plugin.yaml` sets `integrity.mode: hash_pinned`, you must compute the SHA-256 hashes of your Python files and add them to `plugin.yaml` before it will execute under a strict profile.
```bash
sha256sum hooks/before_action.py
sha256sum commands/run_action.py
```
Add the `hashes` array to your `plugin.yaml` under the `integrity` block to enforce zero-trust tampering protections.
