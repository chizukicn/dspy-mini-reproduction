# DSPy Mini Reproduction

A minimal reproduction project for a bug in the DSPy library related to usage tracking when using streaming functionality with MLflow autologging.

## Bug Description

When using `dspy.streaming.streamify()` with MLflow's `mlflow.dspy.autolog()`, a `TypeError` occurs in the usage tracker:

```
TypeError: object of type 'int' has no len()
```

The error originates from `dspy/utils/usage_tracker.py` at line 35:
```python
if usage_entry2 is None or len(usage_entry2) == 0:
```

The issue appears to be that `usage_entry2` is expected to be a list or dict, but receives an `int` instead, causing the `len()` call to fail.

**Note**: This is a non-deterministic bug that occurs with high probability. The error may not manifest on every run, but it happens frequently enough to be reproducible with multiple executions.

## Reproduction Steps

1. Ensure you have a local MLflow server running on `http://localhost:5000`
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run the reproduction script:
   ```bash
   python main.py
   ```
   
   **Note**: Due to the non-deterministic nature of this bug, you may need to run the script multiple times to reproduce the error. The bug occurs with high probability but not on every execution.

## Expected Behavior

The script should execute successfully, streaming the ReAct module's output while logging to MLflow.

## Actual Behavior

The script crashes with a `TypeError` when the usage tracker attempts to merge usage entries.

## Environment

- Python: >=3.13
- DSPy: >=3.0.4
- MLflow: >=3.7.0

## Code Overview

The reproduction script:
1. Configures MLflow tracking and autologging
2. Sets up a DSPy LM (GPT-4o-mini)
3. Creates a ReAct module with a weather search tool
4. Wraps the ReAct module with `streamify()` and adds a stream listener
5. Executes an async query that triggers the bug

## Related Files

- `main.py`: The minimal reproduction script
- `pyproject.toml`: Project dependencies

