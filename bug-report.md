# 🐛 Bug Report

## What happened?

When using `dspy.streaming.streamify()` with MLflow's `mlflow.dspy.autolog()`, a `TypeError` occurs in the usage tracker:

```
TypeError: object of type 'int' has no len()
```

The error originates from `dspy/utils/usage_tracker.py` at line 35:
```python
if usage_entry2 is None or len(usage_entry2) == 0:
```

The issue appears to be that `usage_entry2` is expected to be a list or dict, but receives an `int` instead, causing the `len()` call to fail.

**Actual Behavior**: The script crashes with a `TypeError` when the usage tracker attempts to merge usage entries.

**Note**: This is a non-deterministic bug that occurs with high probability. The error may not manifest on every run, but it happens frequently enough to be reproducible with multiple executions.

## Steps to reproduce

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

### Reproduction Code

```python
import dspy
import dspy.streaming 
import asyncio
import mlflow
import mlflow.dspy
import random
mlflow.set_tracking_uri("http://localhost:5000")  # Use local MLflow server
mlflow.set_experiment("dspy-mini-reproduction")
mlflow.dspy.autolog()

lm = dspy.LM(model="gpt-4o-mini")
dspy.configure(lm=lm)

def search_weather(city: str):
    return {
        "city": city,
        "weather": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
        "temperature": random.randint(0, 40),
        "humidity": random.randint(0, 100),
        "pressure": random.randint(900, 1100),
        "wind_speed": random.randint(0, 100),
        "wind_direction": random.choice(["N", "S", "E", "W"]),
        "wind_gust": random.randint(0, 100),
        "wind_gust_direction": random.choice(["N", "S", "E", "W"]),
        "wind_gust_speed": random.randint(0, 100),
    }

react = dspy.ReAct("question -> answer", tools=[search_weather])


stream_react = dspy.streaming.streamify(react,stream_listeners=[
    dspy.streaming.StreamListener("answer")
])


async def main():
    city = random.choice(["Tokyo", "London", "Paris", "Berlin", "Rome", "Madrid", "Berlin", "Rome", "Madrid", "Berlin", "Rome", "Madrid"])
    pred = stream_react(question=f"What is the weather in {city}?")
    async for chunk in pred:
        print(chunk)

asyncio.run(main())
```

## DSPy version

>=3.0.4

## Proposed Solution

问题根源：在 `_merge_usage_entries` 方法中，当递归合并嵌套字典时，如果 `current_v` 是之前合并产生的 `int` 值（例如 `{"tokens": 10}`），而 `v` 是字典（例如 `{"tokens": {"prompt": 5}}`），代码会尝试调用 `_merge_usage_entries(current_v, v)`，但 `current_v` 是 `int` 而不是 `dict`，导致在方法开头调用 `len(usage_entry2)` 时失败。

修复方案：在方法开头添加类型检查，确保参数是 `dict` 或 `None`。如果遇到非字典类型，需要特殊处理：

```python
def _merge_usage_entries(self, usage_entry1: dict[str, Any] | None, usage_entry2: dict[str, Any] | None) -> dict[str, Any]:
    # 添加类型检查：如果参数不是 dict 或 None，需要特殊处理
    if usage_entry1 is not None and not isinstance(usage_entry1, dict):
        # 如果 usage_entry1 是 int 或其他非 dict 类型，转换为 dict
        usage_entry1 = None
    
    if usage_entry2 is not None and not isinstance(usage_entry2, dict):
        # 如果 usage_entry2 是 int 或其他非 dict 类型，转换为 dict
        usage_entry2 = None
    
    if usage_entry1 is None or len(usage_entry1) == 0:
        return dict(usage_entry2) if usage_entry2 else {}
    
    if usage_entry2 is None or len(usage_entry2) == 0:
        return dict(usage_entry1)
    
    result = dict(usage_entry2)
    
    for k, v in usage_entry1.items():
        current_v = result.get(k)
        
        # 如果两个值都是 dict，递归合并
        if isinstance(v, dict) and isinstance(current_v, dict):
            result[k] = self._merge_usage_entries(current_v, v)
        # 如果只有一个是 dict，另一个是数值，需要特殊处理
        elif isinstance(v, dict) and not isinstance(current_v, dict):
            # current_v 是 int/None，v 是 dict，直接使用 v
            result[k] = v
        elif isinstance(current_v, dict) and not isinstance(v, dict):
            # current_v 是 dict，v 是 int/None，直接使用 current_v
            result[k] = current_v
        else:
            # 两个都是数值，相加
            result[k] = (current_v or 0) + (v or 0)
    
    return result
```

或者更简洁的方案，在递归调用前确保参数类型正确：

```python
def _merge_usage_entries(self, usage_entry1: dict[str, Any] | None, usage_entry2: dict[str, Any] | None) -> dict[str, Any]:
    # 类型检查和转换：确保参数是 dict 或 None
    if usage_entry1 is not None and not isinstance(usage_entry1, dict):
        usage_entry1 = None
    if usage_entry2 is not None and not isinstance(usage_entry2, dict):
        usage_entry2 = None
    
    if usage_entry1 is None or len(usage_entry1) == 0:
        return dict(usage_entry2) if usage_entry2 else {}
    
    if usage_entry2 is None or len(usage_entry2) == 0:
        return dict(usage_entry1)
    
    result = dict(usage_entry2)
    
    for k, v in usage_entry1.items():
        current_v = result.get(k)
        
        if isinstance(v, dict) or isinstance(current_v, dict):
            # 确保两个参数都是 dict 或 None 再递归
            current_v_dict = current_v if isinstance(current_v, dict) else None
            v_dict = v if isinstance(v, dict) else None
            merged = self._merge_usage_entries(current_v_dict, v_dict)
            # 如果合并结果是空 dict，且原来有数值，保留数值
            if not merged and (isinstance(current_v, (int, float)) or isinstance(v, (int, float))):
                result[k] = (current_v or 0) + (v or 0)
            else:
                result[k] = merged
        else:
            result[k] = (current_v or 0) + (v or 0)
    
    return result
```

**推荐方案**：最简单的修复是在方法开头添加类型检查，如果参数不是 `dict` 或 `None`，就将其视为 `None` 处理：

```python
def _merge_usage_entries(self, usage_entry1: dict[str, Any] | None, usage_entry2: dict[str, Any] | None) -> dict[str, Any]:
    # 类型安全检查：确保参数是 dict 或 None
    if usage_entry1 is not None and not isinstance(usage_entry1, dict):
        usage_entry1 = None
    if usage_entry2 is not None and not isinstance(usage_entry2, dict):
        usage_entry2 = None
    
    if usage_entry1 is None or len(usage_entry1) == 0:
        return dict(usage_entry2) if usage_entry2 else {}
    
    if usage_entry2 is None or len(usage_entry2) == 0:
        return dict(usage_entry1)
    
    result = dict(usage_entry2)
    
    for k, v in usage_entry1.items():
        current_v = result.get(k)
        
        if isinstance(v, dict) or isinstance(current_v, dict):
            result[k] = self._merge_usage_entries(
                current_v if isinstance(current_v, dict) else None,
                v if isinstance(v, dict) else None
            )
        else:
            result[k] = (current_v or 0) + (v or 0)
    
    return result
```
