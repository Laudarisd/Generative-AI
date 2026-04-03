# Data Science

## What Data Science Really Is

Data science is the discipline of turning raw data into useful understanding and useful decisions.

It usually includes:

- collecting data
- cleaning data
- exploring patterns
- building simple or advanced models
- communicating results

Machine learning is one part of data science, but data science is broader.

---

## 1. A Practical Workflow

Most real projects follow a flow like this:

1. define the problem
2. gather data
3. clean and prepare data
4. explore data
5. build features
6. train a baseline model
7. evaluate and explain results
8. deploy or report findings

### Python Example

```python
import pandas as pd

df = pd.DataFrame(
    {
        "age": [25, 31, None, 40, 31],
        "country": ["KR", "US", "KR", None, "US"],
    }
)

df = df.drop_duplicates()
df["age"] = df["age"].fillna(df["age"].median())
df["country"] = df["country"].fillna("Unknown")

print(df)
```

---

## 2. Feature Engineering

A **feature** is an input used by a model.

Feature engineering means creating more useful signals from raw data.

### Python Example

```python
import pandas as pd

df = pd.DataFrame(
    {
        "visits": [10, 2, 6],
        "purchases": [2, 0, 3],
    }
)

df["conversion_rate"] = df["purchases"] / df["visits"]
print(df)
```

---

## 3. Data Science vs Machine Learning

| Topic | Data Science | Machine Learning |
| --- | --- | --- |
| Main goal | understand and communicate data | learn predictive patterns |
| Outputs | insights, experiments, dashboards | predictions, scores, clusters |
| Typical work | cleaning, exploration, reporting | training and evaluating models |

Continue to [Machine Learning and AI](../3.ml-and-ai/README.md).
