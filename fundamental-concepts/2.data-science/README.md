# Data Science

## What Data Science Really Is

Data science is the discipline of turning raw data into useful understanding and useful decisions.

It usually includes:

- collecting data
- cleaning data
- exploring patterns
- building features
- building simple or advanced models
- validating results
- communicating findings

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

### Practical Example

Suppose the task is customer churn prediction.

The full project includes:

- joining billing, product, and support data
- handling missing values
- defining what counts as churn
- visualizing churn by plan type and region
- training a first model
- explaining which factors matter most

That whole process is data science.

---

## 2. Data Collection and Data Quality

Data quality determines how useful the final model can be.

Common issues:

- missing values
- duplicated rows
- inconsistent formats
- bad labels
- time leakage
- sampling bias

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

## 3. Exploratory Data Analysis

Exploratory Data Analysis, or **EDA**, means understanding the dataset before making strong assumptions.

Typical questions:

- what is the distribution of each feature?
- are there missing values?
- are there suspicious outliers?
- how are features related to the target?
- is the label imbalanced?

### Python Example

```python
import pandas as pd

sales = pd.DataFrame(
    {
        "spend": [20, 55, 12, 80, 40],
        "churned": [1, 0, 1, 0, 0],
    }
)

print(sales.describe())
print(sales.groupby("churned")["spend"].mean())
```

---

## 4. Feature Engineering

A **feature** is an input used by a model.

Feature engineering means creating better inputs from raw data.

Examples:

- convert timestamp to hour or weekday
- compute purchase frequency
- normalize price values
- turn text into TF-IDF vectors or embeddings

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

## 5. Training, Validation, and Test Splits

A common mistake is evaluating on the same data used for training.

Typical split:

- training set: learn parameters
- validation set: choose settings and compare models
- test set: final unbiased evaluation

Why it matters:

- prevents overly optimistic evaluation
- helps detect overfitting

### Python Example

```python
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train, X_test)
```

---

## 6. Evaluation and Communication

Data science is not finished when the model trains.

You still need to explain:

- what was found
- what the uncertainty is
- what assumptions were made
- what actions should follow

Typical outputs:

- dashboards
- reports
- notebooks
- experiments
- recommendations to a product or business team

---

## 7. Data Science vs Machine Learning

| Topic | Data Science | Machine Learning |
| --- | --- | --- |
| Main goal | understand and communicate data | learn predictive patterns |
| Outputs | insights, experiments, dashboards | predictions, scores, clusters |
| Typical work | cleaning, exploration, reporting | training and evaluating models |

### Practical Example

- a data scientist may discover churn is higher in one region and one product tier
- an ML engineer may productionize the churn prediction model

In many teams, one person may do both.

---

## 8. End-to-End Mini Example

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

# small toy dataset
_df = pd.DataFrame(
    {
        "monthly_spend": [20, 55, 12, 80, 40, 70],
        "support_tickets": [5, 1, 7, 0, 2, 1],
        "churned": [1, 0, 1, 0, 0, 0],
    }
)

X = _df[["monthly_spend", "support_tickets"]]
y = _df["churned"]

model = LogisticRegression()
model.fit(X, y)

print("prediction:", model.predict([[35, 4]]))
print("probability:", model.predict_proba([[35, 4]]))
```

This is not a full production pipeline, but it shows the real idea:

- clean data
- create features
- train model
- generate interpretable outputs

Continue to [Machine Learning and AI](../3.ml-and-ai/README.md).

---

## 9. Types of Data You Will Actually See

Real data science work deals with many data shapes, and each one needs a different approach.

### Tabular Data

Examples:

- customer records
- transactions
- sensor summaries

Typical tools:

- pandas
- SQL
- gradient boosting
- linear models

### Time-Series Data

Examples:

- stock prices
- server metrics
- user activity over time

Main concern:

- never leak future information into the past

### Text Data

Examples:

- support tickets
- product reviews
- chat logs

Typical representations:

- bag of words
- TF-IDF
- embeddings

### Image and Audio Data

Examples:

- medical scans
- photos
- speech recordings

These usually require feature extractors or deep learning models.

---

## 10. Data Leakage and Why It Destroys Trust

**Data leakage** happens when information from the future or from the label accidentally reaches the model during training.

Common leakage patterns:

- using a feature that is created after the event you want to predict
- normalizing with statistics computed on the full dataset before the split
- mixing records from the same user into both train and test

### Practical Example

Suppose you want to predict whether a loan will default. If one feature is "sent to collections", that feature may only appear after the default process has already started. The model will look excellent offline and useless in reality.

### Python Example

```python
import pandas as pd

df = pd.DataFrame(
    {
        "event_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-07"]),
        "feature_time": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-08"]),
        "label": [0, 1, 1],
    }
)

df["leakage"] = df["feature_time"] > df["event_date"]
print(df)
```

If `leakage` is `True`, the feature is arriving after the prediction point.

---

## 11. Baselines, Experiments, and Iteration

A strong data science habit is to start simple.

Good baselines:

- mean prediction for regression
- majority class prediction for classification
- simple linear or logistic regression
- rule-based heuristic

Why this matters:

- you need a reference point
- complex models are only useful if they beat simple ones
- baselines help catch pipeline bugs

### Practical Example

If a complicated churn model barely beats a majority-class baseline, the issue may be poor features rather than model choice.

---

## 12. Communication, Visualization, and Decision-Making

A technically correct analysis is still weak if decision-makers cannot understand it.

Professional data science outputs usually include:

- one sentence for the business question
- one sentence for the result
- one sentence for uncertainty or limitations
- one sentence for the action to take

### Example

"Users with more than three unresolved support tickets are much more likely to churn. The pattern is consistent across the last two quarters. We recommend a support-priority intervention for that segment."

### Python Example

```python
import pandas as pd

report = pd.DataFrame(
    {
        "segment": ["low tickets", "high tickets"],
        "churn_rate": [0.08, 0.31],
    }
)

print(report)
print("difference:", report.loc[1, "churn_rate"] - report.loc[0, "churn_rate"])
```

---

## 13. Data Science in Production

Real systems do not stop at one notebook.

After deployment, teams usually monitor:

- data drift
- concept drift
- latency
- missing-feature rates
- prediction quality over time

### Practical Example

A fraud model may perform well at launch and then degrade as attacker behavior changes. That is concept drift.

### Python Example

```python
import numpy as np

train_mean = 32.5
recent_values = np.array([31.9, 33.2, 34.1, 39.0, 38.6])

recent_mean = recent_values.mean()
drift = recent_mean - train_mean

print("recent_mean:", recent_mean)
print("mean_shift:", drift)
```

Monitoring is part of data science maturity, not an afterthought.
