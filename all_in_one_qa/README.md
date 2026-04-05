# All In One Q&A - AI/ML General Concept Guide

This README is reorganized by concept domains for faster study and implementation practice.

Quick link: [Brief Q&A + Code Examples](BRIEF_QA.md)

---

## =============Foundamental=======

### Q9. 10x more features than samples

Regularize strongly, feature selection, dimensionality reduction, sparse models, and robust cross-validation.

Explanation: Regularize strongly, feature selection, dimensionality reduction, sparse models, and robust cross-validation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q3. Ablation study

Systematic removal/change of components to measure each component’s contribution.

Explanation: Systematic removal/change of components to measure each component’s contribution. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q6. Adam vs AdamW

AdamW decouples weight decay from gradient updates and usually gives better regularization behavior in modern deep learning.

Explanation: AdamW decouples weight decay from gradient updates and usually gives better regularization behavior in modern deep learning. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
prompt = f"Question: {query}\nContext: {context}"\nresponse = llm.generate(prompt)
```

### 22. Adam vs SGD

Adam converges fast and is robust early. SGD+momentum often gives stronger final generalization at scale. Choose based on convergence speed vs final quality.

Explanation: Adam converges fast and is robust early. SGD+momentum often gives stronger final generalization at scale. Choose based on convergence speed vs final quality. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q6. Align outputs with domain constraints

Use constrained prompts, tool use, retrieval, guardrails, and policy checks.

Explanation: Use constrained prompts, tool use, retrieval, guardrails, and policy checks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q24. ARIMA (sometimes mistyped as RMIA)

ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model for univariate time-series forecasting and residual-based anomaly detection.
Unlike LSTM-style deep models, ARIMA models linear temporal relationships explicitly.

It is written as `ARIMA(p, d, q)`:

1. `p` (AutoRegressive part, AR): number of lagged observations used to predict current value.
2. `d` (Integrated part, I): number of differencing operations used to make the series more stationary.
3. `q` (Moving Average part, MA): number of lagged forecast errors used to correct predictions.

Example intuition:
- `ARIMA(2,1,1)` uses 2 past values, applies first-order differencing once, and uses 1 past error term.

Explanation: ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model for univariate time-series forecasting and residual-based anomaly detection. Unlike LSTM-style deep models, ARIMA models linear temporal relationships explicitly. It is written as `ARIMA(p, d, q)`: 1. `p` (AutoRegressive part, AR): number of lagged observations used to predict current value. 2. `d` (Integrated part, I): number of differencing operations used to make the series more stationary. 3. `q` (Moving Average part, MA): number of lagged forecast errors used to correct predictions. Example intuition: - `ARIMA(2,1,1)` uses 2 past values, applies first-order differencing once, and uses 1 past error term. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Difference series to near-stationary (`d`).
2. Choose `p,q` via ACF/PACF and rolling validation.
3. Inspect residuals and backtest before deployment.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from statsmodels.tsa.arima.model import ARIMA

fit = ARIMA(series, order=(2, 1, 1)).fit()
forecast = fit.forecast(steps=7)
```

### Q20. Batching importance

Improves throughput and gradient stability; better hardware utilization.

Explanation: Improves throughput and gradient stability; better hardware utilization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q5. Catastrophic forgetting

New fine-tuning data overwrites old capabilities; mitigate with PEFT, rehearsal, balanced data.

Explanation: New fine-tuning data overwrites old capabilities; mitigate with PEFT, rehearsal, balanced data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q8. Compare models fairly

Same data splits, compute budget, tuning effort, and evaluation rules.

Explanation: Same data splits, compute budget, tuning effort, and evaluation rules. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q3. Context length vs compute tradeoff

Attention cost grows roughly quadratically with sequence length in standard transformers.

Explanation: Attention cost grows roughly quadratically with sequence length in standard transformers. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q23. Context Window Saturation

As context grows, compute and memory rise; long irrelevant context can reduce answer quality. Retrieval and context pruning help.

Explanation: As context grows, compute and memory rise; long irrelevant context can reduce answer quality. Retrieval and context pruning help. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q8. Converges but wrong predictions

Objective-metric mismatch, thresholding issues, label noise, or train-serving skew.

Explanation: Objective-metric mismatch, thresholding issues, label noise, or train-serving skew. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
model.train()
out = model(x)
loss = criterion(out, y)
loss.backward()
optimizer.step()
```

### 1. Describe your most impactful AI project

A strong example is leading a 2D-to-3D BIM generation system end-to-end. The work includes data pipeline design, annotation strategy, model architecture, loss design, deployment, and MLOps. A key challenge is geometric ambiguity (for example symmetric/square objects). Practical fixes include geometry-aware loss constraints and attention modules, which improve robustness on noisy real-world inputs.

Explanation: A strong example is leading a 2D-to-3D BIM generation system end-to-end. The work includes data pipeline design, annotation strategy, model architecture, loss design, deployment, and MLOps. A key challenge is geometric ambiguity (for example symmetric/square objects). Practical fixes include geometry-aware loss constraints and attention modules, which improve robustness on noisy real-world inputs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q7. Design experiments

Start from hypothesis, control confounders, choose meaningful metrics, predefine protocol.

Explanation: Start from hypothesis, control confounders, choose meaningful metrics, predefine protocol. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q7. Design multi-GPU training

Use DDP, distributed sampler, gradient all-reduce, and rank-aware checkpointing/logging.

Explanation: Use DDP, distributed sampler, gradient all-reduce, and rank-aware checkpointing/logging. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

### Q10. Detect/fix model bias

Measure subgroup metrics, identify bias sources, rebalance data/objective, and monitor fairness continuously.

Explanation: Measure subgroup metrics, identify bias sources, rebalance data/objective, and monitor fairness continuously. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q14. Early stopping

Stop training when validation performance stops improving to prevent overfitting.

Explanation: Stop training when validation performance stops improving to prevent overfitting. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Track best validation metric each epoch.
2. Stop when no improvement for `patience` epochs.
3. Restore and export the best checkpoint.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

cov = np.cov(x, y)[0,1]; corr = np.corrcoef(x, y)[0,1]
print(cov, corr)
```

### Q4. Evaluate LLM quality

Task metrics + human eval + factuality/safety/latency/cost evaluations.

Explanation: Task metrics + human eval + factuality/safety/latency/cost evaluations. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q2. Evaluate new method

Check assumptions, baseline fairness, ablations, statistical significance, and real-world constraints.

Explanation: Check assumptions, baseline fairness, ablations, statistical significance, and real-world constraints. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q8. Evaluate time-series models

Use walk-forward backtesting and horizon-aware metrics; avoid random splits.

Explanation: Use walk-forward backtesting and horizon-aware metrics; avoid random splits. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Build lag/rolling/calendar features with strict temporal ordering.
2. Use walk-forward validation rather than random splits.
3. Evaluate per horizon and monitor drift after deployment.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
train, test = series[:-h], series[-h:]
model.fit(train)
pred = model.predict(h)
```

### Q9. Explainable AI for regulator

Use interpretable models where possible, local/global explanations, documentation, and audit trails.

Explanation: Use interpretable models where possible, local/global explanations, documentation, and audit trails. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q7. Exploding gradient

Gradients grow excessively, causing instability.

Explanation: Gradients grow excessively, causing instability. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Q16. Exposure Bias

Mismatch between training (teacher forcing) and inference (model-generated history), causing compounding generation errors.

Explanation: Mismatch between training (teacher forcing) and inference (model-generated history), causing compounding generation errors. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
docs = retriever.get_relevant_documents(query)\ncontext = "\n".join(d.page_content for d in docs[:3])\nanswer = llm.generate(context)
```

### Q6. Forecasting horizon

Future time span being predicted.

Explanation: Future time span being predicted. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Build lag/rolling/calendar features with strict temporal ordering.
2. Use walk-forward validation rather than random splits.
3. Evaluate per horizon and monitor drift after deployment.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
horizon = 24
y_hat = model.predict(X_last, steps=horizon)
```

### Q28. FSDP / ZeRO (why needed)

Shard parameters/gradients/optimizer states to train models that do not fit on one GPU.

Explanation: Shard parameters/gradients/optimizer states to train models that do not fit on one GPU. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Launch one process per GPU and shard data with distributed sampler.
2. Sync gradients using all-reduce (or shard states with FSDP/ZeRO).
3. Save rank-safe checkpoints and aggregate metrics across workers.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

### Q10. Good offline, bad production

Data drift, schema mismatch, missing features, latency constraints, feedback loops, monitoring gaps.

Explanation: Data drift, schema mismatch, missing features, latency constraints, feedback loops, monitoring gaps. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q10. Gradient clipping

Cap gradient norm/value to stabilize training and avoid exploding updates.

Explanation: Cap gradient norm/value to stabilize training and avoid exploding updates. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Backpropagate normally first.
2. Clip before optimizer step (`clip_grad_norm_` or value clip).
3. Track clipping frequency and tune LR/max_norm accordingly.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(rmse)
```

### Q12. Gradient Clipping: by value vs by norm

- By value clips each gradient element independently.
- By norm rescales full gradient vector to max norm.
Norm clipping is usually preferred for deep sequence models.

Explanation: - By value clips each gradient element independently. - By norm rescales full gradient vector to max norm. Norm clipping is usually preferred for deep sequence models. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Backpropagate normally first.
2. Clip before optimizer step (`clip_grad_norm_` or value clip).
3. Track clipping frequency and tune LR/max_norm accordingly.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
tokens = tokenizer(query, return_tensors="pt")
out = model.generate(**tokens, max_new_tokens=64)
```

### 23. Gradient explosion/vanishing

Exploding gradients cause unstable updates; vanishing gradients block learning in early layers. Use clipping, initialization, residuals, gating (LSTM/GRU), normalization.

Explanation: Exploding gradients cause unstable updates; vanishing gradients block learning in early layers. Use clipping, initialization, residuals, gating (LSTM/GRU), normalization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q5. Gradient norms spike

Inspect recent batches/outliers, reduce LR, clip gradients, stabilize architecture/loss.

Explanation: Inspect recent batches/outliers, reduce LR, clip gradients, stabilize architecture/loss. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
for xb, yb in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(xb), yb)
    loss.backward(); optimizer.step()
```

### Q18. Handle large datasets

Sharding, streaming, memory mapping, prefetching, distributed sampling, feature stores.

Explanation: Sharding, streaming, memory mapping, prefetching, distributed sampling, feature stores. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### 5. How do you approach time-series forecasting?

Analyze trend/seasonality/autocorrelation; build lag/rolling/calendar features; use time-aware splits; choose model class (statistical, tree-based, RNN/Transformer/ESN); evaluate with horizon-aware metrics (MAE/RMSE/MAPE/sMAPE) and rolling backtests.

Explanation: Analyze trend/seasonality/autocorrelation; build lag/rolling/calendar features; use time-aware splits; choose model class (statistical, tree-based, RNN/Transformer/ESN); evaluate with horizon-aware metrics (MAE/RMSE/MAPE/sMAPE) and rolling backtests. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### 9. How do you choose evaluation metrics?

Choose metrics based on problem type and error cost. For imbalance, precision/recall/F1/PR-AUC are often better than accuracy. For regression/forecasting, MAE/RMSE/MAPE depending on sensitivity to outliers and scale.

Explanation: Choose metrics based on problem type and error cost. For imbalance, precision/recall/F1/PR-AUC are often better than accuracy. For regression/forecasting, MAE/RMSE/MAPE depending on sensitivity to outliers and scale. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q25. How do you choose retraining cadence for seasonal systems?
Answer: Align retraining with seasonal cycle length, drift speed, and operational risk tolerance.

Example: Weekly retraining may be needed in fast-changing demand systems.

### 2. How do you convert a real-world problem into an AI problem?

Start with domain understanding and objective definition. Translate into ML formulation (classification/regression/forecasting), define input-output contract, constraints (latency, cost, interpretability), and success metrics tied to business impact. Then design data, model, evaluation, and deployment plan.

Explanation: Start with domain understanding and objective definition. Translate into ML formulation (classification/regression/forecasting), define input-output contract, constraints (latency, cost, interpretability), and success metrics tied to business impact. Then design data, model, evaluation, and deployment plan. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q14. How do you design guardrails for tool-calling agents?
Answer: Add allowlists, schema validation, policy checks, and confirmation gates for risky actions.

Example: Never allow direct write/delete actions without explicit approval.

### Q16. How do you detect silent model degradation with no labels?
Answer: Use proxy signals: drift, confidence shift, latency anomalies, and downstream business KPI changes.

Example: Stable latency but dropping conversion can indicate silent quality decay.

### Q15. How do you do cost-aware LLM routing across model sizes?
Answer: Route easy queries to smaller models and escalate uncertain/high-risk cases to larger models.

Example: FAQ requests go to mini model; legal-risk questions go to flagship model.

### Q22. How do you estimate prediction uncertainty for maintenance decisions?
Answer: Use ensembles, quantile models, or Bayesian approximations and trigger actions from risk-aware intervals.

Example: Schedule inspection when upper-risk bound crosses safety threshold.

### Q12. How do you evaluate retrieval quality separately from generation quality?
Answer: Measure retrieval metrics first (`Recall@k`, `MRR`), then evaluate answer quality conditioned on retrieved context.

Example: Good generation cannot fix consistently poor retrieval.

### Q1. How do you pick batch size under fixed GPU memory?
Answer: Start from the largest stable batch that avoids out-of-memory and keeps GPU utilization high, then validate accuracy/latency tradeoff.

Example: If `batch_size=128` OOMs, try `64` and recover effective batch with accumulation.

### 8. How ensure model reliability in production?

Use strong pre-deployment validation (edge cases, stress tests) and post-deployment monitoring (drift, quality, latency, failures). Add alerts, rollback, retraining triggers, and runbooks.

Explanation: Use strong pre-deployment validation (edge cases, stress tests) and post-deployment monitoring (drift, quality, latency, failures). Add alerts, rollback, retraining triggers, and runbooks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q5. How gradient clipping stabilizes exploding gradients

It bounds update magnitude so recurrent/deep chains cannot produce destructive parameter jumps.

Explanation: It bounds update magnitude so recurrent/deep chains cannot produce destructive parameter jumps. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Backpropagate normally first.
2. Clip before optimizer step (`clip_grad_norm_` or value clip).
3. Track clipping frequency and tune LR/max_norm accordingly.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import torch

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Q5. Huge data but poor performance

Likely data quality, objective mismatch, feature issues, or leakage/shift; scale alone does not fix bad signal.

Explanation: Likely data quality, objective mismatch, feature issues, or leakage/shift; scale alone does not fix bad signal. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q2. Integrate AI into engineering systems

Map use-case to workflow, ensure data interfaces, establish reliability and override/fallback mechanisms.

Explanation: Map use-case to workflow, ensure data interfaces, establish reliability and override/fallback mechanisms. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### 16. Integrating domain knowledge

Inject domain constraints into features, architecture, loss terms, priors, and post-processing rules. Hybrid AI + physics/simulation models often improve reliability and interpretability.

Explanation: Inject domain constraints into features, architecture, loss terms, priors, and post-processing rules. Hybrid AI + physics/simulation models often improve reliability and interpretability. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q34. Isolation Forest vs LOF vs OC-SVM (quick comparison)

Isolation Forest scales well and isolates anomalies by random partitioning. LOF is local-density sensitive. OC-SVM can model nonlinear boundaries but is sensitive to kernel/scale choices.

Explanation: Isolation Forest scales well and isolates anomalies by random partitioning. LOF is local-density sensitive. OC-SVM can model nonlinear boundaries but is sensitive to kernel/scale choices. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(kernel="rbf", nu=0.05).fit(X_train_normal)
y_pred = clf.predict(X_test)
```

### Q14. Label Smoothing

Replace hard one-hot targets with softened targets to improve calibration and reduce overconfidence.

Explanation: Replace hard one-hot targets with softened targets to improve calibration and reduce overconfidence. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
messages = [{"role":"system","content":"Answer with grounded facts."},{"role":"user","content":query}]
resp = llm.chat(messages)
```

### Q13. Learning rate scheduling

Vary LR over training (step, cosine, warmup, one-cycle) for speed and stability.

Explanation: Vary LR over training (step, cosine, warmup, one-cycle) for speed and stability. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q13. Learning Rate Warmup

Start with a small LR and gradually increase early in training to avoid unstable updates, especially in Transformers.

Explanation: Start with a small LR and gradually increase early in training to avoid unstable updates, especially in Transformers. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
messages = [{"role":"system","content":"Answer with grounded facts."},{"role":"user","content":query}]
resp = llm.chat(messages)
```

### 17. Limitations of deep learning

Large data demand, high compute cost, lower interpretability, and fragility under distribution shift. Mitigate via model compression, better data curation, uncertainty estimation, and explainability tools.

Explanation: Large data demand, high compute cost, lower interpretability, and fragility under distribution shift. Mitigate via model compression, better data curation, uncertainty estimation, and explainability tools. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### 13. Limited labeled data: what do you do?

Use transfer learning, semi-supervised learning (pseudo-labeling), self-supervised pretraining, augmentation, weak supervision, and active learning for highest-value labeling.

Explanation: Use transfer learning, semi-supervised learning (pseudo-labeling), self-supervised pretraining, augmentation, weak supervision, and active learning for highest-value labeling. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q31. Local Outlier Factor (LOF)

LOF compares local density of a sample to that of neighbors. Lower relative density implies higher outlierness.

Explanation: LOF compares local density of a sample to that of neighbors. Lower relative density implies higher outlierness. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_train_normal)
score = -lof.score_samples(X_test)
```

### Q3. Loss oscillates heavily

Likely LR too high, bad normalization, noisy batches, or unstable objective. Use lower LR, scheduler, gradient clipping, larger batch.

Explanation: Likely LR too high, bad normalization, noisy batches, or unstable objective. Use lower LR, scheduler, gradient clipping, larger batch. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### 20. Low GPU utilization debugging

Profile first. Usually data pipeline bottleneck: tune `num_workers`, `pin_memory`, prefetch, serialization format, CPU transforms, and batch size. Use mixed precision where possible.

Explanation: Profile first. Usually data pipeline bottleneck: tune `num_workers`, `pin_memory`, prefetch, serialization format, CPU transforms, and batch size. Use mixed precision where possible. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q1. Model outputs NaNs: step-by-step

Check data/labels, isolate first NaN layer, lower LR, inspect gradient norms, verify numerically unstable ops (`log`, division), enable anomaly detection, and test mixed-precision settings.

Explanation: Check data/labels, isolate first NaN layer, lower LR, inspect gradient norms, verify numerically unstable ops (`log`, division), enable anomaly detection, and test mixed-precision settings. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q27. Model Parallelism vs Data Parallelism

- Data parallelism splits data across replicas.
- Model parallelism splits model across devices.
Large LLMs often use both.

Explanation: - Data parallelism splits data across replicas. - Model parallelism splits model across devices. Large LLMs often use both. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
x = layer1(x.to("cuda:0"))
x = layer2(x.to("cuda:1"))
# data parallel replicates full model across devices and splits batches
```

### Q37. Modular adaptation methods (foundation-model context)

A practical approach is frozen pretrained backbone + small task-specific adapter head for quick domain adaptation and robust deployment updates.

Explanation: A practical approach is frozen pretrained backbone + small task-specific adapter head for quick domain adaptation and robust deployment updates. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q4. No labeled data

Use self-supervised/unsupervised methods, weak supervision, synthetic labels, and active learning.

Explanation: Use self-supervised/unsupervised methods, weak supervision, synthetic labels, and active learning. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### 28. Non-stationary time-series

Use differencing/transformations, rolling retraining, adaptive windows, and online monitoring for concept drift.

Explanation: Use differencing/transformations, rolling retraining, adaptive windows, and online monitoring for concept drift. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q9. Novelty in research

New idea, new evidence, or new capability beyond existing state of the art.

Explanation: New idea, new evidence, or new capability beyond existing state of the art. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q32. One-Class SVM (OC-SVM)

OC-SVM learns a boundary around normal samples in feature space; points outside are marked anomalies.

Explanation: OC-SVM learns a boundary around normal samples in feature space; points outside are marked anomalies. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from sklearn.svm import OneClassSVM

clf = OneClassSVM(kernel="rbf", nu=0.05).fit(X_train_normal)
y_pred = clf.predict(X_test)
```

### Q17. Optimize slow training pipeline

Profile data + compute + communication; remove bottlenecks one by one.

Explanation: Profile data + compute + communication; remove bottlenecks one by one. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q5. Optimizer vs Activation Function

- Optimizer decides how parameters are updated (SGD, AdamW).
- Activation decides nonlinear transformation inside the network (ReLU, GELU, SiLU).

Rule of thumb:
- Optimizer affects learning dynamics and convergence.
- Activation affects representational power and gradient flow.

Explanation: - Optimizer decides how parameters are updated (SGD, AdamW). - Activation decides nonlinear transformation inside the network (ReLU, GELU, SiLU). Rule of thumb: - Optimizer affects learning dynamics and convergence. - Activation affects representational power and gradient flow. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
prompt = f"Question: {query}\nContext: {context}"\nresponse = llm.generate(prompt)
```

### Q1. Read papers efficiently

Read abstract/figures/conclusion first, then method and experiments with focused notes.

Explanation: Read abstract/figures/conclusion first, then method and experiments with focused notes. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q4. Reproducibility

Ability to replicate results using provided code/data/settings/seeds.

Explanation: Ability to replicate results using provided code/data/settings/seeds. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q10. Research contribution

Clear problem framing, measurable improvement, and transparent analysis of tradeoffs.

Explanation: Clear problem framing, measurable improvement, and transparent analysis of tradeoffs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q10. Residual connection

Skip connection easing optimization of deep networks.

Explanation: Skip connection easing optimization of deep networks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

### Q13. Robustness in harsh environments

Train on diverse conditions, stress test extensively, and include fallback/alert logic.

Explanation: Train on diverse conditions, stress test extensively, and include fallback/alert logic. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q9. Safety concerns

False negatives in critical events, automation bias, cyber risks, bad feedback loops, and weak fail-safe design.

Explanation: False negatives in critical events, automation bias, cyber risks, bad feedback loops, and weak fail-safe design. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q3. Seasonality

Recurring periodic patterns.

Explanation: Recurring periodic patterns. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

### Q3. Simulation + real data

Pretrain on simulation, fine-tune/calibrate on real data, and domain-adapt carefully.

Explanation: Pretrain on simulation, fine-tune/calibrate on real data, and domain-adapt carefully. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q7. Sliding window

Transform sequential data into supervised samples with rolling input windows.

Explanation: Transform sequential data into supervised samples with rolling input windows. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Build lag/rolling/calendar features with strict temporal ordering.
2. Use walk-forward validation rather than random splits.
3. Evaluate per horizon and monitor drift after deployment.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
X, y_out = [], []
for i in range(window, len(series)):
    X.append(series[i-window:i])
    y_out.append(series[i])
```

### Q1. Stationarity

Statistical properties remain stable over time.

Explanation: Statistical properties remain stable over time. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Build lag/rolling/calendar features with strict temporal ordering.
2. Use walk-forward validation rather than random splits.
3. Evaluate per horizon and monitor drift after deployment.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
import pandas as pd\n\ndf["lag_1"] = df["y"].shift(1)\ndf["lag_7"] = df["y"].shift(7)
```

### Q8. Sudden data distribution change

Trigger drift alerts, switch to safe mode, retrain/recalibrate quickly, and monitor recovery.

Explanation: Trigger drift alerts, switch to safe mode, retrain/recalibrate quickly, and monitor recovery. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q3. System causes financial loss

Stabilize system first (rollback/disable), communicate impact, perform RCA, and add controls.

Explanation: Stabilize system first (rollback/disable), communicate impact, perform RCA, and add controls. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q15. Teacher Forcing

Train decoder by feeding ground-truth previous token; speeds convergence but can create train-test mismatch.

Explanation: Train decoder by feeding ground-truth previous token; speeds convergence but can create train-test mismatch. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
tokens = tokenizer(query, return_tensors="pt")
out = model.generate(**tokens, max_new_tokens=64)
```

### 26. Train loss down, validation loss up

Classic overfitting. Add regularization, better validation, early stopping, simpler model, or more representative data.

Explanation: Classic overfitting. Add regularization, better validation, early stopping, simpler model, or more representative data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q4. Trained long but random performance

Possible label mismatch, bug in preprocessing, leakage in validation logic, incorrect target mapping, or frozen gradients.

Explanation: Possible label mismatch, bug in preprocessing, leakage in validation logic, incorrect target mapping, or frozen gradients. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q2. Training stable but very slow

Profile data pipeline, GPU kernels, communication; optimize batching, mixed precision, dataloader, kernels, and distributed setup.

Explanation: Profile data pipeline, GPU kernels, communication; optimize batching, mixed precision, dataloader, kernels, and distributed setup. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q5. Turn research into product

Simplify method, improve robustness, define SLAs, and build monitoring/deployment path.

Explanation: Simplify method, improve robustness, define SLAs, and build monitoring/deployment path. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q7. Validate model in production

Shadow mode, canary rollout, KPI monitoring, drift detection, and rollback plans.

Explanation: Shadow mode, canary rollout, KPI monitoring, drift detection, and rollback plans. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q6. Validation metric fluctuates heavily

High variance data/small validation set/distribution shift. Increase validation size, smooth reporting, use repeated runs.

Explanation: High variance data/small validation set/distribution shift. Increase validation size, smooth reporting, use repeated runs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q6. Vanishing gradient

Gradients shrink through depth/time, slowing learning.

Explanation: Gradients shrink through depth/time, slowing learning. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### Q4. What are signs that warmup is too short or too long?
Answer: Too short causes early instability/spikes; too long slows convergence and wastes steps.

Example: If loss explodes in first 200 steps, increase warmup ratio.

### 6. What is an Echo State Network (ESN)?

ESN is reservoir computing: recurrent reservoir weights are fixed, only readout is trained. It captures temporal dynamics with very cheap training and can be effective in low-latency time-series setups.

Explanation: ESN is reservoir computing: recurrent reservoir weights are fixed, only readout is trained. It captures temporal dynamics with very cheap training and can be effective in low-latency time-series setups. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q36. What is cosine similarity and when should we use it?

Cosine similarity measures the angle-based similarity between two vectors, independent of their absolute magnitude.

Explanation: It is widely used for embeddings (text/image/audio) because direction in embedding space often captures semantics better than raw vector length.

How to do it (practical):
1. Normalize vectors (or use APIs that normalize internally).
2. Compute cosine similarity `sim(a,b) = (a.b) / (||a|| ||b||)`.
3. Use it for retrieval/ranking by selecting top-k most similar vectors.

Code:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

a = np.array([[0.1, 0.3, 0.9]])
b = np.array([[0.2, 0.25, 0.88]])
c = np.array([[0.9, 0.1, 0.0]])

print("numpy cosine(a,b):", float((a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))))
print("sklearn cosine(a,b):", cosine_similarity(a, b)[0, 0])
print("sklearn cosine(a,c):", cosine_similarity(a, c)[0, 0])
```

### 12. What is regularization?

Techniques that reduce overfitting by constraining model complexity: L1/L2 penalties, dropout, early stopping, augmentation, and parameter sharing. L1 (`|w|`) promotes sparsity and can push some weights exactly to zero (feature selection effect). L2 (`w^2`) usually keeps weights non-zero but reduces their magnitude smoothly.

Explanation: Techniques that reduce overfitting by constraining model complexity: L1/L2 penalties, dropout, early stopping, augmentation, and parameter sharing. L1 (`|w|`) promotes sparsity and can push some weights exactly to zero (feature selection effect). L2 (`w^2`) usually keeps weights non-zero but reduces their magnitude smoothly. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q6. What makes research impactful

Novelty + strong evidence + reproducibility + practical relevance.

Explanation: Novelty + strong evidence + reproducibility + practical relevance. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: An ablation shows 70% of the gain came from data cleaning, not architecture changes.

### Q18. What should be in a model card for regulated environments?
Answer: Data scope, assumptions, subgroup metrics, risks, limitations, and approved use boundaries.

Example: Include explicit “not-for-use” conditions and escalation policy.

### Q25. When ARIMA is useful vs not useful

Useful for structured linear time-series with moderate data. Less suitable for highly nonlinear multivariate systems without feature engineering.

Explanation: Useful for structured linear time-series with moderate data. Less suitable for highly nonlinear multivariate systems without feature engineering. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Difference series to near-stationary (`d`).
2. Choose `p,q` via ACF/PACF and rolling validation.
3. Inspect residuals and backtest before deployment.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from statsmodels.tsa.arima.model import ARIMA

fit = ARIMA(series, order=(2, 1, 1)).fit()
forecast = fit.forecast(steps=7)
```

### 7. When choose a simpler model over a complex one?

When constraints are strict (latency, memory, explainability, maintainability) and simple models already meet target KPIs. Prefer simplest model that meets requirements with stable generalization.

Explanation: When constraints are strict (latency, memory, explainability, maintainability) and simple models already meet target KPIs. Prefer simplest model that meets requirements with stable generalization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q2. When should you use cosine scheduler vs one-cycle scheduler?
Answer: Cosine is great for steady long training; one-cycle is useful when you want fast convergence in limited epochs.

Example: For quick fine-tuning jobs, one-cycle often reaches target sooner.

### Q7. When to use GELU vs ReLU?
Answer: GELU is common in Transformers; ReLU is simpler and often sufficient in many MLP/CNN settings.

Example: LLM blocks usually default to GELU/SwiGLU variants.

### Q6. Why do residual connections help optimization depth?
Answer: They preserve gradient flow and make deep stacks easier to optimize.

Example: A 48-layer network converges with residuals but stalls without them.

### Q2. Why L2 shrinks weights but not zero

L2 applies continuous proportional shrinkage; unlike L1, it does not create sharp sparsity-inducing corners at zero. L1 can drive coefficients exactly to zero due to its absolute-value penalty, while L2 mostly reduces coefficient magnitudes without exact sparsity.

Explanation: L2 applies continuous proportional shrinkage; unlike L1, it does not create sharp sparsity-inducing corners at zero. L1 can drive coefficients exactly to zero due to its absolute-value penalty, while L2 mostly reduces coefficient magnitudes without exact sparsity. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Add regularized objective: `L=L_task+lambda1||w||_1+lambda2||w||_2^2`.
2. Use L1 for sparsity (some coefficients become zero).
3. Use L2/weight decay for smooth magnitude shrinkage and stability.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

w = np.array([0.03, -0.8, 1.5])
lam, lr = 0.1, 0.1
w_l1 = np.sign(w) * np.maximum(np.abs(w) - lam, 0.0)
w_l2 = w * (1 - 2 * lr * lam)
print(w_l1, w_l2)
```

### Q11. Why normalization helps optimization

Improves conditioning, aligns feature scales, gives more stable gradient magnitudes.

Explanation: Improves conditioning, aligns feature scales, gives more stable gradient magnitudes. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q4. Why normalization improves convergence mathematically

It reduces anisotropy of curvature (better condition number), so gradient steps are more uniformly effective.

Explanation: It reduces anisotropy of curvature (better condition number), so gradient steps are more uniformly effective. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

mae = np.mean(np.abs(y_true - y_pred))
print(mae)
```

### 18. Working with domain experts

Co-define goals, maintain shared vocabulary, translate ML outputs into domain terms, iterate through feedback loops, and align on measurable operational outcomes.

Explanation: Co-define goals, maintain shared vocabulary, translate ML outputs into domain terms, iterate through feedback loops, and align on measurable operational outcomes. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q6. Works in lab but fails in field

Domain gap, noisy sensors, unseen operating regimes, and fragile assumptions.

Explanation: Domain gap, noisy sensors, unseen operating regimes, and fragile assumptions. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

## ====================ML==========

### Q1. Classification vs regression

Classification predicts discrete classes; regression predicts continuous values.

Explanation: Classification predicts discrete classes; regression predicts continuous values. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
from sklearn.linear_model import LogisticRegression, LinearRegression

clf = LogisticRegression().fit(X_cls, y_cls)
reg = LinearRegression().fit(X_reg, y_reg)
```

### Q11. Curse of dimensionality

High-dimensional spaces become sparse; distance metrics degrade; data needs grow rapidly.

Explanation: High-dimensional spaces become sparse; distance metrics degrade; data needs grow rapidly. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

### Q8. Data leakage

Any information from validation/test/future leaking into training, causing overly optimistic metrics.

Explanation: Any information from validation/test/future leaking into training, causing overly optimistic metrics. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

### Q7. Diagnose underfitting vs overfitting from logs

Underfitting: both train/val poor. Overfitting: train good, val poor with widening gap.

Explanation: Underfitting: both train/val poor. Overfitting: train good, val poor with widening gap. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q9. Feature scaling importance

Improves optimization stability/speed and prevents large-scale features from dominating.

Explanation: Improves optimization stability/speed and prevents large-scale features from dominating. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

### Q1. L2-regularized linear regression update rule

For loss `J(w)= (1/N)||Xw-y||^2 + lambda||w||^2`, gradient is `(2/N)X^T(Xw-y)+2lambda w`; update: `w <- w - eta * grad`.

Explanation: For loss `J(w)= (1/N)||Xw-y||^2 + lambda||w||^2`, gradient is `(2/N)X^T(Xw-y)+2lambda w`; update: `w <- w - eta * grad`. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Add regularized objective: `L=L_task+lambda1||w||_1+lambda2||w||_2^2`.
2. Use L1 for sparsity (some coefficients become zero).
3. Use L2/weight decay for smooth magnitude shrinkage and stability.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q1. Mixed-precision training loop (PyTorch)

Use `torch.cuda.amp.autocast()` and `GradScaler` around forward/loss/backward/step/update.

Explanation: Use `torch.cuda.amp.autocast()` and `GradScaler` around forward/loss/backward/step/update. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Choose metric by business error cost and class balance.
2. Tune decision threshold on validation data only.
3. Report confusion matrix + calibration for operational decisions.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

Code:
```python
from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
```

### Q10. Normalization vs standardization

Normalization scales to fixed range (often [0,1]); standardization centers mean 0 and std 1.

Explanation: Normalization scales to fixed range (often [0,1]); standardization centers mean 0 and std 1. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

### Q6. Precision vs recall vs F1

Precision: correctness of positive predictions. Recall: coverage of actual positives. F1: harmonic mean balancing both.

Explanation: Precision: correctness of positive predictions. Recall: coverage of actual positives. F1: harmonic mean balancing both. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Choose metric by business error cost and class balance.
2. Tune decision threshold on validation data only.
3. Report confusion matrix + calibration for operational decisions.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
from sklearn.metrics import precision_score, recall_score, f1_score

print(precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
```

### Q7. ROC-AUC

Area under ROC curve; ranking quality across thresholds.

Explanation: Area under ROC curve; ranking quality across thresholds. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Choose metric by business error cost and class balance.
2. Tune decision threshold on validation data only.
3. Report confusion matrix + calibration for operational decisions.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_prob)
print(auc)
```

### Q5. What is a loss function and how choose it?

A scalar objective measuring prediction error. Choose based on task semantics and error cost (CE for classification, MAE/RMSE/Huber for regression).

Explanation: A scalar objective measuring prediction error. Choose based on task semantics and error cost (CE for classification, MAE/RMSE/Huber for regression). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
import torch.nn as nn

ce = nn.CrossEntropyLoss()
huber = nn.HuberLoss()
```

### Q5A. What is cross-entropy loss and how do we calculate it?

Cross-entropy measures how well predicted class probabilities match true labels. For one sample with true class `y`, loss is `-log(p_y)`.

Explanation: In classification, cross-entropy strongly penalizes confident wrong predictions, which usually gives better gradients than MSE for probability outputs.

How to do it (practical):
1. Convert logits to probabilities with softmax.
2. Pick probability of the true class.
3. Compute `-log(true_class_probability)`.
4. Average across batch.

Code:
```python
import numpy as np
import torch
import torch.nn.functional as F

# manual single-sample cross-entropy
logits_np = np.array([2.0, 0.5, -1.0])
probs_np = np.exp(logits_np) / np.exp(logits_np).sum()
true_class = 0
ce_manual = -np.log(probs_np[true_class] + 1e-12)
print("manual CE:", ce_manual)

# PyTorch batch cross-entropy (expects raw logits, not softmaxed probs)
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
targets = torch.tensor([0, 1])
ce_torch = F.cross_entropy(logits, targets)
print("torch CE:", ce_torch.item())
```

### Q2. What is cross-validation?

Repeated train/validation splits (for example k-fold) to estimate generalization more reliably.

Explanation: Repeated train/validation splits (for example k-fold) to estimate generalization more reliably. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Split into k folds.
2. Train on k-1 folds and validate on 1 fold, rotating all folds.
3. Report mean/std across folds to estimate variance.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring="f1")
print(scores.mean(), scores.std())
```

### Q3. What is gradient descent?

Iterative optimization updating parameters opposite gradient direction to minimize loss.

Explanation: Iterative optimization updating parameters opposite gradient direction to minimize loss. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
w, lr = 0.0, 1e-2
for _ in range(100):
    grad = dloss_dw(w)
    w -= lr * grad
```

### Q4. What is stochastic gradient descent?

Gradient descent using mini-batches; faster and noisier updates that often improve generalization.

Explanation: Gradient descent using mini-batches; faster and noisier updates that often improve generalization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: In a fraud dataset with only 2% positives, you prefer PR-AUC and F1 over raw accuracy.

Code:
```python
import torch

opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
opt.zero_grad(set_to_none=True); loss = criterion(model(x), y); loss.backward(); opt.step()
```

### 4. What would you do if your model overfits?

Check leakage and split correctness first. Then apply regularization, simplify architecture, early stopping, augmentation, and better feature engineering. Use cross-validation and monitor train/validation gap.

Explanation: Check leakage and split correctness first. Then apply regularization, simplify architecture, early stopping, augmentation, and better feature engineering. Use cross-validation and monitor train/validation gap. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

## =============Statistics==========

### Q2. Autocorrelation

Correlation of a series with lagged versions of itself.

Explanation: Correlation of a series with lagged versions of itself. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Build lag/rolling/calendar features with strict temporal ordering.
2. Use walk-forward validation rather than random splits.
3. Evaluate per horizon and monitor drift after deployment.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
import pandas as pd

lag_7_corr = pd.Series(series).autocorr(lag=7)
print(lag_7_corr)
```

### Q3. Bayesian inference

Update prior beliefs with observed data to obtain posterior distribution.

Explanation: Update prior beliefs with observed data to obtain posterior distribution. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q8. Bias-variance decomposition

Expected test error = irreducible noise + bias^2 + variance (for squared loss setting).

Explanation: Expected test error = irreducible noise + bias^2 + variance (for squared loss setting). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

### Q15. Calibration in ML

Alignment between predicted probabilities and actual event frequencies.

Explanation: Alignment between predicted probabilities and actual event frequencies. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from scipy.stats import ttest_ind

stat, p = ttest_ind(a, b, equal_var=False)
print(p)
```

### Q30. Calibration vs Accuracy

A model can be accurate but poorly calibrated; decision systems often need both.

Explanation: A model can be accurate but poorly calibrated; decision systems often need both. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from sklearn.calibration import calibration_curve

frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
```

### Q8. Convex vs non-convex optimization

Convex has one global minimum structure; non-convex can have many local minima/saddles.

Explanation: Convex has one global minimum structure; non-convex can have many local minima/saddles. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q5. Covariance vs correlation

Covariance measures joint variation (scale-dependent). Correlation is normalized covariance in [-1,1].

Explanation: Covariance measures joint variation (scale-dependent). Correlation is normalized covariance in [-1,1]. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q6. Eigenvalues and ESN stability

Reservoir dynamics remain stable when effective spectral radius is controlled (typically < 1 in many settings).

Explanation: Reservoir dynamics remain stable when effective spectral radius is controlled (typically < 1 in many settings). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

mu, var = np.mean(x), np.var(x)
print(mu, var)
```

### Q4. Expectation and variance

Expectation is average value; variance measures spread around expectation.

Explanation: Expectation is average value; variance measures spread around expectation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q9. Hessian matrix

Second-derivative matrix describing local curvature; helps understand conditioning and step behavior.

Explanation: Second-derivative matrix describing local curvature; helps understand conditioning and step behavior. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

cov = np.cov(x, y)[0,1]; corr = np.corrcoef(x, y)[0,1]
print(cov, corr)
```

### Q7. Hypothesis testing

Framework to assess evidence against null via test statistic, p-value, and significance threshold.

Explanation: Framework to assess evidence against null via test statistic, p-value, and significance threshold. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q3. Ill-conditioned Hessian impact

Optimization zig-zags and converges slowly; sensitive to LR. Fix with normalization, preconditioning, adaptive optimizers.

Explanation: Optimization zig-zags and converges slowly; sensitive to LR. Fix with normalization, preconditioning, adaptive optimizers. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(rmse)
```

### Q10. KL divergence and usage

Measure of distribution mismatch; used in VAEs, distillation, calibration, and drift comparison.

Explanation: Measure of distribution mismatch; used in VAEs, distillation, calibration, and drift comparison. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q2. Maximum likelihood estimation

Choose parameters maximizing likelihood of observed data.

Explanation: Choose parameters maximizing likelihood of observed data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q6. p-value

Probability of observing data as extreme as current under null hypothesis; not probability that null is true.

Explanation: Probability of observing data as extreme as current under null hypothesis; not probability that null is true. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q1. Probability vs likelihood

Probability: data given parameters. Likelihood: parameters given observed data (up to proportionality).

Explanation: Probability: data given parameters. Likelihood: parameters given observed data (up to proportionality). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q33. Robust Covariance / Elliptic Envelope

Assumes approximately Gaussian structure and flags low-probability points via robust Mahalanobis-distance style modeling.

Explanation: Assumes approximately Gaussian structure and flags low-probability points via robust Mahalanobis-distance style modeling. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### Q12. Saddle point

Critical point with mixed curvature directions; gradient near zero but not a minimum.

Explanation: Critical point with mixed curvature directions; gradient near zero but not a minimum. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

### 3. What is the bias-variance tradeoff?

Bias is error from overly simple assumptions (underfitting). Variance is sensitivity to training data (overfitting). Better generalization requires balancing both through model capacity, regularization, data quality, and validation strategy.

Explanation: Bias is error from overly simple assumptions (underfitting). Variance is sensitivity to training data (overfitting). Better generalization requires balancing both through model capacity, regularization, data quality, and validation strategy. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q32. What is the manifold hypothesis?

The manifold hypothesis states that high-dimensional real-world data concentrates near low-dimensional manifolds.

Explanation: This is a core reason deep models can generalize: they learn coordinates/features aligned with latent manifold structure rather than raw pixel/token space.

How to do it (practical):
1. Compare PCA variance curve with nonlinear reducers (UMAP/t-SNE/Isomap).
2. Evaluate whether local neighborhoods remain consistent in latent space.
3. Use latent-space interpolation to see smooth semantic transitions.

Code:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10).fit(X)
print("explained_variance_ratio_sum:", pca.explained_variance_ratio_.sum())
```

### Q7. Why spectral radius matters in recurrent nets

It governs memory decay/amplification over time and thus stability vs expressiveness.

Explanation: It governs memory decay/amplification over time and thus stability vs expressiveness. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

## =============AI& Generative AI=========

### Q1. Attention math

`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`.

Explanation: `Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
messages = [{"role":"system","content":"Answer with grounded facts."},{"role":"user","content":query}]
resp = llm.chat(messages)
```

### Q5. Attention mechanism

Computes weighted context from key-query similarity.

Explanation: Computes weighted context from key-query similarity. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
```

### Q9. Batch normalization

Normalizes intermediate activations to stabilize/accelerate training.

Explanation: Normalizes intermediate activations to stabilize/accelerate training. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Compute batch mean and variance per channel: `mu_B=(1/m)sum(x_i)`, `sigma_B^2=(1/m)sum((x_i-mu_B)^2)`.
2. Normalize with epsilon: `x_hat=(x-mu_B)/sqrt(sigma_B^2+eps)`.
3. Apply `y=gamma*x_hat+beta`; use running mean/variance during inference.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch

x = torch.randn(16, 64, 32, 32)
bn = torch.nn.BatchNorm2d(64)
y = bn(x)
```

### Q16. BatchNorm vs LayerNorm (when to use which)

BatchNorm is usually best in CNN workloads with stable batch size. LayerNorm is preferred for Transformers and variable-length sequence models.

Explanation: BatchNorm is usually best in CNN workloads with stable batch size. LayerNorm is preferred for Transformers and variable-length sequence models. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. BatchNorm path: compute per-channel batch stats `mu_B, sigma_B^2`, normalize, then apply `y=gamma*x_hat+beta`; use running stats at inference.
2. LayerNorm path: compute per-sample feature stats `mu, var`, normalize per sample, then apply `gamma,beta`; no running batch stats.
3. Choose by workload: BatchNorm for CNNs with stable batch size, LayerNorm for Transformers/variable-length sequences.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

bn = nn.BatchNorm2d(64)        # CNN
ln = nn.LayerNorm(512)         # Transformer hidden dim
```

### Q20. Beam Search vs Sampling

Beam search optimizes likely sequences (less diverse). Sampling gives more variety and is common for open-ended generation.

Explanation: Beam search optimizes likely sequences (less diverse). Sampling gives more variety and is common for open-ended generation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
beams = [([], 0.0)]
for _ in range(max_len):
    beams = expand_and_keep_topk(beams, k=4)
```

### Q13. Causal masking

Decoder attention mask that blocks future tokens so generation stays autoregressive.

Explanation: Decoder attention mask that blocks future tokens so generation stays autoregressive. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
import torch

T = 8
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
```

### Q1. CNN

Neural network using convolutions for spatial feature extraction.

Explanation: Neural network using convolutions for spatial feature extraction. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

act = nn.GELU()
```

### Q14. Cross-attention

Decoder attends to encoder outputs in encoder-decoder models, enabling conditioned generation.

Explanation: Decoder attends to encoder outputs in encoder-decoder models, enabling conditioned generation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
out = attn @ v
```

### Q26. Distillation

Train smaller student model to mimic teacher outputs; improves deployment efficiency.

Explanation: Train smaller student model to mimic teacher outputs; improves deployment efficiency. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
teacher_logits = teacher(x).detach()
student_logits = student(x)
loss = 0.5 * ce(student_logits, y) + 0.5 * kl(student_logits, teacher_logits)
```

### Q8. Dropout

Randomly zero activations during training to reduce co-adaptation.

Explanation: Randomly zero activations during training to reduce co-adaptation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Apply dropout in training mode with probability `p`.
2. Keep inverted scaling (`1/(1-p)`) so expectation is preserved.
3. Switch to `model.eval()` for inference (dropout disabled).

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

act = nn.GELU()
```

### 27. Efficient LLM fine-tuning

Use PEFT (LoRA/QLoRA), quantization, gradient checkpointing, accumulation, and high-quality curated data subsets.

Explanation: Use PEFT (LoRA/QLoRA), quantization, gradient checkpointing, accumulation, and high-quality curated data subsets. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q4. Embedding

Dense vector representation of text/items capturing semantic similarity.

Explanation: Dense vector representation of text/items capturing semantic similarity. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define task format and evaluation dataset first.
2. Use retrieval/tooling/guardrails before larger model changes.
3. Track quality, latency, and cost together in each experiment.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode(["motor vibration anomaly"])[0]
```

### Q11. Encoder vs decoder (LLM perspective)

Encoder-focused models are strong for understanding tasks; decoder-focused models are strong for generation tasks.

Explanation: Encoder-focused models are strong for understanding tasks; decoder-focused models are strong for generation tasks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q1. Encoder vs Decoder in Transformers

An encoder builds contextual representations from input tokens (bidirectional context in encoder-only models). A decoder generates output token-by-token, using causal masking and optional cross-attention to encoder outputs.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example:
- Encoder-style use: classification, embedding, retrieval.
- Decoder-style use: text generation, chat completion.
- Encoder-decoder use: translation, summarization.

Explanation: An encoder builds contextual representations from input tokens (bidirectional context in encoder-only models). A decoder generates output token-by-token, using causal masking and optional cross-attention to encoder outputs. Example: - Encoder-style use: classification, embedding, retrieval. - Decoder-style use: text generation, chat completion. - Encoder-decoder use: translation, summarization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
messages = [{"role":"system","content":"Answer with grounded facts."},{"role":"user","content":query}]
resp = llm.chat(messages)
```

### Q12. Encoder-only vs decoder-only vs encoder-decoder

Encoder-only for classification/retrieval, decoder-only for text generation, encoder-decoder for sequence-to-sequence tasks.

Explanation: Encoder-only for classification/retrieval, decoder-only for text generation, encoder-decoder for sequence-to-sequence tasks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q5. ESN vs RNN

ESN trains only readout (faster), RNN trains full recurrence (more flexible but heavier).

Explanation: ESN trains only readout (faster), RNN trains full recurrence (more flexible but heavier). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Start from operational constraints (latency, safety, cost).
2. Validate with realistic backtests or shadow traffic.
3. Deploy with monitoring, alerts, and rollback criteria.

Example: For weekly demand forecasting, include lag-7 and lag-14 features and evaluate with walk-forward splits.

Code:
```python
import numpy as np

W = np.random.randn(200, 200) * 0.05
state = np.zeros(200)
for u in inputs:
    state = np.tanh(W @ state + u)
```

### Q34. Euclidean distance vs geodesic distance on a manifold

Euclidean distance is straight-line in ambient space; geodesic distance follows the manifold surface.

Explanation: On curved manifolds (for example swiss-roll), geodesic distance reflects intrinsic similarity better than straight-line Euclidean distance.

How to do it (practical):
1. Build a k-NN graph on samples.
2. Treat edge weights as local distances.
3. Use shortest paths on the graph as geodesic approximations.

Code:
```python
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

G = kneighbors_graph(X, n_neighbors=10, mode="distance", include_self=False)
D_geo = shortest_path(G, directed=False)
print(D_geo.shape)  # approximate geodesic distance matrix
```

### Q13. Fine-tuning

Continue training pretrained model on target data.

Explanation: Continue training pretrained model on target data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

### Q3. Fine-tuning vs prompt tuning

Fine-tuning updates parameters; prompt tuning optimizes prompts/soft tokens with fewer trainable params.

Explanation: Fine-tuning updates parameters; prompt tuning optimizes prompts/soft tokens with fewer trainable params. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q6. Hallucination

Confident but incorrect generated content.

Explanation: Confident but incorrect generated content. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q33. How do we reduce dimensionality on manifold-like data?

Use linear methods (PCA) when relationships are near-linear, and nonlinear methods (Isomap, UMAP, t-SNE) when geometry is curved.

Explanation: For manifold-shaped datasets, nonlinear methods often preserve neighborhood geometry better than PCA.

How to do it (practical):
1. Use PCA as baseline.
2. Run Isomap/UMAP for nonlinear structure.
3. Compare with trustworthiness or neighborhood overlap.

Code:
```python
from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=10, n_components=2)
Z = iso.fit_transform(X)
print(Z.shape)  # 2D embedding
```

### Q3. How do you choose weight decay for Transformers?
Answer: Sweep small values (for example `0.01`, `0.05`, `0.1`) and select by validation metric and calibration.

How to do it (practical):
1. Add regularized objective: `L=L_task+lambda1||w||_1+lambda2||w||_2^2`.
2. Use L1 for sparsity (some coefficients become zero).
3. Use L2/weight decay for smooth magnitude shrinkage and stability.

Example: Increasing weight decay can reduce overfitting on small instruction datasets.
Code:
```python
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
# typical sweep values for transformer weight decay
for wd in [0.01, 0.05, 0.1]:
    for g in optimizer.param_groups:
        g["weight_decay"] = wd
    # run short validation and log metric for each wd
```

### Q11. How do you chunk documents for RAG effectively?
Answer: Chunk by semantic boundaries with overlap, then validate retrieval hit-rate before tuning generation.

How to do it (practical):
1. Chunk and embed trusted documents, then index in vector store.
2. Retrieve top-k context, rerank, and ground answer in retrieved evidence.
3. Apply source/policy filters to mitigate prompt injection and unsafe outputs.

Example: Policy docs split by headings plus 100-token overlap can improve recall.
Code:
```python
docs = retriever.get_relevant_documents(query)
context = "\n".join(d.page_content for d in docs[:3])
response = llm.generate(f"Question: {query}

Context:
{context}")
```

### Q17. In CNN blocks, what do numbers like `256, 256, 4` mean, and how do we calculate them?

These numbers usually represent tensor shape. In image tasks this often means `Height, Width, Channels` (`H, W, C`).  
In PyTorch, tensor order is typically `N, C, H, W` (batch, channels, height, width), so the same sample is read as `C=4, H=256, W=256`.

Explanation: Shape numbers describe how data flows through layers. For `Conv -> Dropout -> Conv`, only convolution changes spatial size/channels; dropout keeps shape unchanged. In real projects, compute output shapes before training to avoid mismatch bugs and to estimate memory/compute cost.

How to do it (practical):
1. Use Conv2D output formula per spatial dimension: `out = floor((in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)`.
2. Remember dropout does not change tensor shape.
3. Calculate parameter count for Conv2D: `params = out_channels * (in_channels * kH * kW + bias_term)`.
4. Validate with a dummy forward pass (`torch.randn(...)`) and inspect output shapes.

Example: Input `N,C,H,W = 8,4,256,256` -> `Conv(4->16, k=3, s=1, p=1)` keeps `256x256`, so output becomes `8,16,256,256`.  
Then `Dropout2d` keeps `8,16,256,256`.  
Then `Conv(16->32, k=3, s=2, p=1)` downsamples to `128x128`, output `8,32,128,128`.

Code:
```python
import math

def conv_out(n, k=3, s=1, p=1, d=1):
    return math.floor((n + 2*p - d*(k-1) - 1)/s + 1)

print(conv_out(256, 3, 1, 1), conv_out(256, 3, 2, 1))
```

### Q22. KV Cache in LLM Inference

Caches previous keys/values to avoid recomputing attention over old tokens, reducing autoregressive latency.

Explanation: Caches previous keys/values to avoid recomputing attention over old tokens, reducing autoregressive latency. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
out = model(input_ids, use_cache=True)
past = out.past_key_values
next_out = model(next_ids, past_key_values=past, use_cache=True)
```

### Q15. Layer normalization

Normalizes activations across feature dimensions per sample, making training stable without relying on batch statistics.

Explanation: Normalizes activations across feature dimensions per sample, making training stable without relying on batch statistics. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Compute mean/variance across hidden features for each sample.
2. Normalize: `x_hat=(x-mu)/sqrt(var+eps)`.
3. Apply learnable `gamma,beta`; no running batch stats needed.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch

x = torch.randn(8, 128, 512)
ln = torch.nn.LayerNorm(512)
y = ln(x)
```

### Q1. LoRA

Low-rank adapters train small matrices instead of full model weights.

Explanation: Low-rank adapters train small matrices instead of full model weights. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Freeze base model and attach trainable adapter modules.
2. Train adapters on curated task data with validation checkpoints.
3. Serve base+adapter (or merged weights) and validate regression tests.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from peft import LoraConfig, get_peft_model

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, cfg)
```

### Q3. LSTM vs GRU

Both gated RNNs; GRU is simpler/faster, LSTM has separate cell state and can be more expressive.

Explanation: Both gated RNNs; GRU is simpler/faster, LSTM has separate cell state and can be more expressive. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
```

### Q15. Perplexity

`exp(cross_entropy)`; lower values indicate better average next-token prediction.

Explanation: `exp(cross_entropy)`; lower values indicate better average next-token prediction. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
import math

ppl = math.exp(cross_entropy_loss)
print(ppl)
```

### Q21. Positional Encoding vs Learned Positional Embeddings

Sinusoidal encoding is deterministic and extrapolation-friendly; learned positional embeddings can fit better in-domain but may extrapolate less.

Explanation: Sinusoidal encoding is deterministic and extrapolation-friendly; learned positional embeddings can fit better in-domain but may extrapolate less. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode(["motor vibration anomaly"])[0]
```

### Q10. Pre-LN vs Post-LN Transformer blocks

- Pre-LN: normalize before sublayer, often easier optimization for deep transformers.
- Post-LN: original formulation, can be less stable at scale.

Explanation: - Pre-LN: normalize before sublayer, often easier optimization for deep transformers. - Post-LN: original formulation, can be less stable at scale. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
messages = [{"role":"system","content":"Answer with grounded facts."},{"role":"user","content":query}]
resp = llm.chat(messages)
```

### Q24. Prompt Injection (RAG security)

Adversarial instructions in retrieved content can override behavior. Defend with source filtering, policy checks, and tool-guardrails.

Explanation: Adversarial instructions in retrieved content can override behavior. Defend with source filtering, policy checks, and tool-guardrails. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Chunk and embed trusted documents, then index in vector store.
2. Retrieve top-k context, rerank, and ground answer in retrieved evidence.
3. Apply source/policy filters to mitigate prompt injection and unsafe outputs.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
docs = retriever.get_relevant_documents(query)
context = "\n".join(d.page_content for d in docs[:3])
response = llm.generate(f"Question: {query}

Context:
{context}")
```

### Q2. QLoRA

LoRA over quantized base model for lower memory training.

Explanation: LoRA over quantized base model for lower memory training. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Freeze base model and attach trainable adapter modules.
2. Train adapters on curated task data with validation checkpoints.
3. Serve base+adapter (or merged weights) and validate regression tests.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb)
```

### Q25. Quantization: PTQ vs QAT

- PTQ (post-training quantization): fast, minimal retraining.
- QAT (quantization-aware training): better accuracy retention, more effort.

Explanation: - PTQ (post-training quantization): fast, minimal retraining. - QAT (quantization-aware training): better accuracy retention, more effort. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
ptq_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# QAT: prepare_qat -> train -> convert
```

### Q8. RAG

Retrieve relevant documents and condition generation on retrieved context.

Explanation: Retrieve relevant documents and condition generation on retrieved context. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Chunk and embed trusted documents, then index in vector store.
2. Retrieve top-k context, rerank, and ground answer in retrieved evidence.
3. Apply source/policy filters to mitigate prompt injection and unsafe outputs.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
docs = retriever.get_relevant_documents(query)
context = "\n".join(d.page_content for d in docs[:3])
response = llm.generate(f"Question: {query}

Context:
{context}")
```

### Q7. Reduce hallucination

RAG, better prompts, constrained decoding, tool use, verification, and fine-tuning on reliable data.

Explanation: RAG, better prompts, constrained decoding, tool use, verification, and fine-tuning on reliable data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q2. RNN

Sequence model with recurrent state passing through time.

Explanation: Sequence model with recurrent state passing through time. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

cnn = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
```

### Q14. Self-supervised learning

Learn representations from unlabeled data via pretext/objective construction.

Explanation: Learn representations from unlabeled data via pretext/objective construction. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

### Q19. Temperature in generation

Scales logits before softmax. Low temperature makes output conservative; high temperature increases diversity.

Explanation: Scales logits before softmax. Low temperature makes output conservative; high temperature increases diversity. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
scaled_logits = logits / 0.7
probs = torch.softmax(scaled_logits, dim=-1)
```

### Q5. Tokenization

Convert text into model-consumable token IDs.

Explanation: Convert text into model-consumable token IDs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define task format and evaluation dataset first.
2. Use retrieval/tooling/guardrails before larger model changes.
3. Track quality, latency, and cost together in each experiment.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ids = tok("Hello world", return_tensors="pt")["input_ids"]
```

### Q18. Top-k vs Top-p sampling

- Top-k: sample from k highest-probability tokens.
- Top-p: sample from smallest token set whose cumulative probability >= p.
Top-p is often more adaptive.

Explanation: - Top-k: sample from k highest-probability tokens. - Top-p: sample from smallest token set whose cumulative probability >= p. Top-p is often more adaptive. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
# pseudo decoding
logits = top_k_filter(logits, k=50)   # or top_p_filter(logits, p=0.9)
next_token = sample(logits)
```

### Q12. Transfer learning

Reuse pretrained representations for new tasks.

Explanation: Reuse pretrained representations for new tasks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

### Q4. Transformer

Attention-based architecture enabling parallel sequence modeling.

Explanation: Attention-based architecture enabling parallel sequence modeling. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

block = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
```

### 24. Transformers vs RNN/ESN

Transformers handle long-range dependencies and parallelize well. RNN/ESN can still win in low-latency, low-resource streaming settings.

Explanation: Transformers handle long-range dependencies and parallelize well. RNN/ESN can still win in low-latency, low-resource streaming settings. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q9. Vector database

Index/store embeddings for similarity search at scale.

Explanation: Index/store embeddings for similarity search at scale. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Chunk and embed trusted documents, then index in vector store.
2. Retrieve top-k context, rerank, and ground answer in retrieved evidence.
3. Apply source/policy filters to mitigate prompt injection and unsafe outputs.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
docs = retriever.get_relevant_documents(query)\ncontext = "\n".join(d.page_content for d in docs[:3])\nanswer = llm.generate(context)
```

### Q11. Weight Decay vs Dropout

- Weight decay constrains parameter magnitude.
- Dropout stochastically removes activations during training.
They regularize differently and are often combined.

Explanation: - Weight decay constrains parameter magnitude. - Dropout stochastically removes activations during training. They regularize differently and are often combined. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Add regularized objective: `L=L_task+lambda1||w||_1+lambda2||w||_2^2`.
2. Use L1 for sparsity (some coefficients become zero).
3. Use L2/weight decay for smooth magnitude shrinkage and stability.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
prompt = f"Question: {query}\nContext: {context}"\nresponse = llm.generate(prompt)
```

### Q13. What are common causes of hallucination in long-context prompts?
Answer: Noisy context, contradictory sources, weak instructions, and over-trust in low-quality retrieved text.

Example: Mixing outdated and current manuals leads to fabricated synthesis.

### Q31. What is a manifold in machine learning?

A manifold is a lower-dimensional structure embedded in a higher-dimensional space. Many real datasets lie near such structures instead of filling the full ambient space.

Explanation: In representation learning, manifolds explain why high-dimensional inputs (images, audio, text embeddings) can still be modeled efficiently: the true variation often depends on fewer latent factors.

How to do it (practical):
1. Start with dimensionality reduction to inspect whether data concentrates on a low-dimensional structure.
2. Check local-neighborhood preservation after projection.
3. Use manifold-aware methods (autoencoders, diffusion maps, UMAP, Isomap) for analysis or feature extraction.

Code:
```python
from sklearn.datasets import make_swiss_roll
X, _ = make_swiss_roll(n_samples=2000, noise=0.05, random_state=42)  # classic manifold dataset
print(X.shape)  # (2000, 3) points lying on a 2D manifold in 3D
```

### Q3. What is causal masking?

A decoder mask that prevents each token from attending to future tokens, preserving autoregressive generation.

Explanation: A decoder mask that prevents each token from attending to future tokens, preserving autoregressive generation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
import torch

T = 8
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
```

### Q4. What is cross-attention?

In encoder-decoder models, decoder queries attend to encoder keys/values so output is conditioned on source input.

Explanation: In encoder-decoder models, decoder queries attend to encoder keys/values so output is conditioned on source input. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)
out = attn @ v
```

### Q35. What is manifold regularization?

Manifold regularization enforces similar predictions for nearby points on the data manifold.

Explanation: It adds a smoothness constraint using a graph Laplacian, helping semi-supervised learning and improving robustness when labels are limited.

How to do it (practical):
1. Construct neighborhood graph on all samples (labeled + unlabeled).
2. Build graph Laplacian `L = D - W`.
3. Add smoothness term `f^T L f` to the task loss.

Code:
```python
import numpy as np
from sklearn.neighbors import kneighbors_graph

W = kneighbors_graph(X, n_neighbors=10, mode="connectivity", include_self=False).toarray()
D = np.diag(W.sum(axis=1))
L = D - W  # graph Laplacian
# manifold penalty example for prediction vector f: penalty = f.T @ L @ f
```

### Q9. What is RMSNorm and how is it different from LayerNorm?
Answer: RMSNorm scales by root-mean-square only (no mean subtraction), often cheaper and stable in LLMs.

How to do it (practical):
1. Compute mean/variance across hidden features for each sample.
2. Normalize: `x_hat=(x-mu)/sqrt(var+eps)`.
3. Apply learnable `gamma,beta`; no running batch stats needed.

Example: Some large decoder-only models prefer RMSNorm for speed and stability.
Code:
```python
import torch

x = torch.randn(8, 128, 512)
ln = torch.nn.LayerNorm(512)
y = ln(x)
```

### Q8. What is SwiGLU and why used in modern LLMs?
Answer: SwiGLU is a gated feed-forward activation that often improves quality/efficiency tradeoffs.

Example: Many modern decoder architectures replace plain FFN with gated variants.

### Q8. When BatchNorm can fail

Very small batches, non-iid batch composition, or highly variable sequence workloads can make batch statistics noisy.

Explanation: Very small batches, non-iid batch composition, or highly variable sequence workloads can make batch statistics noisy. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Compute batch mean and variance per channel: `mu_B=(1/m)sum(x_i)`, `sigma_B^2=(1/m)sum((x_i-mu_B)^2)`.
2. Normalize with epsilon: `x_hat=(x-mu_B)/sqrt(sigma_B^2+eps)`.
3. Apply `y=gamma*x_hat+beta`; use running mean/variance during inference.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
import torch

x = torch.randn(16, 64, 32, 32)
bn = torch.nn.BatchNorm2d(64)
y = bn(x)
```

### Q10. When should you use Mixture-of-Experts (MoE)?
Answer: Use MoE when you need larger model capacity without proportional per-token compute cost.

Example: Serving constraints allow sparse expert routing but not dense full-model execution.

### Q9. Why cross-entropy over MSE in classification

Cross-entropy aligns with probabilistic likelihood and gives stronger gradients for confident wrong predictions.

Explanation: Cross-entropy aligns with probabilistic likelihood and gives stronger gradients for confident wrong predictions. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Write the exact objective/function you are optimizing.
2. Implement a baseline and verify with held-out evaluation.
3. Run ablations to confirm which change caused improvement.

Example: With an ill-conditioned Hessian, optimization zig-zags until normalization or preconditioning is applied.

Code:
```python
from scipy.stats import ttest_ind

stat, p = ttest_ind(a, b, equal_var=False)
print(p)
```

### Q9. Why LayerNorm in Transformers

It is independent of batch statistics and stable for sequence modeling and distributed setups with varying micro-batches.

Explanation: It is independent of batch statistics and stable for sequence modeling and distributed setups with varying micro-batches. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Compute mean/variance across hidden features for each sample.
2. Normalize: `x_hat=(x-mu)/sqrt(var+eps)`.
3. Apply learnable `gamma,beta`; no running batch stats needed.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
import torch

x = torch.randn(8, 128, 512)
ln = torch.nn.LayerNorm(512)
y = ln(x)
```

### Q11. Why transformers are powerful

Long-range dependency modeling + parallelization + scaling behavior.

Explanation: Long-range dependency modeling + parallelization + scaling behavior. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Project inputs to `Q,K,V` and compute attention scores.
2. Apply mask (causal for decoder) before softmax.
3. Monitor cross-entropy/perplexity and downstream task quality.

Example: Adding residual connections can let a deeper model converge where a plain stack fails.

Code:
```python
import torch.nn as nn

block = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
```

## ============ Digital Twin===========

### 29. AI in industrial systems

Typical use-cases: anomaly detection, predictive maintenance, optimization, quality control, digital twins, and decision support.

Explanation: Typical use-cases: anomaly detection, predictive maintenance, optimization, quality control, digital twins, and decision support. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q5. Anomaly detection in sensor data

Combine statistical baselines + ML detectors + rule checks, with human-in-the-loop triage.

Explanation: Combine statistical baselines + ML detectors + rule checks, with human-in-the-loop triage. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

Code:
```python
if drift_score > 0.2:
    mode = "safe_mode"
```

### Q1. Digital twin

Virtual representation of physical assets/processes continuously updated from data.

Explanation: Virtual representation of physical assets/processes continuously updated from data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q6. Ensure physical consistency

Constraint-aware training, physics-informed losses, and post-hoc rule validation.

Explanation: Constraint-aware training, physics-informed losses, and post-hoc rule validation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q36. Foundation models for industrial anomaly detection

Pretrained multi-modal or time-series foundation models can provide stronger representations, then lightweight heads/adapters detect anomalies with less labeled data.

Explanation: Pretrained multi-modal or time-series foundation models can provide stronger representations, then lightweight heads/adapters detect anomalies with less labeled data. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q21. How do you detect concept drift vs sensor fault?
Answer: Compare multi-sensor consistency and reference checks; drift affects patterns broadly, sensor faults are localized.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: One sensor jumps while correlated sensors remain stable, indicating sensor fault.
Code:
```python
import numpy as np

# if only one sensor drifts while peers remain stable, suspect sensor fault
z_a = (sensor_a - np.mean(sensor_a_ref)) / (np.std(sensor_a_ref) + 1e-8)
z_b = (sensor_b - np.mean(sensor_b_ref)) / (np.std(sensor_b_ref) + 1e-8)
sensor_fault = (abs(z_a) > 3.0) and (abs(z_b) < 1.0)
print("sensor_fault:", sensor_fault)
```

### Q12. Optimize industrial processes

Use forecasting + optimization + control under operational constraints.

Explanation: Use forecasting + optimization + control under operational constraints. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### 30. Physical consistency in AI models

Enforce constraints in loss/architecture, validate against known laws, and combine model outputs with simulation/domain checks.

Explanation: Enforce constraints in loss/architecture, validate against known laws, and combine model outputs with simulation/domain checks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q2. Physically impossible model result

Add constraint checks, retrain with physics-informed loss/features, and block unsafe predictions in serving layer.

Explanation: Add constraint checks, retrain with physics-informed loss/features, and block unsafe predictions in serving layer. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q4. Predictive maintenance

Forecast failure risk/RUL from sensor history to schedule interventions proactively.

Explanation: Forecast failure risk/RUL from sensor history to schedule interventions proactively. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q11. Surrogate modeling

Train fast approximator for expensive simulation.

Explanation: Train fast approximator for expensive simulation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q10. Unreliable sensors

Imputation, sensor health scoring, redundancy, robust filtering, and uncertainty-aware outputs.

Explanation: Imputation, sensor health scoring, redundancy, robust filtering, and uncertainty-aware outputs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

## ============Leadership=============

### Q1. AI predicts failure but engineer disagrees

Review evidence together, compare with sensor history/physics checks, run targeted validation, then decide with safety-first policy.

Explanation: Review evidence together, compare with sensor history/physics checks, run targeted validation, then decide with safety-first policy. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### 31. AI vs physics-model conflict

Investigate both sides: data quality, model assumptions, sensor errors, boundary conditions. Use real-world evidence and hybrid modeling when useful.

Explanation: Investigate both sides: data quality, model assumptions, sensor errors, boundary conditions. Use real-world evidence and hybrid modeling when useful. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q6. Communicating to non-technical teams

Use simple language, visuals, and business-impact framing.

Explanation: Use simple language, visuals, and business-impact framing. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q10. Debug production failure

Triage impact, isolate component, rollback if needed, run RCA, and patch with tests.

Explanation: Triage impact, isolate component, rollback if needed, run RCA, and patch with tests. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### 15. Describe a model failure and recovery

Common pattern: model strong offline, weak online due to distribution shift. Diagnose with data and feature drift analysis, fix preprocessing parity, retrain with representative production slices, and add monitoring/alerts.

Explanation: Common pattern: model strong offline, weak online due to distribution shift. Diagnose with data and feature drift analysis, fix preprocessing parity, retrain with representative production slices, and add monitoring/alerts. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q7. Difficult decision making

Define constraints, evaluate options quantitatively, document rationale, and monitor outcomes.

Explanation: Define constraints, evaluate options quantitatively, document rationale, and monitor outcomes. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q8. Ensuring team productivity

Clear goals, unblock dependencies early, and enforce lightweight execution rituals.

Explanation: Clear goals, unblock dependencies early, and enforce lightweight execution rituals. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q9. Giving feedback

Specific, timely, respectful, behavior-focused, with clear next actions.

Explanation: Specific, timely, respectful, behavior-focused, with clear next actions. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q1. Handling conflict

Clarify goals, align on facts, discuss tradeoffs, and converge on decision criteria.

Explanation: Clarify goals, align on facts, discuss tradeoffs, and converge on decision criteria. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q3. Handling failure

Acknowledge quickly, analyze root cause, communicate transparently, and prevent recurrence.

Explanation: Acknowledge quickly, analyze root cause, communicate transparently, and prevent recurrence. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q10. Leadership style

Context-driven, collaborative, quality-focused, and outcome-oriented.

Explanation: Context-driven, collaborative, quality-focused, and outcome-oriented. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q2. Mentoring juniors

Set clear expectations, pair regularly, provide actionable feedback, and grow ownership gradually.

Explanation: Set clear expectations, pair regularly, provide actionable feedback, and grow ownership gradually. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q4. Prioritizing multiple deadlines

Use impact-risk-effort framework and align with stakeholders on sequence.

Explanation: Use impact-risk-effort framework and align with stakeholders on sequence. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

### Q5. Unclear requirements

Run discovery, define assumptions, propose milestones, and iterate with feedback.

Explanation: Run discovery, define assumptions, propose milestones, and iterate with feedback. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: Two teams disagree on roadmap priority; you align on impact, risk, and effort criteria.

## ===========Deployment===============

### 10. Challenges in deploying AI systems

Data quality/drift, train-serving skew, latency/scalability limits, integration complexity, observability gaps, and ongoing maintenance/retraining burden.

Explanation: Data quality/drift, train-serving skew, latency/scalability limits, integration complexity, observability gaps, and ongoing maintenance/retraining burden. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q7. Deploy in 2 days

Use simplest reliable baseline, strict guardrails, shadow/canary rollout, and clear rollback.

Explanation: Use simplest reliable baseline, strict guardrails, shadow/canary rollout, and clear rollback. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If model and engineer disagree in a safety-critical case, route through evidence review and safe fallback.

### Q1. Design real-time anomaly detection

Streaming ingestion -> feature extraction -> low-latency model -> thresholding -> alerting -> feedback loop.

Explanation: Streaming ingestion -> feature extraction -> low-latency model -> thresholding -> alerting -> feedback loop. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

Code:
```python
if drift_score > 0.2:
    mode = "safe_mode"
```

### 14. Designing real-time AI systems

Define latency SLOs first, then optimize model (quantization/pruning/distillation), serving path (batching, caching, async pipelines), and infrastructure (edge/cloud split). Balance accuracy-latency-cost.

Explanation: Define latency SLOs first, then optimize model (quantization/pruning/distillation), serving path (batching, caching, async pipelines), and infrastructure (edge/cloud split). Balance accuracy-latency-cost. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q10. Efficient LLM deployment

Quantization, distillation, KV-cache, batching, speculative decoding, optimized serving stack.

Explanation: Quantization, distillation, KV-cache, batching, speculative decoding, optimized serving stack. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

### Q4. Ensure 24/7 reliability

Redundancy, health checks, autoscaling, SLO monitoring, and on-call runbooks.

Explanation: Redundancy, health checks, autoscaling, SLO monitoring, and on-call runbooks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q3. Fallback if AI fails

Rule-based backup, safe defaults, circuit breaker, and human escalation.

Explanation: Rule-based backup, safe defaults, circuit breaker, and human escalation. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q5. Handle streaming data

Windowed processing, out-of-order handling, watermarking, and state management.

Explanation: Windowed processing, out-of-order handling, watermarking, and state management. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q19. How do you define rollback criteria before deployment?
Answer: Predefine hard thresholds for latency, error rate, and business KPI regression.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: Roll back if p95 latency rises >20% or critical-alert miss rate rises.
Code:
```python
rollback = (
    current["p95_latency_ms"] > 1.2 * baseline["p95_latency_ms"]
    or current["critical_miss_rate"] > limits["critical_miss_rate"]
    or current["business_kpi_drop"] > limits["business_kpi_drop"]
)
print("rollback:", rollback)
```

### Q17. How do you run safe canary deployment for ML models?
Answer: Start with small traffic percentage, compare against baseline, and auto-rollback on threshold violations.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: Send 5% traffic to candidate and monitor error/latency/failure rates.
Code:
```python
traffic = {"baseline": 0.95, "candidate": 0.05}

candidate_ok = (
    candidate_metrics["error_rate"] <= 1.05 * baseline_metrics["error_rate"]
    and candidate_metrics["p95_latency_ms"] <= 1.10 * baseline_metrics["p95_latency_ms"]
)
if not candidate_ok:
    traffic = {"baseline": 1.0, "candidate": 0.0}  # rollback
```

### Q2. Meet strict latency constraints

Optimize model size, runtime, batching, hardware placement, and avoid slow synchronous dependencies.

Explanation: Optimize model size, runtime, batching, hardware placement, and avoid slow synchronous dependencies. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Start from operational constraints (latency, safety, cost).
2. Validate with realistic backtests or shadow traffic.
3. Deploy with monitoring, alerts, and rollback criteria.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

Code:
```python
if drift_score > 0.2:
    mode = "safe_mode"
```

### Q8. Real-time deployment

Low-latency model, streaming pipeline, bounded inference path, and resilient serving.

Explanation: Low-latency model, streaming pipeline, bounded inference path, and resilient serving. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q14. Scale AI in large systems

Standardized MLOps, shared feature/model services, automated monitoring/retraining.

Explanation: Standardized MLOps, shared feature/model services, automated monitoring/retraining. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: For predictive maintenance, model output triggers a maintenance ticket only after safety checks.

### Q6. Scale to millions of points

Partitioned pipelines, distributed stream processors, and efficient online feature stores.

Explanation: Partitioned pipelines, distributed stream processors, and efficient online feature stores. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q29. Throughput vs Latency

Throughput is requests per second; latency is time per request. Optimizing one may hurt the other.

Explanation: Throughput is requests per second; latency is time per request. Optimizing one may hurt the other. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Define objective and constraints clearly.
2. Implement the simplest reliable baseline.
3. Iterate with metrics, error analysis, and monitoring.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

Code:
```python
throughput_rps = total_requests / elapsed_seconds
latency_ms = (elapsed_seconds / total_requests) * 1000
print(throughput_rps, latency_ms)
```

### Q2. Why LLMs scale with data

Large models with large diverse data learn transferable representations and in-context capabilities.

Explanation: Large models with large diverse data learn transferable representations and in-context capabilities. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A support chatbot uses retrieval from approved docs to reduce hallucination in answers.

## ===========Monitoring===============

### Q18. Anomaly detection metrics

Use Precision, Recall, F1, PR-AUC, ROC-AUC, false alarm rate, detection delay, and event-level recall (not only point-level accuracy).

Explanation: Use Precision, Recall, F1, PR-AUC, ROC-AUC, false alarm rate, detection delay, and event-level recall (not only point-level accuracy). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from scipy.stats import ttest_ind

stat, p = ttest_ind(a, b, equal_var=False)
print(p)
```

### Q26. ARIMA for anomaly detection

Fit ARIMA, compute residuals, and flag anomalies where residuals exceed statistically justified bounds.

Explanation: Fit ARIMA, compute residuals, and flag anomalies where residuals exceed statistically justified bounds. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
from statsmodels.tsa.arima.model import ARIMA

fit = ARIMA(series, order=(2, 1, 1)).fit()
forecast = fit.forecast(steps=7)
```

### Q27. Autoencoder-based anomaly detection

Train an autoencoder on normal data only. At inference, high reconstruction error indicates potential anomaly.

Explanation: Train an autoencoder on normal data only. At inference, high reconstruction error indicates potential anomaly. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
recon = autoencoder(x)
score = ((x - recon) ** 2).mean(dim=1)
```

### Q30. CNN-based anomaly detection for signals

1D-CNNs are effective for vibration/sensor windows, capturing local temporal motifs and abrupt pattern changes.

Explanation: 1D-CNNs are effective for vibration/sensor windows, capturing local temporal motifs and abrupt pattern changes. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q17. Common anomaly detection methods

Z-score/IQR rules, Gaussian models, Isolation Forest, One-Class SVM, Autoencoders, and time-series residual-based detectors.

Explanation: Z-score/IQR rules, Gaussian models, Isolation Forest, One-Class SVM, Autoencoders, and time-series residual-based detectors. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

mu, var = np.mean(x), np.var(x)
print(mu, var)
```

### Q23. CUSUM vs EWMA

CUSUM is strong for fast detection of small sustained shifts; EWMA smooths noise and tracks gradual drift trends effectively.

Explanation: CUSUM is strong for fast detection of small sustained shifts; EWMA smooths noise and tracks gradual drift trends effectively. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
s_pos = max(0.0, s_pos + (x_t - mu0 - k))
if s_pos > h:
    alarm = True
```

### Q7. Design alert thresholds

Risk-based thresholds, precision/recall tradeoffs, dynamic baselines, and escalation tiers.

Explanation: Risk-based thresholds, precision/recall tradeoffs, dynamic baselines, and escalation tiers. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q35. Event-based vs point-based anomaly evaluation

Point metrics score individual timestamps; event metrics score whether an anomalous event window was detected with acceptable delay.

Explanation: Point metrics score individual timestamps; event metrics score whether an anomalous event window was detected with acceptable delay. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

mae = np.mean(np.abs(y_true - y_pred))
print(mae)
```

### Q29. GAN-based anomaly detection (for example AnoGAN-style)

Train a GAN on normal data distribution and use generator/discriminator mismatch or reconstruction in latent space as anomaly score.

Explanation: Train a GAN on normal data distribution and use generator/discriminator mismatch or reconstruction in latent space as anomaly score. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
z = optimize_latent_for_sample(x, generator)
x_hat = generator(z)
score = ((x - x_hat) ** 2).mean()
```

### Q8. Handle delayed data

Buffering, event-time processing, late-arrival correction, and re-computation policies.

Explanation: Buffering, event-time processing, late-arrival correction, and re-computation policies. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q24. How do you evaluate anomaly detector lead time?
Answer: Measure how early an alert appears before confirmed event onset, plus false alarm burden.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A detector that alerts 2 hours early with acceptable precision is operationally useful.
Code:
```python
lead_time_minutes = (event_start_ts - first_alert_ts) / 60.0
is_useful = lead_time_minutes >= 30 and false_alarm_rate <= 0.05
print("lead_time_minutes:", lead_time_minutes, "useful:", is_useful)
```

### 11. How do you handle data drift?

Monitor feature distributions and performance drift (PSI/KS/population shifts). Identify root cause, retrain with fresh representative data, recalibrate thresholds, and automate drift-response workflows.

Explanation: Monitor feature distributions and performance drift (PSI/KS/population shifts). Identify root cause, retrain with fresh representative data, recalibrate thresholds, and automate drift-response workflows. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q23. How do you set alert hysteresis to avoid alarm flapping?
Answer: Use separate on/off thresholds so alerts do not toggle rapidly around one boundary.

Example: Trigger at `0.8`, clear only when score falls below `0.6`.

### Q5. How do you tune gradient clipping threshold?
Answer: Start near `1.0`, inspect gradient norms, and adjust so clipping happens occasionally, not every step.

How to do it (practical):
1. Backpropagate normally first.
2. Clip before optimizer step (`clip_grad_norm_` or value clip).
3. Track clipping frequency and tune LR/max_norm accordingly.

Example: Sequence models may need lower thresholds than vision models.
Code:
```python
import torch

loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
print(float(grad_norm))
```

### Q9. Monitoring pipeline

Monitor input quality, drift, model outputs, latency, errors, and business KPIs.

Explanation: Monitor input quality, drift, model outputs, latency, errors, and business KPIs. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: In streaming anomaly detection, hysteresis reduces noisy alert flapping.

### Q19. Point anomaly vs contextual anomaly vs collective anomaly

Point anomaly: single unusual sample. Contextual anomaly: unusual under context (time/season). Collective anomaly: abnormal pattern over a sequence/window.

Explanation: Point anomaly: single unusual sample. Contextual anomaly: unusual under context (time/season). Collective anomaly: abnormal pattern over a sequence/window. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
print(rmse)
```

### Q16. Statistical anomaly detection (what is it?)

Detects data points or sequences that deviate significantly from expected statistical behavior (distribution, trend, or temporal pattern).

Explanation: Detects data points or sequences that deviate significantly from expected statistical behavior (distribution, trend, or temporal pattern). In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

z = (x - np.mean(x)) / (np.std(x) + 1e-8)
print(z[:5])
```

### Q20. Threshold selection for anomaly scores

Set thresholds using validation data, percentile rules, extreme value theory, or cost-based optimization for false positive vs false negative tradeoff.

Explanation: Set thresholds using validation data, percentile rules, extreme value theory, or cost-based optimization for false positive vs false negative tradeoff. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
import numpy as np

mae = np.mean(np.abs(y_true - y_pred))
print(mae)
```

### Q21. What is CUSUM?

CUSUM (Cumulative Sum Control Chart) is a change detection method that accumulates small deviations from a target mean to detect distribution shifts quickly.

Explanation: CUSUM (Cumulative Sum Control Chart) is a change detection method that accumulates small deviations from a target mean to detect distribution shifts quickly. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
s_pos = max(0.0, s_pos + (x_t - mu0 - k))
if s_pos > h:
    alarm = True
```

### Q20. What is the minimum monitoring dashboard for online inference?
Answer: Input quality, drift, output distribution, latency percentiles, error rate, and business KPI trend.

Example: A concise dashboard can still catch most production regressions early.

### Q28. Why autoencoders work for anomaly detection

They learn a compact manifold of normal patterns; out-of-distribution inputs reconstruct poorly.

Explanation: They learn a compact manifold of normal patterns; out-of-distribution inputs reconstruct poorly. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
recon = autoencoder(x)
score = ((x - recon) ** 2).mean(dim=1)
```

### Q22. Why use CUSUM in monitoring?

It is sensitive to small persistent shifts that simple threshold alarms often miss.

Explanation: It is sensitive to small persistent shifts that simple threshold alarms often miss. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Train detector on representative normal baseline (or labeled anomalies if available).
2. Tune threshold on validation events to balance precision/recall.
3. Track false alarms, detection delay, and event-level recall in production.

Example: A p-value below 0.05 suggests evidence against the null, but does not prove causality.

Code:
```python
s_pos = max(0.0, s_pos + (x_t - mu0 - k))
if s_pos > h:
    alarm = True
```

## ===========Pytorch===============

### 25. `model.eval()` vs `torch.no_grad()`

`model.eval()` changes layer behavior (dropout/batchnorm). `torch.no_grad()` disables gradient tracking. Use both in inference.

Explanation: `model.eval()` changes layer behavior (dropout/batchnorm). `torch.no_grad()` disables gradient tracking. Use both in inference. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q3. `model.train()` vs `model.eval()`

`train()` enables training-time behavior (dropout/bn updates). `eval()` freezes inference behavior.

Explanation: `train()` enables training-time behavior (dropout/bn updates). `eval()` freezes inference behavior. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q4. `torch.no_grad()`

Context manager disabling gradient tracking to save memory/compute.

Explanation: Context manager disabling gradient tracking to save memory/compute. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q1. Autograd in PyTorch

Automatic differentiation engine building computational graph and computing gradients via backprop.

Explanation: Automatic differentiation engine building computational graph and computing gradients via backprop. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Run forward pass and compute loss.
2. Call `loss.backward()` to populate gradients.
3. Step optimizer and clear gradients each iteration.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**3
y.backward()
print(x.grad)
```

### Q5. Avoid memory leaks

Clear references, avoid storing graph tensors, use `detach()` where needed, and monitor retained objects.

Explanation: Clear references, avoid storing graph tensors, use `detach()` where needed, and monitor retained objects. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

### Q6. Backpropagation

Applies chain rule from loss to parameters to compute gradients.

Explanation: Backpropagation computes gradients efficiently by propagating local derivatives backward through the computation graph using chain rule. For a simple scalar path `L(w)= (wx - y)^2`, the gradient is `dL/dw = 2(wx - y)*x`.

How to do it (practical):
1. Write forward equations and local derivatives for each node.
2. Multiply local derivatives backward (chain rule) to get parameter gradients.
3. Optionally verify with finite-difference check: `dL/dw ~= (L(w+eps)-L(w-eps))/(2*eps)`.
4. In training loop: `zero_grad -> forward -> backward -> step`.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch

# Manual gradient for L(w) = (w*x - y)^2
x = torch.tensor(3.0)
y = torch.tensor(7.0)
w = torch.tensor(0.5, requires_grad=True)

loss = (w * x - y) ** 2
loss.backward()
print("autograd dL/dw:", w.grad.item())

# finite-difference gradient check
eps = 1e-4
with torch.no_grad():
    lp = ((w + eps) * x - y) ** 2
    lm = ((w - eps) * x - y) ** 2
num_grad = (lp - lm) / (2 * eps)
print("numeric dL/dw:", num_grad.item())

# one optimization step
opt = torch.optim.SGD([w], lr=0.1)
opt.step()
```

### Q11. Checkpointing

Save model/optimizer/scheduler/scaler states for recovery and reproducibility.

Explanation: Save model/optimizer/scheduler/scaler states for recovery and reproducibility. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Save model + optimizer + scheduler (+ scaler) states.
2. Keep both best-validation and periodic recovery checkpoints.
3. Version checkpoint with config/data hash for reproducibility.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch

torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, "ckpt.pt")
```

### Q2. Computational graph

Directed graph of tensor operations used to compute outputs and gradients.

Explanation: Directed graph of tensor operations used to compute outputs and gradients. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q5. DataLoader

Batches, shuffles, parallel-loads dataset samples for efficient training loops.

Explanation: Batches, shuffles, parallel-loads dataset samples for efficient training loops. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
```

### Q13. DataParallel

Single-process multi-GPU split with central gather; simpler but less scalable.

Explanation: Single-process multi-GPU split with central gather; simpler but less scalable. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Launch one process per GPU and shard data with distributed sampler.
2. Sync gradients using all-reduce (or shard states with FSDP/ZeRO).
3. Save rank-safe checkpoints and aggregate metrics across workers.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch.nn as nn

model = nn.DataParallel(model)
```

### 21. DataParallel vs DistributedDataParallel

`DataParallel` is easier but slower due to central bottleneck. `DistributedDataParallel` is preferred for real workloads: better scaling, less overhead, multi-node ready.

Explanation: `DataParallel` is easier but slower due to central bottleneck. `DistributedDataParallel` is preferred for real workloads: better scaling, less overhead, multi-node ready. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q12. DDP

Multi-process distributed training with gradient all-reduce.

Explanation: Multi-process distributed training with gradient all-reduce. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Launch one process per GPU and shard data with distributed sampler.
2. Sync gradients using all-reduce (or shard states with FSDP/ZeRO).
3. Save rank-safe checkpoints and aggregate metrics across workers.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch.nn.parallel as p

model = p.DistributedDataParallel(model, device_ids=[local_rank])
```

### Q9. Debug NaNs in training

Check inputs/labels, LR, loss scale, division/log operations, exploding grads; enable anomaly detection.

Explanation: Check inputs/labels, LR, loss scale, division/log operations, exploding grads; enable anomaly detection. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q6. Debug slow DataLoader

Profile worker time, serialization overhead, transforms, storage format, and host-device transfer.

Explanation: Profile worker time, serialization overhead, transforms, storage format, and host-device transfer. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

Code:
```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
```

### Q10. Deploy PyTorch model

Export/serve with TorchScript/ONNX/Triton/FastAPI pipeline with observability and rollback.

Explanation: Export/serve with TorchScript/ONNX/Triton/FastAPI pipeline with observability and rollback. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

### Q7. Gradient accumulation

Accumulate gradients over multiple mini-batches before optimizer step to emulate larger batch size.

Explanation: Accumulate gradients over multiple mini-batches before optimizer step to emulate larger batch size. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
accum = 4
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    (criterion(model(x), y) / accum).backward()
    if (i + 1) % accum == 0:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

### Q3. Implement custom loss

Subclass `nn.Module` or write function using tensor ops, ensuring stable numerics.

Explanation: Subclass `nn.Module` or write function using tensor ops, ensuring stable numerics. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

### Q2. Implement gradient accumulation

Scale loss by accumulation steps, call backward each mini-batch, optimizer step every k steps.

Explanation: Scale loss by accumulation steps, call backward each mini-batch, optimizer step every k steps. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Implement a minimal baseline pipeline first.
2. Profile bottlenecks (data, compute, memory) before optimization.
3. Add logging/tests so training and inference behavior stay consistent.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

Code:
```python
accum = 4
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    (criterion(model(x), y) / accum).backward()
    if (i + 1) % accum == 0:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

### Q10. Mixed precision

Use FP16/BF16 for faster compute and lower memory with loss scaling when needed.

Explanation: Use FP16/BF16 for faster compute and lower memory with loss scaling when needed. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Wrap forward/loss in `autocast`.
2. Use `GradScaler` for FP16 to prevent underflow.
3. Validate speedup and numeric stability on validation set.

Example: If training is slow, profile dataloader wait time before changing model architecture.

Code:
```python
import torch

scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = criterion(model(x), y)
scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
```

### Q8. Reduce GPU memory usage

Mixed precision, smaller batches, gradient checkpointing, sequence truncation, activation recomputation, optimizer/state choices.

Explanation: Mixed precision, smaller batches, gradient checkpointing, sequence truncation, activation recomputation, optimizer/state choices. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q4. Variable-length sequences efficiently

Pad + mask, packed sequences, bucketing by length, or attention masks.

Explanation: Pad + mask, packed sequences, bucketing by length, or attention masks. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

## ==========jax===================

### Q16. JIT compilation

Compile computation graphs for optimized execution.

Explanation: Compile computation graphs for optimized execution. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### 19. Large-scale training pipeline (PyTorch/JAX)

Optimize data IO (sharding/prefetch), compute (mixed precision), and scale (DDP/pmap/sharding). Keep sequence/window generation efficient and monitor throughput, memory, and utilization.

Explanation: Optimize data IO (sharding/prefetch), compute (mixed precision), and scale (DDP/pmap/sharding). Keep sequence/window generation efficient and monitor throughput, memory, and utilization. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: You deployed a defect detector where precision improved from 0.78 to 0.91 while maintaining sub-50 ms inference.

### Q14. PyTorch vs TensorFlow vs JAX

PyTorch: flexible/eager ecosystem. TensorFlow: strong production tooling. JAX: functional style + strong compiler transformations.

Explanation: PyTorch: flexible/eager ecosystem. TensorFlow: strong production tooling. JAX: functional style + strong compiler transformations. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

### Q15. When to use JAX

When vectorization/JIT/XLA and functional transformations (`jit`, `vmap`, `pmap`) are major advantages.

Explanation: When vectorization/JIT/XLA and functional transformations (`jit`, `vmap`, `pmap`) are major advantages. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.

## ==========python===============

### Q9. Design experiment tracking

Log configs, data/version hash, metrics, artifacts, model registry, and reproducible seeds.

Explanation: Log configs, data/version hash, metrics, artifacts, model registry, and reproducible seeds. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

### Q8. Implement early stopping

Track best validation metric with patience and checkpoint best model.

Explanation: Track best validation metric with patience and checkpoint best model. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

How to do it (practical):
1. Track best validation metric each epoch.
2. Stop when no improvement for `patience` epochs.
3. Restore and export the best checkpoint.

Example: A model registry plus run metadata lets teams trace exactly which model served production traffic.

Code:
```python
with torch.no_grad():
    val_pred = model(x_val)
val_loss = criterion(val_pred, y_val)
```

### Q19. Profile model performance

Use profiler tools (PyTorch profiler, Nsight), trace step time, kernel time, IO wait, memory.

Explanation: Use profiler tools (PyTorch profiler, Nsight), trace step time, kernel time, IO wait, memory. In real projects, explain assumptions, tradeoffs, and how you validate this with measurable metrics.

Example: If training is slow, profile dataloader wait time before changing model architecture.
