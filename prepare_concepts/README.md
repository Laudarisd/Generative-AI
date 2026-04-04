# Prepare Concepts - AI/ML Interview Q&A (General)

This folder contains a company-neutral, personal-info-removed interview preparation bank for AI/ML, deep learning, LLMs, time-series, systems, production, research, and leadership.

## Brief Explanation
This README is a structured interview-preparation guide.
It combines short conceptual answers, practical implementation thinking, and production-focused scenario questions.
Use this file for full-depth preparation, and use [Brief Q&A + Code Examples](BRIEF_QA.md) for fast revision.

## Notes
- No company-specific content is included.
- No personal names, university names, or personal-profile questions are included.
- Answers are written in a generalized interview-preparation style.

## Quick Navigation
- Fastest revision sheet: [Brief Q&A + Code Examples](BRIEF_QA.md)
- Start with Sections 1-3 for core interview prep.
- Use Sections 4-7 for coding, deep learning, and LLM questions.
- Use Sections 8, 14, and 16 for production and real-world scenarios.
- Use Sections 17 and 20 for reusable code templates.
- Use Sections 21-22 for final practice and revision.

---

## 1) General Interview Q&A

### 1. Describe your most impactful AI project
A strong example is leading a 2D-to-3D BIM generation system end-to-end. The work includes data pipeline design, annotation strategy, model architecture, loss design, deployment, and MLOps. A key challenge is geometric ambiguity (for example symmetric/square objects). Practical fixes include geometry-aware loss constraints and attention modules, which improve robustness on noisy real-world inputs.

### 2. How do you convert a real-world problem into an AI problem?
Start with domain understanding and objective definition. Translate into ML formulation (classification/regression/forecasting), define input-output contract, constraints (latency, cost, interpretability), and success metrics tied to business impact. Then design data, model, evaluation, and deployment plan.

### 3. What is the bias-variance tradeoff?
Bias is error from overly simple assumptions (underfitting). Variance is sensitivity to training data (overfitting). Better generalization requires balancing both through model capacity, regularization, data quality, and validation strategy.

### 4. What would you do if your model overfits?
Check leakage and split correctness first. Then apply regularization, simplify architecture, early stopping, augmentation, and better feature engineering. Use cross-validation and monitor train/validation gap.

### 5. How do you approach time-series forecasting?
Analyze trend/seasonality/autocorrelation; build lag/rolling/calendar features; use time-aware splits; choose model class (statistical, tree-based, RNN/Transformer/ESN); evaluate with horizon-aware metrics (MAE/RMSE/MAPE/sMAPE) and rolling backtests.

### 6. What is an Echo State Network (ESN)?
ESN is reservoir computing: recurrent reservoir weights are fixed, only readout is trained. It captures temporal dynamics with very cheap training and can be effective in low-latency time-series setups.

### 7. When choose a simpler model over a complex one?
When constraints are strict (latency, memory, explainability, maintainability) and simple models already meet target KPIs. Prefer simplest model that meets requirements with stable generalization.

### 8. How ensure model reliability in production?
Use strong pre-deployment validation (edge cases, stress tests) and post-deployment monitoring (drift, quality, latency, failures). Add alerts, rollback, retraining triggers, and runbooks.

### 9. How do you choose evaluation metrics?
Choose metrics based on problem type and error cost. For imbalance, precision/recall/F1/PR-AUC are often better than accuracy. For regression/forecasting, MAE/RMSE/MAPE depending on sensitivity to outliers and scale.

### 10. Challenges in deploying AI systems
Data quality/drift, train-serving skew, latency/scalability limits, integration complexity, observability gaps, and ongoing maintenance/retraining burden.

### 11. How do you handle data drift?
Monitor feature distributions and performance drift (PSI/KS/population shifts). Identify root cause, retrain with fresh representative data, recalibrate thresholds, and automate drift-response workflows.

### 12. What is regularization?
Techniques that reduce overfitting by constraining model complexity: L1/L2 penalties, dropout, early stopping, augmentation, and parameter sharing.

### 13. Limited labeled data: what do you do?
Use transfer learning, semi-supervised learning (pseudo-labeling), self-supervised pretraining, augmentation, weak supervision, and active learning for highest-value labeling.

### 14. Designing real-time AI systems
Define latency SLOs first, then optimize model (quantization/pruning/distillation), serving path (batching, caching, async pipelines), and infrastructure (edge/cloud split). Balance accuracy-latency-cost.

### 15. Describe a model failure and recovery
Common pattern: model strong offline, weak online due to distribution shift. Diagnose with data and feature drift analysis, fix preprocessing parity, retrain with representative production slices, and add monitoring/alerts.

### 16. Integrating domain knowledge
Inject domain constraints into features, architecture, loss terms, priors, and post-processing rules. Hybrid AI + physics/simulation models often improve reliability and interpretability.

### 17. Limitations of deep learning
Large data demand, high compute cost, lower interpretability, and fragility under distribution shift. Mitigate via model compression, better data curation, uncertainty estimation, and explainability tools.

### 18. Working with domain experts
Co-define goals, maintain shared vocabulary, translate ML outputs into domain terms, iterate through feedback loops, and align on measurable operational outcomes.

### 19. Large-scale training pipeline (PyTorch/JAX)
Optimize data IO (sharding/prefetch), compute (mixed precision), and scale (DDP/pmap/sharding). Keep sequence/window generation efficient and monitor throughput, memory, and utilization.

### 20. Low GPU utilization debugging
Profile first. Usually data pipeline bottleneck: tune `num_workers`, `pin_memory`, prefetch, serialization format, CPU transforms, and batch size. Use mixed precision where possible.

### 21. DataParallel vs DistributedDataParallel
`DataParallel` is easier but slower due to central bottleneck. `DistributedDataParallel` is preferred for real workloads: better scaling, less overhead, multi-node ready.

### 22. Adam vs SGD
Adam converges fast and is robust early. SGD+momentum often gives stronger final generalization at scale. Choose based on convergence speed vs final quality.

### 23. Gradient explosion/vanishing
Exploding gradients cause unstable updates; vanishing gradients block learning in early layers. Use clipping, initialization, residuals, gating (LSTM/GRU), normalization.

### 24. Transformers vs RNN/ESN
Transformers handle long-range dependencies and parallelize well. RNN/ESN can still win in low-latency, low-resource streaming settings.

### 25. `model.eval()` vs `torch.no_grad()`
`model.eval()` changes layer behavior (dropout/batchnorm). `torch.no_grad()` disables gradient tracking. Use both in inference.

### 26. Train loss down, validation loss up
Classic overfitting. Add regularization, better validation, early stopping, simpler model, or more representative data.

### 27. Efficient LLM fine-tuning
Use PEFT (LoRA/QLoRA), quantization, gradient checkpointing, accumulation, and high-quality curated data subsets.

### 28. Non-stationary time-series
Use differencing/transformations, rolling retraining, adaptive windows, and online monitoring for concept drift.

### 29. AI in industrial systems
Typical use-cases: anomaly detection, predictive maintenance, optimization, quality control, digital twins, and decision support.

### 30. Physical consistency in AI models
Enforce constraints in loss/architecture, validate against known laws, and combine model outputs with simulation/domain checks.

### 31. AI vs physics-model conflict
Investigate both sides: data quality, model assumptions, sensor errors, boundary conditions. Use real-world evidence and hybrid modeling when useful.

---

## 2) Core ML & Fundamentals

### Q1. Classification vs regression
Classification predicts discrete classes; regression predicts continuous values.

### Q2. What is cross-validation?
Repeated train/validation splits (for example k-fold) to estimate generalization more reliably.

### Q3. What is gradient descent?
Iterative optimization updating parameters opposite gradient direction to minimize loss.

### Q4. What is stochastic gradient descent?
Gradient descent using mini-batches; faster and noisier updates that often improve generalization.

### Q5. What is a loss function and how choose it?
A scalar objective measuring prediction error. Choose based on task semantics and error cost (CE for classification, MAE/RMSE/Huber for regression).

### Q6. Precision vs recall vs F1
Precision: correctness of positive predictions. Recall: coverage of actual positives. F1: harmonic mean balancing both.

### Q7. ROC-AUC
Area under ROC curve; ranking quality across thresholds.

### Q8. Data leakage
Any information from validation/test/future leaking into training, causing overly optimistic metrics.

### Q9. Feature scaling importance
Improves optimization stability/speed and prevents large-scale features from dominating.

### Q10. Normalization vs standardization
Normalization scales to fixed range (often [0,1]); standardization centers mean 0 and std 1.

### Q11. Curse of dimensionality
High-dimensional spaces become sparse; distance metrics degrade; data needs grow rapidly.

---

## 3) Statistics & Optimization

### Q1. Probability vs likelihood
Probability: data given parameters. Likelihood: parameters given observed data (up to proportionality).

### Q2. Maximum likelihood estimation
Choose parameters maximizing likelihood of observed data.

### Q3. Bayesian inference
Update prior beliefs with observed data to obtain posterior distribution.

### Q4. Expectation and variance
Expectation is average value; variance measures spread around expectation.

### Q5. Covariance vs correlation
Covariance measures joint variation (scale-dependent). Correlation is normalized covariance in [-1,1].

### Q6. p-value
Probability of observing data as extreme as current under null hypothesis; not probability that null is true.

### Q7. Hypothesis testing
Framework to assess evidence against null via test statistic, p-value, and significance threshold.

### Q8. Convex vs non-convex optimization
Convex has one global minimum structure; non-convex can have many local minima/saddles.

### Q9. Hessian matrix
Second-derivative matrix describing local curvature; helps understand conditioning and step behavior.

### Q10. Gradient clipping
Cap gradient norm/value to stabilize training and avoid exploding updates.

### Q11. Why normalization helps optimization
Improves conditioning, aligns feature scales, gives more stable gradient magnitudes.

### Q12. Saddle point
Critical point with mixed curvature directions; gradient near zero but not a minimum.

### Q13. Learning rate scheduling
Vary LR over training (step, cosine, warmup, one-cycle) for speed and stability.

### Q14. Early stopping
Stop training when validation performance stops improving to prevent overfitting.

### Q15. Calibration in ML
Alignment between predicted probabilities and actual event frequencies.

### Q16. Statistical anomaly detection (what is it?)
Detects data points or sequences that deviate significantly from expected statistical behavior (distribution, trend, or temporal pattern).

### Q17. Common anomaly detection methods
Z-score/IQR rules, Gaussian models, Isolation Forest, One-Class SVM, Autoencoders, and time-series residual-based detectors.

### Q18. Anomaly detection metrics
Use Precision, Recall, F1, PR-AUC, ROC-AUC, false alarm rate, detection delay, and event-level recall (not only point-level accuracy).

### Q19. Point anomaly vs contextual anomaly vs collective anomaly
Point anomaly: single unusual sample. Contextual anomaly: unusual under context (time/season). Collective anomaly: abnormal pattern over a sequence/window.

### Q20. Threshold selection for anomaly scores
Set thresholds using validation data, percentile rules, extreme value theory, or cost-based optimization for false positive vs false negative tradeoff.

### Q21. What is CUSUM?
CUSUM (Cumulative Sum Control Chart) is a change detection method that accumulates small deviations from a target mean to detect distribution shifts quickly.

### Q22. Why use CUSUM in monitoring?
It is sensitive to small persistent shifts that simple threshold alarms often miss.

### Q23. CUSUM vs EWMA
CUSUM is strong for fast detection of small sustained shifts; EWMA smooths noise and tracks gradual drift trends effectively.

### Q24. ARIMA (sometimes mistyped as RMIA)
ARIMA combines autoregression (AR), differencing (I), and moving average (MA) terms for univariate time-series forecasting.

### Q25. When ARIMA is useful vs not useful
Useful for structured linear time-series with moderate data. Less suitable for highly nonlinear multivariate systems without feature engineering.

### Q26. ARIMA for anomaly detection
Fit ARIMA, compute residuals, and flag anomalies where residuals exceed statistically justified bounds.

### Q27. Autoencoder-based anomaly detection
Train an autoencoder on normal data only. At inference, high reconstruction error indicates potential anomaly.

### Q28. Why autoencoders work for anomaly detection
They learn a compact manifold of normal patterns; out-of-distribution inputs reconstruct poorly.

### Q29. GAN-based anomaly detection (for example AnoGAN-style)
Train a GAN on normal data distribution and use generator/discriminator mismatch or reconstruction in latent space as anomaly score.

### Q30. CNN-based anomaly detection for signals
1D-CNNs are effective for vibration/sensor windows, capturing local temporal motifs and abrupt pattern changes.

### Q31. Local Outlier Factor (LOF)
LOF compares local density of a sample to that of neighbors. Lower relative density implies higher outlierness.

### Q32. One-Class SVM (OC-SVM)
OC-SVM learns a boundary around normal samples in feature space; points outside are marked anomalies.

### Q33. Robust Covariance / Elliptic Envelope
Assumes approximately Gaussian structure and flags low-probability points via robust Mahalanobis-distance style modeling.

### Q34. Isolation Forest vs LOF vs OC-SVM (quick comparison)
Isolation Forest scales well and isolates anomalies by random partitioning. LOF is local-density sensitive. OC-SVM can model nonlinear boundaries but is sensitive to kernel/scale choices.

### Q35. Event-based vs point-based anomaly evaluation
Point metrics score individual timestamps; event metrics score whether an anomalous event window was detected with acceptable delay.

### Q36. Foundation models for industrial anomaly detection
Pretrained multi-modal or time-series foundation models can provide stronger representations, then lightweight heads/adapters detect anomalies with less labeled data.

### Q37. Modular adaptation methods (foundation-model context)
A practical approach is frozen pretrained backbone + small task-specific adapter head for quick domain adaptation and robust deployment updates.

---

## 4) PyTorch / Coding / Systems

### Q1. Autograd in PyTorch
Automatic differentiation engine building computational graph and computing gradients via backprop.

### Q2. Computational graph
Directed graph of tensor operations used to compute outputs and gradients.

### Q3. `model.train()` vs `model.eval()`
`train()` enables training-time behavior (dropout/bn updates). `eval()` freezes inference behavior.

### Q4. `torch.no_grad()`
Context manager disabling gradient tracking to save memory/compute.

### Q5. DataLoader
Batches, shuffles, parallel-loads dataset samples for efficient training loops.

### Q6. Backpropagation
Applies chain rule from loss to parameters to compute gradients.

### Q7. Gradient accumulation
Accumulate gradients over multiple mini-batches before optimizer step to emulate larger batch size.

### Q8. Reduce GPU memory usage
Mixed precision, smaller batches, gradient checkpointing, sequence truncation, activation recomputation, optimizer/state choices.

### Q9. Debug NaNs in training
Check inputs/labels, LR, loss scale, division/log operations, exploding grads; enable anomaly detection.

### Q10. Mixed precision
Use FP16/BF16 for faster compute and lower memory with loss scaling when needed.

### Q11. Checkpointing
Save model/optimizer/scheduler/scaler states for recovery and reproducibility.

### Q12. DDP
Multi-process distributed training with gradient all-reduce.

### Q13. DataParallel
Single-process multi-GPU split with central gather; simpler but less scalable.

### Q14. PyTorch vs TensorFlow vs JAX
PyTorch: flexible/eager ecosystem. TensorFlow: strong production tooling. JAX: functional style + strong compiler transformations.

### Q15. When to use JAX
When vectorization/JIT/XLA and functional transformations (`jit`, `vmap`, `pmap`) are major advantages.

### Q16. JIT compilation
Compile computation graphs for optimized execution.

### Q17. Optimize slow training pipeline
Profile data + compute + communication; remove bottlenecks one by one.

### Q18. Handle large datasets
Sharding, streaming, memory mapping, prefetching, distributed sampling, feature stores.

### Q19. Profile model performance
Use profiler tools (PyTorch profiler, Nsight), trace step time, kernel time, IO wait, memory.

### Q20. Batching importance
Improves throughput and gradient stability; better hardware utilization.

---

## 5) Deep Learning

### Q1. CNN
Neural network using convolutions for spatial feature extraction.

### Q2. RNN
Sequence model with recurrent state passing through time.

### Q3. LSTM vs GRU
Both gated RNNs; GRU is simpler/faster, LSTM has separate cell state and can be more expressive.

### Q4. Transformer
Attention-based architecture enabling parallel sequence modeling.

### Q5. Attention mechanism
Computes weighted context from key-query similarity.

### Q6. Vanishing gradient
Gradients shrink through depth/time, slowing learning.

### Q7. Exploding gradient
Gradients grow excessively, causing instability.

### Q8. Dropout
Randomly zero activations during training to reduce co-adaptation.

### Q9. Batch normalization
Normalizes intermediate activations to stabilize/accelerate training.

### Q10. Residual connection
Skip connection easing optimization of deep networks.

### Q11. Why transformers are powerful
Long-range dependency modeling + parallelization + scaling behavior.

### Q12. Transfer learning
Reuse pretrained representations for new tasks.

### Q13. Fine-tuning
Continue training pretrained model on target data.

### Q14. Self-supervised learning
Learn representations from unlabeled data via pretext/objective construction.

### Q15. Layer normalization
Normalizes activations across feature dimensions per sample, making training stable without relying on batch statistics.

### Q16. BatchNorm vs LayerNorm (when to use which)
BatchNorm is usually best in CNN workloads with stable batch size. LayerNorm is preferred for Transformers and variable-length sequence models.

---

## 6) Time-Series

### Q1. Stationarity
Statistical properties remain stable over time.

### Q2. Autocorrelation
Correlation of a series with lagged versions of itself.

### Q3. Seasonality
Recurring periodic patterns.

### Q4. ARIMA
Autoregressive integrated moving average model for univariate forecasting.

### Q5. ESN vs RNN
ESN trains only readout (faster), RNN trains full recurrence (more flexible but heavier).

### Q6. Forecasting horizon
Future time span being predicted.

### Q7. Sliding window
Transform sequential data into supervised samples with rolling input windows.

### Q8. Evaluate time-series models
Use walk-forward backtesting and horizon-aware metrics; avoid random splits.

---

## 7) LLM / Modern AI

### Q1. LoRA
Low-rank adapters train small matrices instead of full model weights.

### Q2. QLoRA
LoRA over quantized base model for lower memory training.

### Q3. Fine-tuning vs prompt tuning
Fine-tuning updates parameters; prompt tuning optimizes prompts/soft tokens with fewer trainable params.

### Q4. Embedding
Dense vector representation of text/items capturing semantic similarity.

### Q5. Tokenization
Convert text into model-consumable token IDs.

### Q6. Hallucination
Confident but incorrect generated content.

### Q7. Reduce hallucination
RAG, better prompts, constrained decoding, tool use, verification, and fine-tuning on reliable data.

### Q8. RAG
Retrieve relevant documents and condition generation on retrieved context.

### Q9. Vector database
Index/store embeddings for similarity search at scale.

### Q10. Efficient LLM deployment
Quantization, distillation, KV-cache, batching, speculative decoding, optimized serving stack.

### Q11. Encoder vs decoder (LLM perspective)
Encoder-focused models are strong for understanding tasks; decoder-focused models are strong for generation tasks.

### Q12. Encoder-only vs decoder-only vs encoder-decoder
Encoder-only for classification/retrieval, decoder-only for text generation, encoder-decoder for sequence-to-sequence tasks.

### Q13. Causal masking
Decoder attention mask that blocks future tokens so generation stays autoregressive.

### Q14. Cross-attention
Decoder attends to encoder outputs in encoder-decoder models, enabling conditioned generation.

### Q15. Perplexity
`exp(cross_entropy)`; lower values indicate better average next-token prediction.

---

## 8) Industrial AI (General)

### Q1. Digital twin
Virtual representation of physical assets/processes continuously updated from data.

### Q2. Integrate AI into engineering systems
Map use-case to workflow, ensure data interfaces, establish reliability and override/fallback mechanisms.

### Q3. Simulation + real data
Pretrain on simulation, fine-tune/calibrate on real data, and domain-adapt carefully.

### Q4. Predictive maintenance
Forecast failure risk/RUL from sensor history to schedule interventions proactively.

### Q5. Anomaly detection in sensor data
Combine statistical baselines + ML detectors + rule checks, with human-in-the-loop triage.

### Q6. Ensure physical consistency
Constraint-aware training, physics-informed losses, and post-hoc rule validation.

### Q7. Validate model in production
Shadow mode, canary rollout, KPI monitoring, drift detection, and rollback plans.

### Q8. Real-time deployment
Low-latency model, streaming pipeline, bounded inference path, and resilient serving.

### Q9. Safety concerns
False negatives in critical events, automation bias, cyber risks, bad feedback loops, and weak fail-safe design.

### Q10. Unreliable sensors
Imputation, sensor health scoring, redundancy, robust filtering, and uncertainty-aware outputs.

### Q11. Surrogate modeling
Train fast approximator for expensive simulation.

### Q12. Optimize industrial processes
Use forecasting + optimization + control under operational constraints.

### Q13. Robustness in harsh environments
Train on diverse conditions, stress test extensively, and include fallback/alert logic.

### Q14. Scale AI in large systems
Standardized MLOps, shared feature/model services, automated monitoring/retraining.

---

## 9) Research & Innovation

### Q1. Read papers efficiently
Read abstract/figures/conclusion first, then method and experiments with focused notes.

### Q2. Evaluate new method
Check assumptions, baseline fairness, ablations, statistical significance, and real-world constraints.

### Q3. Ablation study
Systematic removal/change of components to measure each component’s contribution.

### Q4. Reproducibility
Ability to replicate results using provided code/data/settings/seeds.

### Q5. Turn research into product
Simplify method, improve robustness, define SLAs, and build monitoring/deployment path.

### Q6. What makes research impactful
Novelty + strong evidence + reproducibility + practical relevance.

### Q7. Design experiments
Start from hypothesis, control confounders, choose meaningful metrics, predefine protocol.

### Q8. Compare models fairly
Same data splits, compute budget, tuning effort, and evaluation rules.

### Q9. Novelty in research
New idea, new evidence, or new capability beyond existing state of the art.

### Q10. Research contribution
Clear problem framing, measurable improvement, and transparent analysis of tradeoffs.

---

## 10) Mathematical/Theoretical (Hard)

### Q1. L2-regularized linear regression update rule
For loss `J(w)= (1/N)||Xw-y||^2 + lambda||w||^2`, gradient is `(2/N)X^T(Xw-y)+2lambda w`; update: `w <- w - eta * grad`.

### Q2. Why L2 shrinks weights but not zero
L2 applies continuous proportional shrinkage; unlike L1, it does not create sharp sparsity-inducing corners at zero.

### Q3. Ill-conditioned Hessian impact
Optimization zig-zags and converges slowly; sensitive to LR. Fix with normalization, preconditioning, adaptive optimizers.

### Q4. Why normalization improves convergence mathematically
It reduces anisotropy of curvature (better condition number), so gradient steps are more uniformly effective.

### Q5. How gradient clipping stabilizes exploding gradients
It bounds update magnitude so recurrent/deep chains cannot produce destructive parameter jumps.

### Q6. Eigenvalues and ESN stability
Reservoir dynamics remain stable when effective spectral radius is controlled (typically < 1 in many settings).

### Q7. Why spectral radius matters in recurrent nets
It governs memory decay/amplification over time and thus stability vs expressiveness.

### Q8. Bias-variance decomposition
Expected test error = irreducible noise + bias^2 + variance (for squared loss setting).

### Q9. Why cross-entropy over MSE in classification
Cross-entropy aligns with probabilistic likelihood and gives stronger gradients for confident wrong predictions.

### Q10. KL divergence and usage
Measure of distribution mismatch; used in VAEs, distillation, calibration, and drift comparison.

---

## 11) Optimization & Debugging (Hard Practical)

### Q1. Model outputs NaNs: step-by-step
Check data/labels, isolate first NaN layer, lower LR, inspect gradient norms, verify numerically unstable ops (`log`, division), enable anomaly detection, and test mixed-precision settings.

### Q2. Training stable but very slow
Profile data pipeline, GPU kernels, communication; optimize batching, mixed precision, dataloader, kernels, and distributed setup.

### Q3. Loss oscillates heavily
Likely LR too high, bad normalization, noisy batches, or unstable objective. Use lower LR, scheduler, gradient clipping, larger batch.

### Q4. Trained long but random performance
Possible label mismatch, bug in preprocessing, leakage in validation logic, incorrect target mapping, or frozen gradients.

### Q5. Gradient norms spike
Inspect recent batches/outliers, reduce LR, clip gradients, stabilize architecture/loss.

### Q6. Validation metric fluctuates heavily
High variance data/small validation set/distribution shift. Increase validation size, smooth reporting, use repeated runs.

### Q7. Diagnose underfitting vs overfitting from logs
Underfitting: both train/val poor. Overfitting: train good, val poor with widening gap.

### Q8. Converges but wrong predictions
Objective-metric mismatch, thresholding issues, label noise, or train-serving skew.

### Q9. 10x more features than samples
Regularize strongly, feature selection, dimensionality reduction, sparse models, and robust cross-validation.

### Q10. Good offline, bad production
Data drift, schema mismatch, missing features, latency constraints, feedback loops, monitoring gaps.

---

## 12) Coding + System Design (Hard)

### Q1. Mixed-precision training loop (PyTorch)
Use `torch.cuda.amp.autocast()` and `GradScaler` around forward/loss/backward/step/update.

### Q2. Implement gradient accumulation
Scale loss by accumulation steps, call backward each mini-batch, optimizer step every k steps.

### Q3. Implement custom loss
Subclass `nn.Module` or write function using tensor ops, ensuring stable numerics.

### Q4. Variable-length sequences efficiently
Pad + mask, packed sequences, bucketing by length, or attention masks.

### Q5. Avoid memory leaks
Clear references, avoid storing graph tensors, use `detach()` where needed, and monitor retained objects.

### Q6. Debug slow DataLoader
Profile worker time, serialization overhead, transforms, storage format, and host-device transfer.

### Q7. Design multi-GPU training
Use DDP, distributed sampler, gradient all-reduce, and rank-aware checkpointing/logging.

### Q8. Implement early stopping
Track best validation metric with patience and checkpoint best model.

### Q9. Design experiment tracking
Log configs, data/version hash, metrics, artifacts, model registry, and reproducible seeds.

### Q10. Deploy PyTorch model
Export/serve with TorchScript/ONNX/Triton/FastAPI pipeline with observability and rollback.

---

## 13) LLM Advanced

### Q1. Attention math
`Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V`.

### Q2. Why LLMs scale with data
Large models with large diverse data learn transferable representations and in-context capabilities.

### Q3. Context length vs compute tradeoff
Attention cost grows roughly quadratically with sequence length in standard transformers.

### Q4. Evaluate LLM quality
Task metrics + human eval + factuality/safety/latency/cost evaluations.

### Q5. Catastrophic forgetting
New fine-tuning data overwrites old capabilities; mitigate with PEFT, rehearsal, balanced data.

### Q6. Align outputs with domain constraints
Use constrained prompts, tool use, retrieval, guardrails, and policy checks.

---

## 14) Real-Time & Reliability

### Q1. Design real-time anomaly detection
Streaming ingestion -> feature extraction -> low-latency model -> thresholding -> alerting -> feedback loop.

### Q2. Meet strict latency constraints
Optimize model size, runtime, batching, hardware placement, and avoid slow synchronous dependencies.

### Q3. Fallback if AI fails
Rule-based backup, safe defaults, circuit breaker, and human escalation.

### Q4. Ensure 24/7 reliability
Redundancy, health checks, autoscaling, SLO monitoring, and on-call runbooks.

### Q5. Handle streaming data
Windowed processing, out-of-order handling, watermarking, and state management.

### Q6. Scale to millions of points
Partitioned pipelines, distributed stream processors, and efficient online feature stores.

### Q7. Design alert thresholds
Risk-based thresholds, precision/recall tradeoffs, dynamic baselines, and escalation tiers.

### Q8. Handle delayed data
Buffering, event-time processing, late-arrival correction, and re-computation policies.

### Q9. Monitoring pipeline
Monitor input quality, drift, model outputs, latency, errors, and business KPIs.

### Q10. Debug production failure
Triage impact, isolate component, rollback if needed, run RCA, and patch with tests.

---

## 15) Leadership & Behavioral (Generalized)

### Q1. Handling conflict
Clarify goals, align on facts, discuss tradeoffs, and converge on decision criteria.

### Q2. Mentoring juniors
Set clear expectations, pair regularly, provide actionable feedback, and grow ownership gradually.

### Q3. Handling failure
Acknowledge quickly, analyze root cause, communicate transparently, and prevent recurrence.

### Q4. Prioritizing multiple deadlines
Use impact-risk-effort framework and align with stakeholders on sequence.

### Q5. Unclear requirements
Run discovery, define assumptions, propose milestones, and iterate with feedback.

### Q6. Communicating to non-technical teams
Use simple language, visuals, and business-impact framing.

### Q7. Difficult decision making
Define constraints, evaluate options quantitatively, document rationale, and monitor outcomes.

### Q8. Ensuring team productivity
Clear goals, unblock dependencies early, and enforce lightweight execution rituals.

### Q9. Giving feedback
Specific, timely, respectful, behavior-focused, with clear next actions.

### Q10. Leadership style
Context-driven, collaborative, quality-focused, and outcome-oriented.

---

## 16) Ultra-Hard Scenarios

### Q1. AI predicts failure but engineer disagrees
Review evidence together, compare with sensor history/physics checks, run targeted validation, then decide with safety-first policy.

### Q2. Physically impossible model result
Add constraint checks, retrain with physics-informed loss/features, and block unsafe predictions in serving layer.

### Q3. System causes financial loss
Stabilize system first (rollback/disable), communicate impact, perform RCA, and add controls.

### Q4. No labeled data
Use self-supervised/unsupervised methods, weak supervision, synthetic labels, and active learning.

### Q5. Huge data but poor performance
Likely data quality, objective mismatch, feature issues, or leakage/shift; scale alone does not fix bad signal.

### Q6. Works in lab but fails in field
Domain gap, noisy sensors, unseen operating regimes, and fragile assumptions.

### Q7. Deploy in 2 days
Use simplest reliable baseline, strict guardrails, shadow/canary rollout, and clear rollback.

### Q8. Sudden data distribution change
Trigger drift alerts, switch to safe mode, retrain/recalibrate quickly, and monitor recovery.

### Q9. Explainable AI for regulator
Use interpretable models where possible, local/global explanations, documentation, and audit trails.

### Q10. Detect/fix model bias
Measure subgroup metrics, identify bias sources, rebalance data/objective, and monitor fairness continuously.

---

## 17) Quick Coding Templates (Short)

### Full PyTorch training loop skeleton
```python
model.train()
for batch in train_loader:
    x, y = [t.to(device) for t in batch]
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        pred = model(x)
        loss = criterion(pred, y)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### Early stopping skeleton
```python
best = float("inf")
patience, wait = 10, 0
for epoch in range(max_epochs):
    train_one_epoch()
    val = validate()
    if val < best:
        best, wait = val, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            break
```

### Gradient accumulation skeleton
```python
optimizer.zero_grad(set_to_none=True)
for i, (x, y) in enumerate(loader):
    with torch.cuda.amp.autocast(enabled=use_amp):
        loss = criterion(model(x), y) / accum_steps
    scaler.scale(loss).backward()
    if (i + 1) % accum_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

---

## 18) Extended Question Coverage Map (for repeated/duplicate prompts)
All repeated questions in the full list (coding sections A-E, domain sections A-D, system design A-D, hard practical blocks) are covered by Sections **2-17** above. Use this mapping:

- Core theory duplicates -> Sections 2, 3, 10
- Optimization/debugging duplicates -> Sections 4, 11, 12
- PyTorch implementation duplicates -> Sections 4, 12, 17
- JAX questions -> Sections 4 (Q14-Q16) and 12
- Time-series duplicates -> Sections 1, 6, 8
- Physics/industrial/digital-twin duplicates -> Sections 1, 8, 14, 16
- LLM/RAG/serving duplicates -> Sections 7, 13, 14
- Production/MLOps/system-design duplicates -> Sections 8, 12, 14
- Leadership/senior behavior duplicates -> Section 15

This keeps one clean answer per concept while still covering the full question bank.

---

## 19) Advanced Additions (Missing Questions + Examples)

### Q1. Encoder vs Decoder in Transformers
An encoder builds contextual representations from input tokens (bidirectional context in encoder-only models). A decoder generates output token-by-token, using causal masking and optional cross-attention to encoder outputs.

Example:
- Encoder-style use: classification, embedding, retrieval.
- Decoder-style use: text generation, chat completion.
- Encoder-decoder use: translation, summarization.

### Q2. Encoder-only vs Decoder-only vs Encoder-Decoder
- Encoder-only (for example BERT): strong understanding tasks.
- Decoder-only (for example GPT-style): strong generation tasks.
- Encoder-decoder (for example T5): mapping input sequence to output sequence.

### Q3. What is causal masking?
A decoder mask that prevents each token from attending to future tokens, preserving autoregressive generation.

### Q4. What is cross-attention?
In encoder-decoder models, decoder queries attend to encoder keys/values so output is conditioned on source input.

### Q5. Optimizer vs Activation Function
- Optimizer decides how parameters are updated (SGD, AdamW).
- Activation decides nonlinear transformation inside the network (ReLU, GELU, SiLU).

Rule of thumb:
- Optimizer affects learning dynamics and convergence.
- Activation affects representational power and gradient flow.

### Q6. Adam vs AdamW
AdamW decouples weight decay from gradient updates and usually gives better regularization behavior in modern deep learning.

### Q7. BatchNorm vs LayerNorm
- BatchNorm normalizes across batch dimension; works very well in CNNs with stable batch sizes.
- LayerNorm normalizes across feature dimension per sample; preferred in Transformers and variable-length sequence settings.

### Q8. When BatchNorm can fail
Very small batches, non-iid batch composition, or highly variable sequence workloads can make batch statistics noisy.

### Q9. Why LayerNorm in Transformers
It is independent of batch statistics and stable for sequence modeling and distributed setups with varying micro-batches.

### Q10. Pre-LN vs Post-LN Transformer blocks
- Pre-LN: normalize before sublayer, often easier optimization for deep transformers.
- Post-LN: original formulation, can be less stable at scale.

### Q11. Weight Decay vs Dropout
- Weight decay constrains parameter magnitude.
- Dropout stochastically removes activations during training.
They regularize differently and are often combined.

### Q12. Gradient Clipping: by value vs by norm
- By value clips each gradient element independently.
- By norm rescales full gradient vector to max norm.
Norm clipping is usually preferred for deep sequence models.

### Q13. Learning Rate Warmup
Start with a small LR and gradually increase early in training to avoid unstable updates, especially in Transformers.

### Q14. Label Smoothing
Replace hard one-hot targets with softened targets to improve calibration and reduce overconfidence.

### Q15. Teacher Forcing
Train decoder by feeding ground-truth previous token; speeds convergence but can create train-test mismatch.

### Q16. Exposure Bias
Mismatch between training (teacher forcing) and inference (model-generated history), causing compounding generation errors.

### Q17. Perplexity
`PPL = exp(cross_entropy)`; lower perplexity means better average next-token prediction.

### Q18. Top-k vs Top-p sampling
- Top-k: sample from k highest-probability tokens.
- Top-p: sample from smallest token set whose cumulative probability >= p.
Top-p is often more adaptive.

### Q19. Temperature in generation
Scales logits before softmax. Low temperature makes output conservative; high temperature increases diversity.

### Q20. Beam Search vs Sampling
Beam search optimizes likely sequences (less diverse). Sampling gives more variety and is common for open-ended generation.

### Q21. Positional Encoding vs Learned Positional Embeddings
Sinusoidal encoding is deterministic and extrapolation-friendly; learned positional embeddings can fit better in-domain but may extrapolate less.

### Q22. KV Cache in LLM Inference
Caches previous keys/values to avoid recomputing attention over old tokens, reducing autoregressive latency.

### Q23. Context Window Saturation
As context grows, compute and memory rise; long irrelevant context can reduce answer quality. Retrieval and context pruning help.

### Q24. Prompt Injection (RAG security)
Adversarial instructions in retrieved content can override behavior. Defend with source filtering, policy checks, and tool-guardrails.

### Q25. Quantization: PTQ vs QAT
- PTQ (post-training quantization): fast, minimal retraining.
- QAT (quantization-aware training): better accuracy retention, more effort.

### Q26. Distillation
Train smaller student model to mimic teacher outputs; improves deployment efficiency.

### Q27. Model Parallelism vs Data Parallelism
- Data parallelism splits data across replicas.
- Model parallelism splits model across devices.
Large LLMs often use both.

### Q28. FSDP / ZeRO (why needed)
Shard parameters/gradients/optimizer states to train models that do not fit on one GPU.

### Q29. Throughput vs Latency
Throughput is requests per second; latency is time per request. Optimizing one may hurt the other.

### Q30. Calibration vs Accuracy
A model can be accurate but poorly calibrated; decision systems often need both.

---

## 20) Practical Mini Examples

### Example A: BatchNorm vs LayerNorm in PyTorch
```python
import torch.nn as nn

cnn_block = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
)

transformer_ffn = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
    nn.LayerNorm(512),
)
```

### Example B: Optimizer choice (AdamW vs SGD)
```python
import torch

adamw = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sgd = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
```

### Example C: Gradient clipping + mixed precision
```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### Example D: LayerNorm in a Transformer-style block
```python
import torch.nn as nn

class TinyBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x):
        return x + self.ff(self.ln1(x))
```

### Example E: Decoder causal mask (PyTorch)
```python
import torch

T = 8
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # True means masked future positions
```

### Example F: Top-k / Top-p sampling pseudocode
```python
# logits -> apply temperature -> filter by top-k or top-p -> sample next token
```

---

## 21) Extra High-Value Questions to Practice

### Optimization & Training
- How do you pick batch size under fixed GPU memory?
- When should you use cosine scheduler vs one-cycle scheduler?
- How do you choose weight decay for Transformers?
- What are signs that warmup is too short or too long?
- How do you tune gradient clipping threshold?

### Architecture & Modeling
- Why do residual connections help optimization depth?
- When to use GELU vs ReLU?
- What is SwiGLU and why used in modern LLMs?
- What is RMSNorm and how is it different from LayerNorm?
- When should you use Mixture-of-Experts (MoE)?

### LLM Systems
- How do you chunk documents for RAG effectively?
- How do you evaluate retrieval quality separately from generation quality?
- What are common causes of hallucination in long-context prompts?
- How do you design guardrails for tool-calling agents?
- How do you do cost-aware LLM routing across model sizes?

### Production & Reliability
- How do you detect silent model degradation with no labels?
- How do you run safe canary deployment for ML models?
- What should be in a model card for regulated environments?
- How do you define rollback criteria before deployment?
- What is the minimum monitoring dashboard for online inference?

### Time-Series / Industrial
- How do you detect concept drift vs sensor fault?
- How do you estimate prediction uncertainty for maintenance decisions?
- How do you set alert hysteresis to avoid alarm flapping?
- How do you evaluate anomaly detector lead time?
- How do you choose retraining cadence for seasonal systems?

---

## 22) Fast Revision Checklist

- Can explain encoder/decoder/cross-attention clearly.
- Can explain BatchNorm vs LayerNorm with usage context.
- Can justify optimizer and activation choices.
- Can debug NaNs, slow training, and unstable loss step-by-step.
- Can design production monitoring, drift handling, and rollback.
- Can discuss LLM inference efficiency (KV cache, quantization, batching).
- Can discuss reliability and safety for real-world systems.
