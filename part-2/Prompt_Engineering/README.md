# Prompt Engineering

Prompt engineering is the practice of designing prompts so a language model produces more useful, correct, and structured outputs.

## 1. Why Prompting Matters

Even with the same model, outputs can change dramatically depending on:

- wording
- role framing
- examples
- output constraints
- context quality

## 2. Common Prompting Styles

### Zero-Shot

Ask directly without examples.

### Few-Shot

Provide a few input-output examples first.

### Role Prompting

Tell the model what role to play.

Example:

```text
You are a senior Python reviewer.
```

### Structured Prompting

Constrain the output format.

Example:

```text
Return JSON with keys: summary, risks, next_steps
```

### Chain-of-Thought Style Prompting

Ask the model to reason step by step.

This can improve performance on multi-step tasks, though in product settings you may prefer hidden reasoning with visible structured answers.

### Retrieval-Augmented Prompting

Insert external context documents into the prompt so the model answers using retrieved information instead of only parametric memory.

## 3. Why Prompt Quality Changes Output

Prompts affect:

- clarity of task
- level of detail
- format reliability
- reasoning style
- safety and scope

## 4. Practical Example

Weak prompt:

```text
Explain transformers.
```

Stronger prompt:

```text
Explain transformers for a beginner in 5 bullet points and include one simple analogy.
```

Why the second prompt is better:

- it defines the audience
- it constrains output length
- it asks for a teaching device
- it reduces ambiguity

## 5. Prompt Template Design

A strong prompt often contains:

1. role or system framing
2. task definition
3. relevant context
4. output format constraints
5. edge-case instructions
6. evaluation criteria

Example template:

```text
You are an expert financial analyst.
Task: summarize the report below.
Audience: executive leadership.
Constraints: maximum 5 bullets, no jargon, mention 3 risks.
Output format: JSON with keys summary, risks, actions.
```

## 6. Python Example

```python
prompt = """
You are a helpful assistant.
Summarize the following text in 3 bullet points.
"""

print(prompt)
```

## 7. Chat Prompt Structure Example

```python
messages = [
    {"role": "system", "content": "You are a careful code reviewer."},
    {"role": "user", "content": "Review this Python function for bugs."},
]

for m in messages:
    print(m["role"], "=>", m["content"])
```

## 8. Prompting Patterns

- ask for steps
- ask for assumptions
- request a table
- request JSON
- provide examples
- specify audience and tone
- ask for citations when available
- request concise answers or detailed answers explicitly

## 9. Prompt Engineering for Different Tasks

### Classification

Tell the model exactly which labels are allowed.

### Summarization

Specify audience, length, and whether to preserve numbers.

### Code Generation

Specify language, style constraints, edge cases, and test expectations.

### Extraction

Demand machine-readable output such as JSON.

## 10. Common Failure Modes

- vague prompts
- conflicting instructions
- too much irrelevant context
- missing output schema
- hidden assumptions about domain knowledge
- asking for impossible certainty

## 11. Evaluation Matters

Prompt engineering is not just writing. It is experimental design.

You should compare prompts by:

- accuracy
- format compliance
- latency
- cost
- hallucination rate
- human preference

## 12. Few-Shot Example

```text
Input: "I loved the movie."
Output: positive

Input: "The plot was confusing."
Output: negative

Input: "The acting was excellent."
Output:
```

Few-shot prompting teaches the model the task pattern without changing weights.

## 13. Limits of Prompt Engineering

Prompting can improve behavior, but it cannot fully replace:

- missing knowledge
- poor model quality
- weak retrieval
- bad system design

That is why strong AI systems often combine:

- prompting
- retrieval
- tool use
- fine-tuning
- evaluation pipelines

## Summary

Prompt engineering is practical interface design for language models. Good prompts reduce ambiguity and improve usefulness, but they work best when combined with strong models and good system context.
