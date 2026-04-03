# Fine-Tuning and Adaptation

Fine-tuning means adapting a pretrained model to a narrower task, domain, format, or behavior target.

A pretrained base model may have strong general knowledge, but it is not automatically optimized for:

- your company documents
- your product tone
- your label schema
- your coding conventions
- your domain vocabulary
- your desired output format

Fine-tuning changes the model so these requirements are reflected in the weights rather than being repeated only in prompts.

## 1. Why Fine-Tune

A base model may know general language patterns, but still fail on:

- your domain terminology
- your desired response format
- your company documents
- your coding style
- your label schema

Fine-tuning changes model weights so the model internalizes those requirements.

## 2. Common Goals of Fine-Tuning

Teams fine-tune models for different reasons:

- supervised task performance such as classification or extraction
- instruction following quality
- output formatting reliability
- domain adaptation for legal, medical, finance, or code
- reducing prompt size for repeated tasks
- improving accuracy on specialized data

## 3. Main Fine-Tuning Styles

### Full Fine-Tuning

Update all model parameters.

Advantages:

- maximum flexibility
- strongest adaptation potential
- good for major domain shifts if compute is available

Disadvantages:

- expensive GPU memory cost
- large checkpoint storage
- slower iteration
- harder to reproduce cheaply

### Parameter-Efficient Fine-Tuning (PEFT)

Update only a small subset of additional parameters.

Advantages:

- cheaper than full fine-tuning
- smaller checkpoints
- often enough for many tasks
- easier to maintain multiple adapters for different tasks

### Instruction Tuning

Train on instruction-response examples so the model learns how to follow task requests in a chat-like format.

### Supervised Fine-Tuning (SFT)

SFT is one of the most common practical forms of fine-tuning. You provide examples of desired outputs, and the model is trained by teacher forcing to predict the correct next token sequence.

### Preference Alignment

Train using pairwise preferences or reward models to shape behavior after SFT. Examples include RLHF-style pipelines and direct preference optimization methods.

## 4. What SFT Actually Does

During SFT, the model sees a prompt and the target answer. The loss is usually standard next-token cross-entropy over the target response tokens.

If the training example is:

```text
Instruction: Summarize this paragraph.
Response: The paragraph explains how transformers use attention.
```

then the model learns to increase the probability of the correct response tokens.

Mathematically, if the target sequence is $y_1, y_2, \dots, y_T$, the objective is:

```math
\mathcal{L}_{SFT} = - \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)
```

where $x$ is the conditioning prompt.

## 5. LoRA

LoRA means Low-Rank Adaptation.

Instead of updating a full weight matrix $W$, LoRA learns a low-rank update:

```math
W' = W + BA
```

where:

- $A \in \mathbb{R}^{r \times d}$
- $B \in \mathbb{R}^{k \times r}$
- $r$ is a small rank

Because $r$ is much smaller than the full dimension, trainable parameter count drops sharply.

## 6. Why LoRA Helps

If a full matrix is $4096 \times 4096$, the dense update has over 16 million parameters.

If LoRA rank is 16, the trainable update is approximately:

```math
4096 \times 16 + 16 \times 4096 = 131{,}072
```

which is dramatically smaller.

That is why LoRA is attractive for limited-GPU setups.

## 7. QLoRA

QLoRA combines:

- low-rank adapters
- quantized base weights

The base model is kept in low precision, often 4-bit, while adapters are trained in higher precision. This makes large-model adaptation possible on much smaller hardware than full fine-tuning.

## 8. Quantization Basics

Quantization stores values with fewer bits.

Common choices include:

- FP16
- BF16
- INT8
- 4-bit quantization

Lower precision reduces memory use and sometimes increases throughput, but it can hurt quality if done poorly.

## 9. SFT Workflow End to End

A practical supervised fine-tuning workflow looks like:

1. choose a base model
2. define task format and prompt template
3. prepare instruction-response data
4. tokenize inputs and outputs
5. choose method: full tuning, LoRA, or QLoRA
6. set learning rate, batch size, and max length
7. train and save checkpoints
8. evaluate on held-out examples
9. compare against the base model and prompt-only baseline
10. deploy adapters or merged weights

## 10. Example Training Data Formats

### Instruction Format

```json
{
  "instruction": "Explain overfitting in simple language.",
  "input": "",
  "output": "Overfitting happens when a model memorizes the training data..."
}
```

### Chat Format

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise tutor."},
    {"role": "user", "content": "What is gradient descent?"},
    {"role": "assistant", "content": "Gradient descent is an optimization method..."}
  ]
}
```

## 11. Minimal Dataset Build Example

```python
samples = [
    {
        "instruction": "Classify the sentiment",
        "input": "The movie was excellent",
        "output": "positive",
    },
    {
        "instruction": "Classify the sentiment",
        "input": "The movie was boring",
        "output": "negative",
    },
]

for s in samples:
    print(s)
```

## 12. Prompt Formatting for SFT

```python
def format_example(example):
    return (
        f"Instruction: {example['instruction']}\n"
        f"Input: {example['input']}\n"
        f"Response: {example['output']}"
    )

print(format_example(samples[0]))
```

## 13. Tokenization Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = format_example(samples[0])
encoded = tokenizer(text, truncation=True, padding=False)

print(encoded["input_ids"][:20])
print("token count:", len(encoded["input_ids"]))
```

## 14. PEFT with Hugging Face Style Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"],
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

## 15. QLoRA-Style Loading Example

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
)
```

## 16. Full SFT Training Script Skeleton

```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

samples = [
    {"text": "Instruction: Explain attention\nResponse: Attention lets tokens weigh other tokens."},
    {"text": "Instruction: Define LoRA\nResponse: LoRA is low-rank adaptation for efficient fine-tuning."},
]

dataset = Dataset.from_list(samples)

def tokenize_fn(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = dataset.map(tokenize_fn, batched=True)

args = TrainingArguments(
    output_dir="./sft_outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=10,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

print("ready to train")
# trainer.train()
```

## 17. LoRA SFT Skeleton

```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["c_attn"])
model = get_peft_model(model, lora_config)

train_data = Dataset.from_list([
    {"text": "Instruction: Explain RAG\nResponse: RAG combines retrieval and generation."},
    {"text": "Instruction: Explain QLoRA\nResponse: QLoRA combines quantization with low-rank adapters."},
])

def preprocess(batch):
    out = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    out["labels"] = out["input_ids"].copy()
    return out

train_data = train_data.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./lora_outputs",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=1,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
print("ready for LoRA SFT")
```

## 18. Important Hyperparameters

Key choices include:

- learning rate
- batch size
- gradient accumulation
- max sequence length
- number of epochs
- warmup ratio
- weight decay
- LoRA rank
- LoRA target modules

A too-large learning rate can destroy pretrained knowledge quickly.

## 19. Full Fine-Tuning vs LoRA vs QLoRA

| Method | Trainable Scope | Memory Need | Typical Use |
| --- | --- | --- | --- |
| Full fine-tuning | all weights | highest | major adaptation when compute is available |
| LoRA | adapter weights only | low | efficient domain or task adaptation |
| QLoRA | quantized base + adapters | very low | adapting larger models on limited GPUs |

## 20. From Scratch vs Fine-Tuning

### From Scratch

Train all parameters from random initialization.

You need:

- massive data
- major compute budget
- tokenizer design
- stable optimization

### Fine-Tuning

Start from a pretrained base model.

You need much less data and compute.

That is why most teams fine-tune instead of training from scratch.

## 21. Evaluation Questions

After fine-tuning, ask:

- did task accuracy improve?
- did hallucination rate change?
- did format compliance improve?
- did general behavior regress?
- did safety behavior degrade?
- did latency or memory footprint change?

## 22. Common Failure Modes

- overfitting to a small dataset
- catastrophic forgetting
- data leakage
- bad prompt formatting during training
- poor label quality
- too high learning rate
- wrong masking of labels
- inconsistent chat templates

## 23. When Retrieval Is Better Than Fine-Tuning

Fine-tuning is not always the answer.

If the problem is mainly factual freshness or access to private documents, retrieval may be better than modifying model weights.

## 24. Practical Mental Model

Use:

- SFT when you want the model to answer in a desired style or task format
- LoRA when you want cheap adaptation
- QLoRA when the base model is too large for standard full-precision tuning
- retrieval when the challenge is external knowledge, not behavior learning

## Summary

Fine-tuning is the main bridge from generic models to specialized systems. SFT is the core starting point, and PEFT methods such as LoRA and QLoRA made that bridge much cheaper. In modern LLM engineering, adaptation usually means choosing the smallest and safest training method that actually solves the problem.
