# Chatbot with LLM

This subchapter focuses on one of the most common LLM applications: building a chatbot that can answer user questions across multiple turns.

## 1. What a Chatbot Is

A chatbot is a system that receives user messages, keeps conversational context, and produces a response.

In modern LLM systems, the chatbot is usually built on top of:

- a base language model
- a prompt or system message
- chat history handling
- optional retrieval or tool use
- output post-processing

## 2. Minimal Chat Loop

A simplified loop is:

1. user sends a message
2. system builds a prompt from history and instructions
3. model predicts the next response tokens
4. application returns the generated answer
5. history is updated for the next turn

## 3. Why Context Management Matters

Chatbots are not just about generation quality. They also need:

- memory management
- truncation strategy for long conversations
- prompt safety
- tool integration
- evaluation of helpfulness and correctness

## 4. Basic Python Example

```python
history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is attention in transformers?"},
]

for message in history:
    print(message["role"], ":", message["content"])
```

## 5. Retrieval-Augmented Chatbots

A more capable chatbot often uses retrieval-augmented generation (RAG):

1. embed the user query
2. retrieve relevant documents
3. inject them into the prompt
4. generate a grounded answer

This improves factual grounding and reduces reliance on model memory alone.

## 6. Toy Prompt Assembly Example

```python
def build_prompt(user_question, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""
You are a careful assistant.
Use the context below to answer the question.

Context:
{context}

Question:
{user_question}
"""

prompt = build_prompt(
    "What is a transformer?",
    ["Transformers use attention.", "They process sequences in parallel."],
)
print(prompt)
```

## 7. Practical Design Questions

When building a chatbot, you need to choose:

- base model
- context window strategy
- whether to use retrieval
- whether to use tools or functions
- streaming vs non-streaming output
- latency and cost budget

## Summary

A chatbot is an application layer on top of an LLM. The model is important, but the surrounding system design usually determines whether the chatbot feels reliable and useful.
