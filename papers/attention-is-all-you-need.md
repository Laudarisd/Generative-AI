# Understanding "Attention Is All You Need" in Detail
---

(paper_link)[https://arxiv.org/pdf/1706.03762]

## Overview of the Transformer

The Transformer is designed for sequence transduction, converting an input sequence (e.g., an English sentence) into an output sequence (e.g., a German sentence). It uses an encoder-decoder architecture based entirely on attention, avoiding RNNs and convolutions.

- **Encoder**: Processes the input sequence into a continuous representation.
- **Decoder**: Generates the output sequence from the encoder's representation.
- Both consist of 6 identical layers (in the base model), with attention and feed-forward networks as key components.

---

## Key Concepts and Mathematics

### 1. Scaled Dot-Product Attention

The core mechanism of the Transformer, computing how much focus each word gives to others.

- **Inputs**:
  - Queries \( Q \) (\( n \times d_k \)): What we're looking for.
  - Keys \( K \) (\( n \times d_k \)): What we compare against.
  - Values \( V \) (\( n \times d_v \)): What we retrieve.
  - \( n \): Sequence length, \( d_k \): Key/query dimension, \( d_v \): Value dimension.
- **Formula**:
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
  \]
  - \( Q K^T \): Dot product similarity (\( n \times n \) matrix).
  - \( \frac{1}{\sqrt{d_k}} \): Scales to prevent large values from skewing softmax gradients.
  - \( \text{softmax} \): Normalizes scores into probabilities.
  - Multiply by \( V \): Weights values, output is \( n \times d_v \).

- **Intuition**: For "The cat sleeps," the query for "sleeps" finds "cat" relevant via keys and retrieves info from values.

### 2. Multi-Head Attention

Uses multiple attention "heads" to capture different relationships.

- **Process**:
  - Split \( Q \), \( K \), \( V \) into \( h \) heads (e.g., \( h = 8 \)).
  - For each head \( i \):
    - Project: \( Q W_i^Q \), \( K W_i^K \), \( V W_i^V \) (to \( d_k = d_v = d_{\text{model}} / h = 64 \)).
    - Compute: \( \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V) \).
  - Concatenate: \( \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O \).
  - \( W^O \): Projects to \( d_{\text{model}} = 512 \).

- **Why?**: Each head focuses on different aspects (e.g., syntax, semantics).

### 3. Positional Encoding

Adds sequence order info since there’s no recurrence.

- **Formula**:
  \[
  PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
  \]
  - \( pos \): Position (0, 1, 2, …).
  - \( i \): Dimension index (0 to \( d_{\text{model}}/2 - 1 \)).
  - Wavelengths range from \( 2\pi \) to \( 10000 \cdot 2\pi \).

- **Intuition**: Sinusoids encode position uniquely, enabling generalization to longer sequences.

### 4. Feed-Forward Networks (FFN)

Applied position-wise in each layer:
\[
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
\]
- Input/output: \( d_{\text{model}} = 512 \).
- Inner layer: \( d_{ff} = 2048 \).

### 5. Layer Normalization and Residual Connections

- **Residual**: \( x + \text{Sublayer}(x) \) (e.g., after attention/FFN).
- **LayerNorm**: Normalizes across features for stability.

### 6. Decoder Masking

Masks future positions in decoder self-attention (sets to \(-\infty\)) to enforce auto-regressive generation.

---

## Architecture Recap

- **Encoder**: 6 layers, each with:
  - Multi-head self-attention.
  - FFN.
  - Residual + LayerNorm.
- **Decoder**: 6 layers, each with:
  - Masked multi-head self-attention.
  - Multi-head attention over encoder output.
  - FFN.
  - Residual + LayerNorm.

---

## Understanding the Plots

The paper includes several figures (plots) to illustrate the Transformer’s behavior.

### Figure 1: Transformer Model Architecture

- **Description**: Diagram of encoder (left) and decoder (right) stacks.
- **Key Elements**:
  - Encoder: Input embedding + PE → 6 layers → output to decoder.
  - Decoder: Output embedding + PE → 6 layers → linear + softmax.
  - Arrows show encoder-decoder attention.
- **Purpose**: Shows data flow and structure.

### Figure 2: Scaled Dot-Product and Multi-Head Attention

- **Left**: \( Q K^T / \sqrt{d_k} \rightarrow \text{softmax} \rightarrow V \).
- **Right**: Multiple attention blocks, concatenated, projected.
- **Purpose**: Explains attention computation and multi-head benefits.

### Figure 3: Attention Example (Layer 5, Encoder Self-Attention)

- **Description**: Attention weights for "making" (e.g., in "making…more difficult").
- **Details**:
  - X-axis: Input words.
  - Y-axis: Heads (colored).
  - Lines show "making" attending to "difficult."
- **Insight**: Captures long-distance dependencies.

### Figure 4: Anaphora Resolution (Layer 5, Heads 5 and 6)

- **Top**: Full attention for head 5.
- **Bottom**: "its" attention for heads 5 and 6 (sharp peaks).
- **Insight**: Heads resolve pronouns (e.g., "its" to its referent).

### Figure 5: Sentence Structure Behavior (Layer 5)

- **Description**: Two heads with distinct patterns.
- **Details**: One focuses locally (syntax), another broadly (semantics).
- **Insight**: Heads learn interpretable tasks.

---

## Why It Works

- **Parallelization**: No sequential processing, trains faster (e.g., 12 hours vs. weeks).
- **Long-Range Dependencies**: Attention connects all positions in \( O(1) \) steps (Table 1).
- **Performance**: 28.4 BLEU (EN-DE), 41.8 BLEU (EN-FR).

---

## Further Exploration

1. **Math**: Want a numerical attention example?
2. **Plots**: Which figure needs more clarification?
3. **Details**: More on positional encoding, training, or another section?

Let me know how to proceed!