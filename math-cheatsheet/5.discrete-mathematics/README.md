# Discrete Mathematics

Discrete mathematics is the language of finite structures. It matters in algorithms, graph learning, tokenization, combinatorics, logic, and formal reasoning.

## 1. Sets, Relations, and Functions

Basic objects:

- sets
- subsets
- Cartesian products
- relations
- mappings

These appear everywhere in computer science and ML notation.

## 2. Logic

Logical reasoning underlies:

- rule systems
- proofs
- symbolic AI
- query languages
- control flow

Common operators:

- AND
- OR
- NOT
- implication
- equivalence

## 3. Counting and Combinatorics

Factorial:

```math
n! = n(n-1)\cdots 1
```

Permutations:

```math
P(n,r) = \frac{n!}{(n-r)!}
```

Combinations:

```math
\binom{n}{r} = \frac{n!}{r!(n-r)!}
```

Why this matters:

- search space size
- model configurations
- token sequence counting
- probability calculations

## 4. Graphs

A graph is:

```math
G = (V, E)
```

where:

- $V$ is the vertex set
- $E$ is the edge set

Applications in AI:

- knowledge graphs
- molecular graphs
- social networks
- recommendation systems
- graph neural networks

## 5. Trees

Trees matter in:

- decision trees
- parsing
- search
- hierarchical clustering

Important property:

- a tree with $n$ nodes has $n-1$ edges

## 6. Recurrence Relations

Many algorithms are described recursively.

Example:

```math
T(n) = 2T(n/2) + n
```

This helps analyze algorithm complexity.

## 7. Big-O Notation

Growth rates:

- $O(1)$ constant
- $O(\log n)$ logarithmic
- $O(n)$ linear
- $O(n \log n)$ quasi-linear
- $O(n^2)$ quadratic

Why this matters in AI:

- training cost
- inference cost
- attention complexity

For standard full attention:

```math
O(n^2)
```

with respect to sequence length.

## 8. Markov Chains

A Markov chain uses transition probabilities between states.

This matters for:

- stochastic processes
- PageRank
- state models
- generative processes

## 9. Example: Counting Token Sequences

If a vocabulary has size $V$ and sequence length is $T$, then the number of possible sequences is:

```math
V^T
```

This simple fact explains why language modeling is a huge search problem.

### Python Example

```python
V = 50000
T = 10
print(V ** T)
```

## Practice Problems

1. Compute $\binom{5}{2}$.
2. Explain why attention cost grows quadratically with sequence length.
3. Give a real AI example of a graph.
4. Compare permutation and combination.
5. Write a recurrence relation for binary tree traversal.
