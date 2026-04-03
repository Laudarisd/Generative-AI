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

## 2.5 Proof Techniques

Discrete mathematics is tightly connected to proof writing.

Common proof methods:

- direct proof
- proof by contradiction
- proof by contrapositive
- mathematical induction

### Induction Example

Claim:

```math
1 + 2 + \dots + n = \frac{n(n+1)}{2}
```

Induction is one of the cleanest ways to prove this.

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

## 3.5 Binomial Theorem

```math
(a+b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k}b^k
```

This appears in probability expansions and polynomial reasoning.

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

## 4.5 Graph Traversal

Two fundamental graph algorithms:

- breadth-first search, or BFS
- depth-first search, or DFS

Why this matters:

- shortest path intuition in unweighted graphs
- knowledge graph exploration
- graph-based reasoning systems

## 5. Trees

Trees matter in:

- decision trees
- parsing
- search
- hierarchical clustering

Important property:

- a tree with $n$ nodes has $n-1$ edges

Binary trees matter especially in algorithms and parsing.

## 6. Recurrence Relations

Many algorithms are described recursively.

Example:

```math
T(n) = 2T(n/2) + n
```

This helps analyze algorithm complexity.

### Master-Theorem Intuition

Recurrences like:

```math
T(n)=2T(n/2)+n
```

often lead to:

```math
T(n)=O(n \log n)
```

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

Transition matrix:

```math
P_{ij} = P(X_{t+1}=j \mid X_t=i)
```

This matters for:

- stochastic processes
- PageRank
- state models
- generative processes

## 8.5 Boolean Algebra

Boolean variables take values in $\{0,1\}$.

Operations:

- AND
- OR
- NOT

This matters in:

- digital logic
- masking
- rule-based reasoning
- conditional routing

### Python Example

```python
a = True
b = False

print(a and b)
print(a or b)
print(not a)
```

## 9. Probability Trees and Counting Logic

Discrete math supports probabilistic reasoning because many probability spaces are countable.

Example:

- two coin flips produce four outcomes:
  - HH
  - HT
  - TH
  - TT

If outcomes are equally likely, then:

```math
P(\text{exactly one head}) = \frac{2}{4} = \frac{1}{2}
```

## 10. Example: Counting Token Sequences

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

## 11. Worked Example: Recurrence

Suppose:

```math
T(n) = T(n-1) + 2, \quad T(1)=1
```

Then:

- $T(2)=3$
- $T(3)=5$
- $T(4)=7$

So the closed form is:

```math
T(n)=2n-1
```

## Practice Problems

1. Compute $\binom{5}{2}$.
2. Explain why attention cost grows quadratically with sequence length.
3. Give a real AI example of a graph.
4. Compare permutation and combination.
5. Write a recurrence relation for binary tree traversal.
6. Count how many 3-bit binary strings exist.
