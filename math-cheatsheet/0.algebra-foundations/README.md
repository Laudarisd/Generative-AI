# Algebra Foundations

This chapter fills the gap between school mathematics and the more advanced chapters in this repository. If calculus is the language of change, algebra is the language of structure, manipulation, and symbolic reasoning.

## 1. Numbers and Basic Operations

Important number systems:

- natural numbers: $1,2,3,\dots$
- integers: $\dots,-2,-1,0,1,2,\dots$
- rational numbers: $\frac{a}{b}$
- real numbers
- complex numbers

Basic rules:

- commutative law: $a+b=b+a$, $ab=ba$
- associative law: $(a+b)+c=a+(b+c)$
- distributive law: $a(b+c)=ab+ac$

## 2. Fractions, Ratios, and Proportions

Fractions are everywhere in probability, normalization, and rates.

Example:

```math
\frac{2}{3} + \frac{1}{6} = \frac{4}{6} + \frac{1}{6} = \frac{5}{6}
```

Ratio example:

If a dataset has 80 positive samples and 20 negative samples, the positive-to-negative ratio is:

```math
80:20 = 4:1
```

## 3. Exponents and Radicals

Important rules:

```math
a^m a^n = a^{m+n}
```

```math
\frac{a^m}{a^n} = a^{m-n}
```

```math
(a^m)^n = a^{mn}
```

```math
a^{-n} = \frac{1}{a^n}
```

```math
\sqrt[n]{a} = a^{1/n}
```

## 4. Polynomials

A polynomial is an expression like:

```math
p(x) = a_n x^n + a_{n-1}x^{n-1} + \dots + a_1x + a_0
```

Examples:

- linear: $ax+b$
- quadratic: $ax^2+bx+c$
- cubic: $ax^3+bx^2+cx+d$

### Factoring Example

```math
x^2 - 5x + 6 = (x-2)(x-3)
```

## 5. Solving Equations

### Linear Equation

```math
2x + 5 = 11
```

Solution:

```math
2x = 6 \Rightarrow x = 3
```

### Quadratic Equation

General form:

```math
ax^2 + bx + c = 0
```

Quadratic formula:

```math
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
```

### Worked Example

Solve:

```math
x^2 - 3x - 4 = 0
```

Solution:

```math
x = \frac{3 \pm \sqrt{25}}{2}
= \frac{3 \pm 5}{2}
```

So:

- $x=4$
- $x=-1$

## 6. Inequalities

Example:

```math
2x + 1 > 7 \Rightarrow 2x > 6 \Rightarrow x > 3
```

Important sign rule:

- multiplying or dividing an inequality by a negative number flips the sign

## 7. Absolute Value

Definition:

```math
|x| =
\begin{cases}
x, & x \ge 0 \\
-x, & x < 0
\end{cases}
```

Why it matters:

- MAE loss
- distance formulas
- robustness metrics

## 8. Functions

A function maps input to output:

```math
f: X \to Y
```

Common function types:

- linear
- quadratic
- polynomial
- rational
- exponential
- logarithmic
- trigonometric

### Example

```math
f(x) = 2x + 1
```

Then:

- $f(0)=1$
- $f(3)=7$

## 9. Exponential and Logarithmic Functions

Exponential:

```math
f(x)=a^x
```

Logarithm:

```math
\log_a x = y \iff a^y = x
```

Important identities:

```math
\log(ab)=\log a + \log b
```

```math
\log\left(\frac{a}{b}\right)=\log a - \log b
```

```math
\log(a^k)=k\log a
```

## 10. Trigonometry Basics

Important ratios:

- $\sin \theta$
- $\cos \theta$
- $\tan \theta$

Important identity:

```math
\sin^2 \theta + \cos^2 \theta = 1
```

## 11. Complex Numbers

A complex number has the form:

```math
z = a + bi
```

where $i^2 = -1$.

Magnitude:

```math
|z| = \sqrt{a^2+b^2}
```

## 12. Sequences and Basic Series

Arithmetic sequence:

```math
a_n = a_1 + (n-1)d
```

Geometric sequence:

```math
a_n = a_1 r^{n-1}
```

Finite geometric sum:

```math
S_n = a_1 \frac{1-r^n}{1-r}, \quad r \neq 1
```

## 13. Worked Examples

### Example 1: Simplify an Expression

```math
3(x-2)+4x = 7x-6
```

### Example 2: Solve a Rational Equation

```math
\frac{x}{2} + \frac{x}{3} = 5
```

Solution:

```math
\frac{5x}{6}=5 \Rightarrow x=6
```

### Example 3: Use Logs

If:

```math
\log_{10} x = 2
```

then:

```math
x = 100
```

## 14. Python Examples

```python
import math

a, b, c = 1, -3, -4
disc = b**2 - 4*a*c
x1 = (-b + math.sqrt(disc)) / (2*a)
x2 = (-b - math.sqrt(disc)) / (2*a)
print("quadratic roots:", x1, x2)

z = 3 + 4j
print("complex magnitude:", abs(z))

a1, r = 2, 3
seq = [a1 * (r ** n) for n in range(5)]
print("geometric sequence:", seq)
```

## Practice Problems

1. Solve $5x-7=18$.
2. Factor $x^2+7x+12$.
3. Solve $x^2-9=0$.
4. Simplify $\frac{2}{5}+\frac{3}{10}$.
5. If $\log_2 x = 5$, find $x$.
6. Compute the magnitude of $1-2i$.
