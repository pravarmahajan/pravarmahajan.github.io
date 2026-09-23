---
layout: post
title: "Finding an Induction Circuit in a Two-Layer Transformer"
---

I created a sequence of random tokens and followed it with an exact copy of itself. The model was bad at predicting the first sequence, which was expected: the tokens were random. But when it reached the repeated sequence, its next-token loss dropped sharply. Somehow, the model was using a pattern it had just encountered in its context.

The exact random sequence was extremely unlikely to have appeared in the model's training data. So how could the model use it? I worked through the [ARENA mechanistic interpretability exercises](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/2-finding-induction-heads/) with TransformerLens to understand one answer: an **induction circuit**.

<!-- AUTHOR PROMPT: Add 2–3 sentences about why you wanted to understand this rather than merely complete the exercise. What did “looking inside a transformer” mean to you before this experiment? -->

## The behavior I wanted to explain

<!-- AUTHOR PROMPT: Introduce the two-layer attention-only model and the repeated-random-token task. Explain why random tokens make the first half difficult and why the repeated half creates an opportunity for in-context copying. Avoid implementation details here. -->

I measured the negative log probability assigned to each actual next token. Across 20 independently sampled sequences, mean loss fell from **14.491 ± 0.506** in the first half to **4.373 ± 0.824** in the repeated half. The loss decreased in all 20 trials, with an average reduction of **10.119 ± 0.503**. Here, `±` denotes the standard deviation across sequence-level means.

<!-- VISUAL 1: Averaged next-token loss across position, with a 10th–90th percentile band and a marked boundary between the fresh and repeated sequences. -->

One detail in the curve matters: the first token in the repeated half is still difficult. At that point, the model has not previously seen the final token of the first half followed by the first token of the sequence. The induction lookup becomes useful from the following prediction onward.

## What pattern should an induction head produce?

Consider this short sequence:

```text
position:  0    1  2  3  4    5  6  7  8
token:    BOS   A  B  C  D    A  B  C  D
```

In a simple induction circuit, a first-layer previous-token head carries information about `A` to the following `B` position. When the model reaches the later `A`, a second-layer head can match it to the information at the earlier `B` position and attend there. Information from that position can then increase the probability of predicting `B` again.

Before inspecting the result, I predicted that an induction head's attention heatmap would contain a diagonal stripe of high values. For each token in the repeated half, the head should attend to the position immediately after that token's earlier occurrence. As both positions advance together, the bright cells should form a diagonal.

<!-- VISUAL 2: An A-B-C-D mechanism diagram followed by one attention heatmap with the predicted offset diagonal highlighted. -->

## Turning the visual pattern into a score

To avoid selecting heads only by eye, I scored every head by its mean attention along the predicted diagonal. Repeated copies of a token are `seq_len` positions apart, but an induction head attends one position after the earlier copy. This gives a distance of `seq_len - 1`. Because destination positions form the vertical axis and source positions form the horizontal axis, the PyTorch diagonal offset is `1 - seq_len`.

Two heads stood out: `1.4` and `1.10`, meaning heads 4 and 10 in layer 1 under zero-based indexing. Across 20 sequences, their scores were:

| Head | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `1.4` | 0.669 | 0.062 | 0.561 | 0.792 |
| `1.10` | 0.873 | 0.028 | 0.798 | 0.909 |

The strongest remaining head never exceeded **0.062**. The separation therefore did not depend on choosing a convenient threshold such as 0.4.

<!-- VISUAL 3: Induction-score heatmap for every layer and head. -->

## Did these heads cause the improvement?

An attention pattern is correlational evidence. To test whether these heads contributed to the prediction improvement, I used a TransformerLens hook to set each head's output to zero and reran the model on the same tokens.

| Intervention | First-half loss change | Repeated-half loss change |
|---|---:|---:|
| Ablate `1.3` control | −0.345 | −0.131 |
| Ablate `1.4` | −0.842 | +2.441 |
| Ablate `1.10` | −0.019 | +5.632 |
| Ablate `1.4` and `1.10` | −0.855 | +8.612 |

<!-- AUTHOR PROMPT: Interpret this table in your own words. Explain why comparing each intervention on the same tokens matters, why the repeated-half column is the key result, and why the control strengthens the conclusion. -->

<!-- VISUAL 4: Paired or grouped chart showing the loss change for the first and repeated halves under each intervention. -->

## Testing the first layer of the circuit

The proposed circuit requires more than the two induction heads. A layer-zero head must first make information about the preceding token available at the next position. My earlier detector identified head `0.7` as a previous-token head.

I predicted that if `0.7` supplied information used by `1.4` and `1.10`, removing it would both increase repeated-half loss and weaken their diagonal attention patterns. Ablating a different layer-zero head should not cause the same collapse.

| Layer-zero intervention | First-half loss change | Repeated-half loss change |
|---|---:|---:|
| Ablate `0.3` control | −1.074 | −1.724 |
| Ablate `0.7` | −0.284 | +9.130 |

The attention-score change was even more direct:

| Condition | Head `1.4` | Head `1.10` |
|---|---:|---:|
| Normal | 0.672 | 0.868 |
| Ablate `0.3` control | 0.875 | 0.945 |
| Ablate `0.7` | 0.005 | 0.010 |

<!-- AUTHOR PROMPT: Explain what this result means in your own words. Include the surprising improvement under the control intervention without inventing an explanation. You may offer interference in the residual stream as a hypothesis, clearly labeled as one. -->

<!-- VISUAL 5: Existing condition-by-layer-one-head heatmap. -->

## What I showed—and what I did not

<!-- AUTHOR PROMPT: Separate the claims carefully:
- Observed: repeated-token loss falls; 1.4 and 1.10 show the predicted stripe.
- Causal evidence: zeroing 1.4/1.10 damages repeated prediction; zeroing 0.7 collapses their stripe and damages repeated prediction.
- Not established: the information written by 0.7, the exact residual-stream direction, whether layer-one heads use it through keys or queries, and whether this is the only contributing circuit.
Mention that zero ablation is a strong intervention that can move activations away from their normal distribution.
-->

The specific mechanism proposed for induction is **K-composition**: the output of the earlier head influences the keys used by the later induction heads. Establishing that path directly would require inspecting the weight composition or using more targeted activation or path patching. That is a natural next experiment, rather than a claim supported by the work above.

## What I learned

<!-- AUTHOR PROMPT: End personally. What changed in how you think about attention patterns, mechanistic explanations, and causal interventions? Aim for one concrete lesson rather than a general summary. -->

---

<!-- AUTHOR TODO: Add the public notebook URL after pushing it to your fork. -->

These experiments follow the [ARENA induction-head exercises](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/2-finding-induction-heads/). The model was inspected with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
