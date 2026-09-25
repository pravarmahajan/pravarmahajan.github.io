---
layout: post
title: "Finding an Induction Circuit in a Two-Layer Transformer"
---

I am an ML practitioner, and I have always been drawn to the question of what happens under the hood inside a deep neural network. To explore that question hands-on, I started working through ARENA's mechanistic interpretability exercises. I am writing this post to share my understanding as I develop it, beginning with one of the simplest interesting mechanisms found in transformers: induction circuits.

I created a random sequence of tokens and followed it with an exact copy. I wanted to test a simple question: can a transformer use a sequence it has just encountered in its context, even when that exact sequence was extremely unlikely to have appeared in its training data? The model was bad at predicting the first copy, as expected. When it reached the repeated copy, however, its next-token loss dropped sharply. Somehow, it was using the pattern it had just seen.

I had two reasons for doing this exercise. First, I wanted to understand attention mechanisms and satisfy my own curiosity. Second, as language models have become increasingly capable, I have kept returning to a harder question: do these architectures merely encode surface-level statistics, or do they learn reusable internal mechanisms? I worked through the [ARENA induction-head exercises](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/2-finding-induction-heads/) with TransformerLens to investigate one such mechanism.

## The behavior I wanted to explain

I used [`callummcdougall/attn_only_2L_half`](https://huggingface.co/callummcdougall/attn_only_2L_half), a small pretrained transformer used in mechanistic interpretability education. It has two attention-only layers and no MLP layers, making its internal behavior easier to study.

I sampled 50 random, nonspecial token IDs from the model's vocabulary, duplicated the sequence, and prefixed a beginning-of-sequence token. Predicting the first copy should be difficult because its ordering is random. In the second copy, every transition except the first has already appeared in the model's context. If the model can retrieve those transitions, it should become much better at predicting the repeated half.

I measured the negative log probability assigned to each actual next token. Across 20 independently sampled sequences, mean loss fell from **14.491 ± 0.506** in the first half to **4.373 ± 0.824** in the repeated half. The loss decreased in all 20 trials, with an average reduction of **10.119 ± 0.503**. Here, `±` denotes the standard deviation across sequence-level means.

<iframe class="interactive-figure" src="{{ site.baseurl }}/assets/induction-circuits/loss-across-positions.html" title="Next-token loss across repeated random sequences" loading="lazy"></iframe>
<p class="figure-caption">Mean next-token loss over 20 random sequences. The shaded region spans the 10th to 90th percentile; hover for position-level values.</p>

One detail in the curve matters: the first token in the repeated half is still difficult. At that point, the model has not previously seen the final token of the first half followed by the first token of the sequence. The induction lookup becomes useful from the following prediction onward.

## What pattern should an induction head produce?

Consider this short sequence:

```text
position:  0    1  2  3  4    5  6  7  8
token:    BOS   A  B  C  D    A  B  C  D
```

In a simple induction circuit, a first-layer previous-token head carries information about `A` to the following `B` position. When the model reaches the later `A`, a second-layer head can match it to the information at the earlier `B` position and attend there. Information from that position can then increase the probability of predicting `B` again.

Before inspecting the result, I predicted that an induction head's attention heatmap would contain a diagonal stripe of high values. For each token in the repeated half, the head should attend to the position immediately after that token's earlier occurrence. As both positions advance together, the bright cells should form a diagonal.

<iframe class="interactive-figure interactive-figure--tall" src="{{ site.baseurl }}/assets/induction-circuits/induction-stripe-head-1-10.html" title="Attention heatmap for induction head 1.10" loading="lazy"></iframe>
<p class="figure-caption">Head 1.10's attention from the repeated half back to the first copy. Hover over a cell to inspect its query token, source token, and attention weight.</p>

## Turning the visual pattern into a score

To avoid selecting heads only by eye, I scored every head by its mean attention along the predicted diagonal. Repeated copies of a token are `seq_len` positions apart, but an induction head attends one position after the earlier copy. This gives a distance of `seq_len - 1`. Because destination positions form the vertical axis and source positions form the horizontal axis, the PyTorch diagonal offset is `1 - seq_len`.

Two heads stood out: `1.4` and `1.10`, meaning heads 4 and 10 in layer 1 under zero-based indexing. Across 20 sequences, their scores were:

| Head | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `1.4` | 0.669 | 0.062 | 0.561 | 0.792 |
| `1.10` | 0.873 | 0.028 | 0.798 | 0.909 |

The strongest remaining head never exceeded **0.062**. The separation therefore did not depend on choosing a convenient threshold such as 0.4.

<iframe class="interactive-figure interactive-figure--compact" src="{{ site.baseurl }}/assets/induction-circuits/induction-scores.html" title="Induction scores for every attention head" loading="lazy"></iframe>
<p class="figure-caption">Mean attention on the predicted induction diagonal. Heads 1.4 and 1.10 clearly separate from the remaining heads.</p>

## Did these heads cause the improvement?

An attention pattern is correlational evidence. To test whether these heads contributed to the prediction improvement, I used a TransformerLens hook to set each head's output to zero and reran the model on the same tokens.

| Intervention | First-half loss change | Repeated-half loss change |
|---|---:|---:|
| Ablate `1.3` control | −0.345 | −0.131 |
| Ablate `1.4` | −0.842 | +2.441 |
| Ablate `1.10` | −0.019 | +5.632 |
| Ablate `1.4` and `1.10` | −0.855 | +8.612 |

Ablating either candidate increased repeated-half loss, and ablating both produced the largest increase. The control intervention on head `1.3` did not damage repeated-half performance. Because every condition used the same tokens within each trial, these differences isolate the intervention rather than variation between random sequences. The result provides causal evidence that heads `1.4` and `1.10` contribute to the model's improved predictions on the repeated half.

<iframe class="interactive-figure" src="{{ site.baseurl }}/assets/induction-circuits/induction-head-ablation.html" title="Loss changes after ablating candidate induction heads" loading="lazy"></iframe>
<p class="figure-caption">Change from the normal model under zero ablation. Error bars show standard deviation across 20 paired trials.</p>

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

Ablating `0.7` increased repeated-half loss by **9.130** and almost completely erased the induction patterns of both layer-one heads. This is evidence that the behavior of `1.4` and `1.10` depends on information supplied by `0.7`.

Removing control head `0.3` unexpectedly improved both loss and induction scores on this task. Head `0.3` is a first-token head rather than an inert component, so one possibility is that its residual-stream contribution partially interferes with induction on this artificial task. I did not test that explanation, and the result remains an interesting follow-up question.

<iframe class="interactive-figure interactive-figure--compact" src="{{ site.baseurl }}/assets/induction-circuits/layer0-ablation-scores.html" title="Layer-one induction scores after layer-zero ablation" loading="lazy"></iframe>
<p class="figure-caption">Ablating previous-token head 0.7 collapses the induction scores of layer-one heads 1.4 and 1.10.</p>

## What I showed—and what I did not

These experiments identified two heads with the attention pattern predicted for induction, showed that their scores were stable across independently sampled sequences, and found that zeroing their outputs damaged repeated-token prediction. Ablating the previous-token head `0.7` also collapsed their induction patterns and sharply increased repeated-half loss. Together, these results provide causal evidence that these three heads participate in the behavior.

They do not reveal exactly what information `0.7` writes, which residual-stream directions carry it, whether the layer-one heads read it through their keys or queries, or whether this is the only circuit contributing to the model's predictions. Zero ablation is also a strong intervention that can move activations away from their normal distribution.

The specific mechanism proposed for induction is **K-composition**: the output of the earlier head influences the keys used by the later induction heads. Establishing that path directly would require inspecting the weight composition or using more targeted activation or path patching. That is a natural next experiment rather than a claim supported directly by the work above.

## What I learned

Before this exercise, I tended to think of attention mainly as a way to encode statistical relationships between tokens. This experiment gave me a more concrete picture: a small collection of heads can implement a reusable, algorithm-like behavior that operates on a sequence sampled after training. It does not settle the larger question of what kinds of representations language models learn, but it shows how mechanistic interpretability can turn a behavioral observation into a testable internal hypothesis.

The most useful lesson was methodological. I first predicted an attention pattern, turned that prediction into a quantitative score, checked it across multiple examples, and then intervened on the suspected components. That progression—from observation to measurement to causal testing—is what made the explanation feel more substantive than an attention visualization alone. It has encouraged me to continue learning about mechanistic interpretability.

---

The annotated notebook will be linked here after it is published. These experiments follow the [ARENA induction-head exercises](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/2-finding-induction-heads/), and the model was inspected with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
