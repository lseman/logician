# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

> Compiled article maintained by the wiki builder.

- ID: `papers-2604-md`
- Source: [source note](../../source/papers/2604.md)
- Source path: `papers/2604.md`
- Updated: `2026-04-09T16:23:32Z`
- Words: `998`
- Summary: > Research paper imported from `https://arxiv.org/pdf/2604.04921` into raw artifact `papers/2604.pdf`.

## Concepts

- [[concepts/folder-papers|Papers]]

## Backlinks

- No explicit backlinks yet.

## Related Notes

- No related notes inferred yet.

## Headings

- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
- Summary
- Paper Metadata
- Reading Notes
- Suggested Follow-Up Pages
- Extracted Text Excerpt

## Source Material

# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

> Research paper imported from `https://arxiv.org/pdf/2604.04921` into raw artifact `papers/2604.pdf`.

## Summary

Extended reasoning in large language models (LLMs) creates severe KV cache memory bottle- necks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making repre- sentative queries very few, leading to poor top- key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly con- centrated around fixed non-zero centers and re- main stable across positions—Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K- token generation, TriAttention matches Full At- tention reasoning accuracy while achieving 2.5...

## Paper Metadata

- PDF: `papers/2604.pdf`
- Extracted text: `papers/2604.txt`
- Metadata: `papers/2604.json`
- Imported at: `2026-04-09T16:23:32Z`
- SHA256: `9c14b75a7cc792d450d5571ec74e12db9a01f44fcb918e5b082bc4a8f51c1350`
- Pages detected: `19`
- Source URL: https://arxiv.org/pdf/2604.04921
- Authors: Weian Mao; Xi Lin; Wei Huang; Yuxin Xie; Tianfu Fu; Bohan Zhuang; Song Han; Yukang Chen
- arXiv: `2604.04921`
- DOI: `10.48550/arXiv.2604.04921`

## Reading Notes

- Problem: _fill in the research question or gap._
- Method: _summarize the core approach and assumptions._
- Evidence: _capture datasets, experiments, proofs, or qualitative support._
- Results: _record the strongest reported findings with page/section anchors when possible._
- Limitations: _note caveats, threats to validity, missing comparisons, and uncertainty._
- Follow-up: _link related papers, repos, datasets, or wiki notes._

## Suggested Follow-Up Pages

- `create` `concepts/reasoning-compression.md`: This source appears to introduce or strengthen a durable topic that is not yet represented as its own wiki note.

## Extracted Text Excerpt

```text
TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
Weian Mao1∗ Xi Lin 3∗ Wei Huang2∗ Yuxin Xie1 Tianfu Fu 1
Bohan Zhuang 3 Song Han 1 2 Yukang Chen2
Abstract
Extended reasoning in large language models
(LLMs) creates severe KV cache memory bottle-
necks. Leading KV cache compression methods
estimate KV importance using attention scores
from recent post-RoPE queries. However, queries
rotate with position during RoPE, making repre-
sentative queries very few, leading to poor top-
key selection and unstable reasoning. To avoid
this issue, we turn to the pre-RoPE space, where
we observe that Q and K vectors are highly con-
centrated around fixed non-zero centers and re-
main stable across positions—Q/K concentration.
We show that this concentration causes queries to
preferentially attend to keys at specific distances
(e.g., nearest keys), with the centers determining
which distances are preferred via a trigonometric
series. Based on this, we propose TriAttention
to estimate key importance by leveraging these
centers. Via the trigonometric series, we use the
distance preference characterized by these centers
to score keys according to their positions, and also
leverage Q/K norms as an additional signal for
importance estimation. On AIME25 with 32K-
token generation, TriAttention matches Full At-
tention reasoning accuracy while achieving 2.5×
higher throughput or 10.7× KV memory reduc-
tion, whereas leading baselines achieve only about
half the accuracy at the same efficiency. TriAtten-
tion enables OpenClaw deployment on a single
consumer GPU, where long context would other-
wise cause out-of-memory with Full Attention.
Our code is available at https://github.
com/WeianMao/triattention.
1. Introduction
Long reasoning in LLMs produces chain-of-thought se-
quences spanning tens of thousands of tokens (Wei et al.,
2022; DeepSeek-AI, 2025; Shao et al., 2024; Chen et al.,
1MIT 2NVIDIA 3ZJU ∗Equal contribution.
Figure 1.Performance trade-offs on AIME25 (Qwen3-8B).(A)At
equivalent accuracy (40.8%), TriAttention achieves 2.5× higher
throughput than Full Attention.(B)TriAttention reduces KV cache
memory by 10.7×while matching Full Attention accuracy.
2025). KV cache grows proportionally, creating severe
memory bottlenecks. KV cache compression addresses this
by retaining only the most important tokens, with impor-
tance estimated from attention scores computed over recent
queries (Li et al., 2024b; Zhang et al., 2023; Devoto et al.,
2025; Zhou et al., 2024; Tang et al., 2024; Shi et al., 2024).
However, these methods are inherently unstable: only a
few queries are usable for importance estimation. They
operate on post-RoPE queries, which rotate with position
as illustrated in Figure 2(B); consequently, only the most
recent queries retain up-to-date orientations, forming a tiny
observation window. With so few representative queries, im-
portant keys go undetected—a token receiving low attention
during this short window may be permanently evicted, even
if it becomes critical later. This is particularly challenging
for retrieval heads (Wu et al., 2025; Xiao et al., 2025), where
relevant tokens can remain dormant for long periods before
becoming essential. In reasoning, such loss breaks the chain
of thought. Prior work confirms this limitation: increasing
the observation window does not help—performance peaks
at around 25 queries, a tiny fraction of typical long contexts,
and declines thereafter (Zhang et al., 2025).
We therefore turn to the pre-RoPE space, where we observe
a notable phenomenon: across a large fraction of attention
heads, Q and K vectors are highly concentrated around a
fixed non-zero center, as shown in Figure 2(A)—a property
1
arXiv:2604.04921v1  [cs.CL]  6 Apr 2026

TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
Figure 2.Q/K concentration and its implications for attention.(A)Pre-RoPE Q/K vectors at the dominant frequency band are highly
concentrated (high Mean Resultant Lengt

[truncated]
```
