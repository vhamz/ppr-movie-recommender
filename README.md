# Personalized PageRank Movie Recommendations - HW6

Live demo: https://vhamz.github.io/ppr-movie-recommender/
Source: https://github.com/vhamz/ppr-movie-recommender

The original course specification is preserved in `assignment.md`.
The original scaffold is the first git commit; implementation and evidence are
in the next commit. The HW5 directory and dataset files are unchanged.

## Run

1. In this directory, run `python3 -m http.server 8056`.
2. Open http://localhost:8056/ (file:// cannot fetch local datasets).
3. Click **Build graph**, then **Run tests** (12 pass, 0 fail, 0 pending).
   Each action automatically builds the graph if needed. Buttons show a running
   state; a persistent progress message stays visible while computing. Results
   scroll into view on completion. Conflicting actions are disabled until done.
4. Click **Evaluate Recall@10 - all four methods**. The first run sweeps
   validation alphas before choosing the final test setting, loads the exact
   HW5 MF checkpoint, and runs additional numerical/memory checks.
   On the measured machine this takes about 1.5-2 minutes. Progress is shown.
5. Use **Recommend Top-10**, the method/user controls, and **Run two-user demo**.
6. Export verification JSON, metrics CSV and recommendation CSV with the buttons.

Dependencies actually used: TensorFlow.js 4.22.0 (pinned CDN), d3 7.9.0
(vendored), modern JavaScript browser. Python 3 is only needed to serve files.
The first TensorFlow.js CDN load needs internet access; failures show a message.

Optional independent verification: Python 3, NumPy 2.0.2 and SciPy 1.13.1 were
used. Run `python3 evidence/verify.py`. It rebuilds the split and solves a
different linear system instead of reusing the application power iteration.

## Core algorithm

Graph nodes are training users followed by training movies, mapped to contiguous
indices without a padding node. R is binary and includes only training ratings
>= 4. Each positive edge has both directions. Alpha means link following.

```
p_0 = v
p_next = alpha * (Q.T * p + dangling_mass * v) + (1-alpha) * v
```

Global PageRank restarts uniformly; PPR restarts at the selected user node.
Stop at L1 < 1e-6, or after 500 updates with nonconvergence reported.
Core MovieLens propagation is batched TensorFlow.js matrix multiplication using
the two R blocks, with no per-edge JavaScript propagation and no square tensor.
Probability vectors are read asynchronously where practical and tensors disposed.
Interactive PageRank/PPR walks use the asynchronous batched kernel too.
Validation and test ranking loops yield between groups of users so scrolling
and progress updates continue throughout evaluation.

Some alpha-0.95 WebGL walks exhibit a float32 numerical plateau just above the
tolerance. Only affected seeds restart on the TF.js CPU backend, with the SAME
tolerance and iteration cap. Initial capped attempts and successful retries are
recorded explicitly. The measured run needed 27 CPU retries; no final walk was
silently accepted as converged. No physical-GPU timing claim is made.

## Shared evaluation protocol

- Identical MovieLens 100K files and per-user timestamp split as HW5:
  79,619 train / 9,596 validation / 10,785 test. This is not a globally temporal
  split, and rating timestamps are not viewing sessions.
- 943 users, 1,682 loaded movies, 100,000 ratings, sparsity 93.6953%.
- Graph: 2,554 nodes, 45,602 positive edges, one zero-degree user (685),
  253 zero-degree training movies.
- Candidate pool: all 1,611 movies observed in training, including zero-degree
  movies. Validation excludes training-rated items; test excludes train+val.
- Relevant held-out rating: >=4. Macro averaging over the SAME shared users.
  Validation cohort 846; test cohort 861. All candidates can be scored by MF.
- Test: 13 unseen relevant movie IDs, 82 users without eligible relevant movies.
  User 685 also has no positive training edge and is already in those 82.
  Validation: 4 unseen relevant movie IDs and 97 users with no eligible relevant.
- Score ordering: descending, exact ties by ascending raw movie ID.

## Results

Validation PPR Recall@10:

| Alpha | Recall |
|---|---:|
| 0.50 | 0.1144831591 |
| 0.70 | 0.1047969508 |
| 0.85 | 0.0900207716 |
| 0.95 | 0.0737503039 |

Selected alpha: **0.50**, shared by all users and by the primary global
PageRank comparison. Global PageRank was not separately optimized.

| Method | Test Recall@10 | Mean positive-training popularity |
|---|---:|---:|
| Popularity | 0.0760229134 | 271.9969 |
| Global PageRank | 0.0796932415 | 257.8203 |
| PPR | 0.1121885322 | 244.2067 |
| Saved HW5 MF | 0.0200156180 | 50.3842 |

MF reuses the HW5 epoch-2 checkpoint: 16 factors, 20 original epochs, Adam
lr=0.01, L2=0.05, batch=64, validation RMSE=1.002885. Prediction remains
clip(training mean + user dot movie, 1, 5); no replacement MF is trained.
MF was selected by validation RMSE and PPR by validation recall.

Users 1 and 2 are the two smallest IDs in the common test cohort. Every method
uses those same users. Their full lists and three one-factor controls are in
`recommendations.csv`. All reported lists have ten eligible nonzero-score items.
User 1: PPR 3/19 relevant hits, MF 1/19. User 2: all methods 0/2.
The previous HW5 example user 405 cannot serve as an eligible HW6 example:
there is no eligible relevant test item for that user.

Seed-only demo at alpha 0.85: 0/10 list overlap under user-specific masks.
With one shared eligible mask, overlap is 8/10: filtering explains much of the
visible difference. Alpha-only demo for user 1: 7/10 overlap at 0.85 vs 0.50.
PPR scores are graph-walk probabilities, not ratings or purchase probabilities.
Less popularity is not automatically better discovery.

## Evidence and changes

`verification.json`: 12 supplied checks plus extra checks and batch audits.
`results.csv`: four-method test results, validation sweep and convergence.
`recommendations.csv`: user/method/alpha/rank/ID/title/genres/score/popularity.
`evidence/experiment.json`: full actual run, vectors and per-user convergence.
`evidence/independent-verification.json`: independent SciPy solution, all
displayed-list checks and paired bootstrap (5,000 user resamples, seed 20261021).
`evidence/controls.json`: fixed-mask and fixed-alpha controlled comparisons.
`evidence/initial-webgl-diagnostic.json`: initial capped attempts and CPU reruns.
`evidence/sha256.json`: dataset/checkpoint/specification hashes.

The seven starter stubs are filled. Necessary integrations additionally fix the
raw-ID/node-index adapter, validation-before-test selection, MF reuse and clipping,
and the graph-view cap that originally kept only user nodes. The display now
reserves movie-neighbor slots and draws a small connected induced subgraph.
This cap never reduces the graph used in propagation.

Source references and the single business decision are documented in the report.
The offline evidence supports a PPR pilot for the main warm-user section, not a
claim of improved viewing completion, retention or revenue without an experiment.
