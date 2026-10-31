# Evaluation report (auto-generated)

## Quality (raw model output; accuracy / macro-F1 / head vs rare)

| config | suite | n | accuracy | macro_f1 | head_f1 | rare_f1 | em_crop | em_both | parse_ok_rate | refusal_rate | grounded_mean | citation_correct_rate | dose_violation_rate | retrieval_hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mock / rag=off / raw | test | 150 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |  |  | 0.0 |  |
| mock / rag=on / raw | test | 150 | 0.913 | 0.897 | 0.939 | 0.863 | 0.987 | 0.913 | 1.0 | 0.0 | 0.996 | 0.913 | 0.0 | 1.0 |
| mock / rag=on / raw | handwritten | 36 | 0.889 | 0.885 | 0.85 | 0.913 | 1.0 | 0.889 | 1.0 | 0.0 | 0.993 | 0.889 | 0.0 | 0.917 |
| mock / rag=on / raw | terminology | 40 | 0.975 | 0.97 | 0.975 | 0.967 | 1.0 | 0.975 | 1.0 | 0.0 | 0.994 | 0.975 | 0.0 | 1.0 |

## Robustness: accuracy by prompt length

| config | handwritten | long1024 | long2048 | long512 | terminology | test |
|---|---|---|---|---|---|---|
| mock / rag=off / raw |  |  |  |  |  | 0.0 |
| mock / rag=on / graph | 0.778 |  |  |  |  | 0.627 |
| mock / rag=on / raw | 0.889 | 0.433 | 0.3 | 0.6 | 0.975 | 0.913 |

## Robustness: OOD behaviour

| config | n | refusal_rate | false_answer_rate | dose_hallucination_rate | dose_refusal_rate |
|---|---|---|---|---|---|
| mock / rag=off / raw | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| mock / rag=on / raw | 40 | 0.0 | 1.0 | 0.0 | 0.0 |
| mock / rag=on / graph | 40 | 0.975 | 0.031 | 0.0 | 1.0 |

## Decision layer (graph pipeline)

| config | suite | n | accuracy | false_accept_rate | false_fallback_rate | retry_rate | citation_correct_rate |
|---|---|---|---|---|---|---|---|
| mock / rag=on / graph | test | 150 | 0.627 | 0.04 | 0.333 | 0.327 | 0.94 |
| mock / rag=on / graph | handwritten | 36 | 0.778 | 0.0 | 0.222 | 0.222 | 1.0 |

## Efficiency

_not run_

## Perplexity (held-out agri text)

_not run_

## Automatic findings: where the average hides failures

- **mock / rag=on / raw**: accuracy 0.91 on short prompts falls to 0.30 at ~2048 tokens.
- **mock / rag=on / raw**: answers 100% of out-of-distribution questions instead of refusing.

## Retrieval

| suite | config | recall@1 | recall@3 | recall@5 | mrr | ndcg@5 | n | latency_ms |
|---|---|---|---|---|---|---|---|---|
| test | dense | 0.873 | 0.967 | 0.993 | 0.924 | 0.903 | 150 | 4.146 |
| test | bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 0.956 | 150 | 0.189 |
| test | hybrid | 0.913 | 1.0 | 1.0 | 0.956 | 0.911 | 150 | 4.271 |
| test | hybrid_no_crop_filter | 0.893 | 1.0 | 1.0 | 0.946 | 0.929 | 150 | 4.192 |
| test | hybrid_no_alias_expansion | 0.913 | 1.0 | 1.0 | 0.956 | 0.911 | 150 | 4.194 |
| handwritten | dense | 0.861 | 0.861 | 0.917 | 0.874 | 0.838 | 36 | 4.121 |
| handwritten | bm25 | 0.889 | 0.917 | 0.944 | 0.91 | 0.888 | 36 | 0.215 |
| handwritten | hybrid | 0.889 | 0.917 | 0.944 | 0.908 | 0.874 | 36 | 4.431 |
| handwritten | hybrid_no_crop_filter | 0.833 | 0.889 | 0.889 | 0.856 | 0.836 | 36 | 4.591 |
| handwritten | hybrid_no_alias_expansion | 0.889 | 0.917 | 0.944 | 0.908 | 0.874 | 36 | 4.392 |
| terminology | dense | 0.95 | 1.0 | 1.0 | 0.971 | 0.95 | 40 | 4.734 |
| terminology | bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 0.978 | 40 | 0.215 |
| terminology | hybrid | 0.975 | 1.0 | 1.0 | 0.988 | 0.966 | 40 | 4.943 |
| terminology | hybrid_no_crop_filter | 0.975 | 0.975 | 0.975 | 0.975 | 0.944 | 40 | 4.984 |
| terminology | hybrid_no_alias_expansion | 0.975 | 1.0 | 1.0 | 0.988 | 0.963 | 40 | 5.581 |
| long1024 | dense | 0.3 | 0.733 | 0.933 | 0.536 | 0.636 | 30 | 171.425 |
| long1024 | bm25 | 0.367 | 0.767 | 0.933 | 0.584 | 0.668 | 30 | 8.675 |
| long1024 | hybrid | 0.433 | 0.833 | 0.933 | 0.64 | 0.697 | 30 | 185.045 |
| long1024 | hybrid_no_crop_filter | 0.633 | 0.7 | 0.833 | 0.698 | 0.706 | 30 | 184.127 |
| long1024 | hybrid_no_alias_expansion | 0.433 | 0.833 | 0.933 | 0.64 | 0.697 | 30 | 184.188 |