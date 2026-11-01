# Evaluation report (auto-generated)

## Quality (raw model output; accuracy / macro-F1 / head vs rare)

| config | suite | n | accuracy | macro_f1 | head_f1 | rare_f1 | em_crop | em_both | parse_ok_rate | refusal_rate | grounded_mean | citation_correct_rate | dose_violation_rate | retrieval_hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nf4 / rag=on / raw | test | 150 | 0.54 | 0.553 | 0.634 | 0.488 | 0.653 | 0.48 | 0.98 | 0.24 | 0.525 | 0.12 | 0.0 | 1.0 |
| nf4 / rag=on / raw | handwritten | 36 | 0.583 | 0.62 | 0.521 | 0.7 | 0.556 | 0.444 | 0.944 | 0.278 | 0.432 | 0.25 | 0.0 | 0.917 |
| nf4 / rag=on / raw | terminology | 40 | 0.3 | 0.352 | 0.458 | 0.267 | 0.25 | 0.175 | 1.0 | 0.6 | 0.274 | 0.2 | 0.025 | 1.0 |
| fp16 / rag=off / raw | test | 150 | 0.213 | 0.133 | 0.238 | 0.05 | 0.827 | 0.193 | 1.0 | 0.007 |  |  | 0.073 |  |
| fp16 / rag=off / raw | handwritten | 36 | 0.111 | 0.119 | 0.1 | 0.133 | 0.75 | 0.111 | 0.972 | 0.0 |  |  | 0.056 |  |
| fp16 / rag=off / raw | terminology | 40 | 0.65 | 0.699 | 0.613 | 0.767 | 0.55 | 0.4 | 0.95 | 0.0 |  |  | 0.0 |  |
| lora_fp16 / rag=on / raw | test | 150 | 0.72 | 0.743 | 0.818 | 0.684 | 0.813 | 0.72 | 1.0 | 0.173 | 0.822 | 0.64 | 0.0 | 1.0 |
| lora_fp16 / rag=on / raw | handwritten | 36 | 0.667 | 0.759 | 0.833 | 0.7 | 0.667 | 0.667 | 1.0 | 0.333 | 0.672 | 0.639 | 0.0 | 0.917 |
| lora_fp16 / rag=on / raw | terminology | 40 | 0.35 | 0.398 | 0.438 | 0.367 | 0.35 | 0.35 | 1.0 | 0.65 | 0.391 | 0.35 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=off / raw | test | 150 | 0.36 | 0.291 | 0.429 | 0.18 | 0.687 | 0.353 | 0.953 | 0.28 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=off / raw | handwritten | 36 | 0.278 | 0.306 | 0.437 | 0.2 | 0.611 | 0.278 | 1.0 | 0.361 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=off / raw | terminology | 40 | 0.325 | 0.335 | 0.38 | 0.3 | 0.45 | 0.325 | 1.0 | 0.55 |  |  | 0.0 |  |
| nf4 / rag=off / raw | test | 150 | 0.193 | 0.111 | 0.249 | 0.0 | 0.773 | 0.173 | 0.98 | 0.047 |  |  | 0.353 |  |
| nf4 / rag=off / raw | handwritten | 36 | 0.083 | 0.084 | 0.106 | 0.067 | 0.722 | 0.083 | 1.0 | 0.139 |  |  | 0.278 |  |
| nf4 / rag=off / raw | terminology | 40 | 0.55 | 0.598 | 0.679 | 0.533 | 0.35 | 0.25 | 0.975 | 0.2 |  |  | 0.225 |  |
| qlora_nf4 / rag=on / raw | test | 150 | 0.92 | 0.921 | 0.926 | 0.917 | 0.987 | 0.92 | 1.0 | 0.0 | 0.962 | 0.753 | 0.0 | 1.0 |
| qlora_nf4 / rag=on / raw | handwritten | 36 | 0.833 | 0.826 | 0.825 | 0.827 | 0.944 | 0.833 | 1.0 | 0.056 | 0.908 | 0.806 | 0.0 | 0.917 |
| qlora_nf4 / rag=on / raw | terminology | 40 | 0.725 | 0.792 | 0.824 | 0.767 | 0.75 | 0.725 | 1.0 | 0.25 | 0.732 | 0.75 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=on / raw | test | 150 | 0.36 | 0.415 | 0.496 | 0.35 | 0.393 | 0.36 | 1.0 | 0.607 | 0.418 | 0.36 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=on / raw | handwritten | 36 | 0.222 | 0.278 | 0.333 | 0.233 | 0.222 | 0.222 | 1.0 | 0.778 | 0.27 | 0.222 | 0.0 | 0.917 |
| lora_merged_nf4 / rag=on / raw | terminology | 40 | 0.075 | 0.093 | 0.125 | 0.067 | 0.1 | 0.075 | 1.0 | 0.9 | 0.155 | 0.1 | 0.0 | 1.0 |
| int8 / rag=off / raw | test | 150 | 0.227 | 0.143 | 0.26 | 0.05 | 0.827 | 0.2 | 0.973 | 0.007 |  |  | 0.073 |  |
| int8 / rag=off / raw | handwritten | 36 | 0.083 | 0.073 | 0.101 | 0.05 | 0.667 | 0.083 | 0.972 | 0.0 |  |  | 0.028 |  |
| int8 / rag=off / raw | terminology | 40 | 0.675 | 0.723 | 0.668 | 0.767 | 0.575 | 0.425 | 1.0 | 0.0 |  |  | 0.05 |  |
| int8 / rag=on / raw | test | 150 | 0.653 | 0.603 | 0.677 | 0.543 | 0.98 | 0.647 | 0.993 | 0.013 | 0.845 | 0.193 | 0.007 | 1.0 |
| int8 / rag=on / raw | handwritten | 36 | 0.611 | 0.57 | 0.561 | 0.577 | 0.944 | 0.611 | 0.972 | 0.0 | 0.87 | 0.194 | 0.0 | 0.917 |
| int8 / rag=on / raw | terminology | 40 | 0.75 | 0.768 | 0.686 | 0.833 | 0.9 | 0.7 | 0.925 | 0.05 | 0.645 | 0.475 | 0.025 | 1.0 |
| lora_fp16 / rag=off / raw | test | 150 | 0.48 | 0.278 | 0.521 | 0.083 | 0.88 | 0.48 | 1.0 | 0.013 |  |  | 0.0 |  |
| lora_fp16 / rag=off / raw | handwritten | 36 | 0.278 | 0.244 | 0.382 | 0.133 | 0.778 | 0.278 | 1.0 | 0.056 |  |  | 0.0 |  |
| lora_fp16 / rag=off / raw | terminology | 40 | 0.55 | 0.462 | 0.499 | 0.433 | 0.65 | 0.5 | 1.0 | 0.05 |  |  | 0.0 |  |
| qlora_nf4 / rag=off / raw | test | 150 | 0.467 | 0.3 | 0.469 | 0.164 | 0.88 | 0.46 | 1.0 | 0.0 |  |  | 0.0 |  |
| qlora_nf4 / rag=off / raw | handwritten | 36 | 0.278 | 0.255 | 0.406 | 0.133 | 0.75 | 0.278 | 1.0 | 0.083 |  |  | 0.0 |  |
| qlora_nf4 / rag=off / raw | terminology | 40 | 0.45 | 0.408 | 0.418 | 0.4 | 0.6 | 0.4 | 1.0 | 0.175 |  |  | 0.0 |  |
| fp16 / rag=on / raw | test | 150 | 0.66 | 0.596 | 0.683 | 0.526 | 0.967 | 0.647 | 0.993 | 0.013 | 0.844 | 0.24 | 0.0 | 1.0 |
| fp16 / rag=on / raw | handwritten | 36 | 0.611 | 0.57 | 0.561 | 0.577 | 0.917 | 0.611 | 0.972 | 0.0 | 0.85 | 0.222 | 0.0 | 0.917 |
| fp16 / rag=on / raw | terminology | 40 | 0.775 | 0.772 | 0.695 | 0.833 | 0.9 | 0.725 | 0.975 | 0.05 | 0.631 | 0.525 | 0.025 | 1.0 |

## Robustness: accuracy by prompt length

| config | handwritten | long1024 | long2048 | long512 | terminology | test |
|---|---|---|---|---|---|---|
| fp16 / rag=off / raw | 0.111 | 0.133 | 0.1 | 0.1 | 0.65 | 0.213 |
| fp16 / rag=on / raw | 0.611 | 0.533 | 0.667 | 0.6 | 0.775 | 0.66 |
| int8 / rag=off / raw | 0.083 | 0.167 | 0.167 | 0.133 | 0.675 | 0.227 |
| int8 / rag=on / raw | 0.611 | 0.5 | 0.633 | 0.6 | 0.75 | 0.653 |
| lora_fp16 / rag=off / raw | 0.278 | 0.3 | 0.167 | 0.333 | 0.55 | 0.48 |
| lora_fp16 / rag=on / raw | 0.667 | 0.7 | 0.667 | 0.867 | 0.35 | 0.72 |
| lora_merged_nf4 / rag=off / raw | 0.278 | 0.267 | 0.267 | 0.167 | 0.325 | 0.36 |
| lora_merged_nf4 / rag=on / raw | 0.222 | 0.7 | 0.6 | 0.767 | 0.075 | 0.36 |
| nf4 / rag=off / raw | 0.083 | 0.1 | 0.033 | 0.1 | 0.55 | 0.193 |
| nf4 / rag=on / graph | 0.417 |  |  |  |  | 0.307 |
| nf4 / rag=on / raw | 0.583 | 0.5 | 0.433 | 0.633 | 0.3 | 0.54 |
| qlora_nf4 / rag=off / raw | 0.278 | 0.367 | 0.133 | 0.167 | 0.45 | 0.467 |
| qlora_nf4 / rag=on / graph | 0.778 |  |  |  |  | 0.76 |
| qlora_nf4 / rag=on / raw | 0.833 | 0.7 | 0.6 | 0.9 | 0.725 | 0.92 |

## Robustness: OOD behaviour

| config | n | refusal_rate | false_answer_rate | dose_hallucination_rate | dose_refusal_rate |
|---|---|---|---|---|---|
| nf4 / rag=on / raw | 40 | 0.775 | 0.062 | 0.375 | 0.125 |
| fp16 / rag=off / raw | 40 | 0.55 | 0.344 | 0.75 | 0.125 |
| lora_fp16 / rag=on / raw | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=off / raw | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| nf4 / rag=off / raw | 40 | 0.625 | 0.312 | 0.625 | 0.375 |
| qlora_nf4 / rag=on / graph | 40 | 0.975 | 0.031 | 0.0 | 1.0 |
| qlora_nf4 / rag=on / raw | 40 | 0.95 | 0.062 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=on / raw | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| int8 / rag=off / raw | 40 | 0.5 | 0.375 | 0.875 | 0.0 |
| nf4 / rag=on / graph | 40 | 1.0 | 0.0 | 0.0 | 1.0 |
| int8 / rag=on / raw | 40 | 0.4 | 0.5 | 0.5 | 0.0 |
| lora_fp16 / rag=off / raw | 40 | 0.575 | 0.312 | 0.0 | 0.125 |
| qlora_nf4 / rag=off / raw | 40 | 0.875 | 0.156 | 0.0 | 1.0 |
| fp16 / rag=on / raw | 40 | 0.375 | 0.531 | 0.625 | 0.0 |

## Decision layer (graph pipeline)

| config | suite | n | accuracy | false_accept_rate | false_fallback_rate | retry_rate | citation_correct_rate |
|---|---|---|---|---|---|---|---|
| qlora_nf4 / rag=on / graph | test | 150 | 0.76 | 0.04 | 0.2 | 0.2 | 0.783 |
| qlora_nf4 / rag=on / graph | handwritten | 36 | 0.778 | 0.0 | 0.222 | 0.222 | 0.964 |
| nf4 / rag=on / graph | test | 150 | 0.307 | 0.1 | 0.593 | 0.507 | 0.115 |
| nf4 / rag=on / graph | handwritten | 36 | 0.417 | 0.028 | 0.556 | 0.528 | 0.188 |

## Efficiency

| variant | rag | gpu | gpu_mem_after_load_gb | disk_mb | n_params_m | n_prompts | ttft_ms_p50 | ttft_ms_p95 | tpot_ms_mean | latency_s_mean | tokens_in_mean | tokens_out_mean | throughput_tok_s_single | energy_wh_total | energy_wh_per_query | gpu_peak_gb_single | throughput_tok_s_batch8 | gpu_peak_gb_batch8 | cost_usd_per_1k_queries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| int8 | True | Tesla T4 | 0.631 | 631.525 | 494.033 | 20 | 334.442 | 342.695 | 160.839 | 12.817 | 931.35 | 78.6 | 6.133 | 2.448 | 0.122 | 0.798 | 27.782 | 1.888 | 1.246 |
| lora_merged_nf4 | True | Tesla T4 | 0.457 | 457.349 | 315.119 | 20 | 82.179 | 87.163 | 47.913 | 3.895 | 931.35 | 80.7 | 20.717 | 0.876 | 0.044 | 0.624 | 109.831 | 0.856 | 0.379 |
| nf4 | True | Tesla T4 | 0.457 | 457.349 | 315.119 | 20 | 101.873 | 110.02 | 53.236 | 4.299 | 931.35 | 79.9 | 18.587 | 0.962 | 0.048 | 0.625 | 85.821 | 1.71 | 0.418 |
| qlora_nf4 | True | Tesla T4 | 0.493 | 539.265 | 323.918 | 20 | 183.101 | 197.002 | 93.159 | 9.582 | 931.35 | 101.85 | 10.629 | 1.991 | 0.1 | 0.66 | 58.772 | 1.746 | 0.932 |
| fp16 | True | Tesla T4 | 0.988 | 988.099 | 494.033 | 20 | 44.926 | 46.216 | 32.957 | 2.629 | 931.35 | 79.45 | 30.219 | 0.798 | 0.04 | 1.156 | 167.46 | 1.384 | 0.256 |
| lora_fp16 | True | Tesla T4 | 0.988 | 988.099 | 494.033 | 20 | 45.465 | 46.672 | 34.266 | 3.253 | 931.35 | 94.75 | 29.129 | 0.965 | 0.048 | 1.156 | 168.05 | 1.384 | 0.316 |

## Perplexity (held-out agri text)

| variant | source | n_texts | n_tokens | perplexity |
|---|---|---|---|---|
| fp16 | data/heldout_agri_text.txt | 25 | 480 | 33.366 |
| lora_merged_nf4 | data/heldout_agri_text.txt | 25 | 480 | 49.268 |
| int8 | data/heldout_agri_text.txt | 25 | 480 | 33.155 |
| lora_fp16 | data/heldout_agri_text.txt | 25 | 480 | 32.841 |
| nf4 | data/heldout_agri_text.txt | 25 | 480 | 47.66 |
| qlora_nf4 | data/heldout_agri_text.txt | 25 | 480 | 46.683 |

## Automatic findings: where the average hides failures

- **fp16 / rag=off / raw**: head F1 0.24 vs rare F1 0.05 (gap 0.19); average accuracy 0.21 hides it.
- **fp16 / rag=off / raw**: answers 34% of out-of-distribution questions instead of refusing.
- **fp16 / rag=off / raw**: unsupported dose numbers in 7% of in-scope answers.
- **lora_merged_nf4 / rag=off / raw**: head F1 0.43 vs rare F1 0.18 (gap 0.25); average accuracy 0.36 hides it.
- **nf4 / rag=off / raw**: head F1 0.25 vs rare F1 0.00 (gap 0.25); average accuracy 0.19 hides it.
- **nf4 / rag=off / raw**: answers 31% of out-of-distribution questions instead of refusing.
- **nf4 / rag=off / raw**: unsupported dose numbers in 35% of in-scope answers.
- **qlora_nf4 / rag=on / raw**: accuracy 0.92 on short prompts falls to 0.60 at ~2048 tokens.
- **int8 / rag=off / raw**: head F1 0.26 vs rare F1 0.05 (gap 0.21); average accuracy 0.23 hides it.
- **int8 / rag=off / raw**: answers 38% of out-of-distribution questions instead of refusing.
- **int8 / rag=off / raw**: unsupported dose numbers in 7% of in-scope answers.
- **int8 / rag=on / raw**: answers 50% of out-of-distribution questions instead of refusing.
- **int8 / rag=on / raw**: unsupported dose numbers in 1% of in-scope answers.
- **lora_fp16 / rag=off / raw**: head F1 0.52 vs rare F1 0.08 (gap 0.44); average accuracy 0.48 hides it.
- **lora_fp16 / rag=off / raw**: accuracy 0.48 on short prompts falls to 0.17 at ~2048 tokens.
- **lora_fp16 / rag=off / raw**: answers 31% of out-of-distribution questions instead of refusing.
- **qlora_nf4 / rag=off / raw**: head F1 0.47 vs rare F1 0.16 (gap 0.30); average accuracy 0.47 hides it.
- **qlora_nf4 / rag=off / raw**: accuracy 0.47 on short prompts falls to 0.13 at ~2048 tokens.
- **fp16 / rag=on / raw**: head F1 0.68 vs rare F1 0.53 (gap 0.16); average accuracy 0.66 hides it.
- **fp16 / rag=on / raw**: answers 53% of out-of-distribution questions instead of refusing.

## Retrieval

| suite | config | recall@1 | recall@3 | recall@5 | mrr | ndcg@5 | n | latency_ms |
|---|---|---|---|---|---|---|---|---|
| test | dense | 0.873 | 0.967 | 0.993 | 0.924 | 0.903 | 150 | 3.999 |
| test | bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 0.956 | 150 | 0.183 |
| test | hybrid | 0.913 | 1.0 | 1.0 | 0.956 | 0.911 | 150 | 4.097 |
| test | hybrid_no_crop_filter | 0.893 | 1.0 | 1.0 | 0.946 | 0.929 | 150 | 4.049 |
| test | hybrid_no_alias_expansion | 0.913 | 1.0 | 1.0 | 0.956 | 0.911 | 150 | 4.029 |
| handwritten | dense | 0.861 | 0.861 | 0.917 | 0.874 | 0.838 | 36 | 4.135 |
| handwritten | bm25 | 0.889 | 0.917 | 0.944 | 0.91 | 0.888 | 36 | 0.214 |
| handwritten | hybrid | 0.889 | 0.917 | 0.944 | 0.908 | 0.874 | 36 | 4.293 |
| handwritten | hybrid_no_crop_filter | 0.833 | 0.889 | 0.889 | 0.856 | 0.836 | 36 | 4.236 |
| handwritten | hybrid_no_alias_expansion | 0.889 | 0.917 | 0.944 | 0.908 | 0.874 | 36 | 4.28 |
| terminology | dense | 0.95 | 1.0 | 1.0 | 0.971 | 0.95 | 40 | 4.616 |
| terminology | bm25 | 1.0 | 1.0 | 1.0 | 1.0 | 0.978 | 40 | 0.212 |
| terminology | hybrid | 0.975 | 1.0 | 1.0 | 0.988 | 0.966 | 40 | 4.755 |
| terminology | hybrid_no_crop_filter | 0.975 | 0.975 | 0.975 | 0.975 | 0.944 | 40 | 4.734 |
| terminology | hybrid_no_alias_expansion | 0.975 | 1.0 | 1.0 | 0.988 | 0.963 | 40 | 4.528 |
| long1024 | dense | 0.3 | 0.733 | 0.933 | 0.536 | 0.636 | 30 | 162.289 |
| long1024 | bm25 | 0.367 | 0.767 | 0.933 | 0.584 | 0.668 | 30 | 8.444 |
| long1024 | hybrid | 0.433 | 0.833 | 0.933 | 0.64 | 0.697 | 30 | 167.909 |
| long1024 | hybrid_no_crop_filter | 0.633 | 0.7 | 0.833 | 0.698 | 0.706 | 30 | 171.522 |
| long1024 | hybrid_no_alias_expansion | 0.433 | 0.833 | 0.933 | 0.64 | 0.697 | 30 | 166.358 |

## Chunking x embedding ablation

| embedder | chunking | max_words | n_chunks | build_s | recall@1 | recall@3 | recall@5 | mrr | ndcg@5 | n | latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BAAI/bge-small-en-v1.5 | structured | 60 | 112 | 1.009 | 0.909 | 0.984 | 0.989 | 0.946 | 0.904 | 186 | 4.336 |
| BAAI/bge-small-en-v1.5 | structured | 120 | 112 | 0.68 | 0.909 | 0.984 | 0.989 | 0.946 | 0.904 | 186 | 4.659 |
| BAAI/bge-small-en-v1.5 | structured | 200 | 112 | 0.656 | 0.909 | 0.984 | 0.989 | 0.946 | 0.904 | 186 | 4.486 |
| BAAI/bge-small-en-v1.5 | naive | 120 | 28 | 0.406 | 0.882 | 0.995 | 1.0 | 0.939 | 0.928 | 186 | 4.335 |
| BAAI/bge-small-en-v1.5 | naive | 60 | 56 | 0.399 | 0.887 | 0.978 | 0.995 | 0.934 | 0.904 | 186 | 4.449 |
| BAAI/bge-small-en-v1.5 | naive | 200 | 19 | 0.296 | 0.86 | 0.995 | 1.0 | 0.928 | 0.947 | 186 | 4.277 |

## Retrieval load test

| mode | concurrency | queries | qps | p50_ms | p95_ms |
|---|---|---|---|---|---|
| local | 1 | 150 | 242.762 | 3.918 | 5.402 |
| local | 4 | 150 | 477.21 | 7.891 | 13.291 |
| local | 8 | 150 | 644.315 | 11.832 | 18.075 |
| local | 16 | 150 | 634.762 | 20.685 | 42.079 |