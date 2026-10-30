# Evaluation report (auto-generated)

## Quality (raw model output; accuracy / macro-F1 / head vs rare)

| config | suite | n | accuracy | macro_f1 | head_f1 | rare_f1 | em_crop | em_both | parse_ok_rate | refusal_rate | grounded_mean | citation_correct_rate | dose_violation_rate | retrieval_hit_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_fp16 / rag=off / raw | test | 15 | 0.6 | 0.596 | 0.794 | 0.0 | 0.933 | 0.6 | 1.0 | 0.0 |  |  | 0.0 |  |
| lora_fp16 / rag=off / raw | handwritten | 15 | 0.333 | 0.211 | 0.396 | 0.0 | 0.867 | 0.333 | 1.0 | 0.0 |  |  | 0.0 |  |
| lora_fp16 / rag=off / raw | terminology | 15 | 0.6 | 0.533 | 0.567 | 0.489 | 0.6 | 0.6 | 1.0 | 0.0 |  |  | 0.0 |  |
| lora_fp16 / rag=on / raw | test | 15 | 0.8 | 0.792 | 0.722 | 1.0 | 0.867 | 0.8 | 1.0 | 0.133 | 0.867 | 0.667 | 0.0 | 1.0 |
| lora_fp16 / rag=on / raw | handwritten | 15 | 0.8 | 0.756 | 0.708 | 0.81 | 0.933 | 0.8 | 1.0 | 0.067 | 0.909 | 0.8 | 0.0 | 0.867 |
| lora_fp16 / rag=on / raw | terminology | 15 | 0.533 | 0.643 | 0.625 | 0.667 | 0.533 | 0.533 | 1.0 | 0.467 | 0.57 | 0.533 | 0.0 | 1.0 |
| qlora_nf4 / rag=off / raw | test | 15 | 0.533 | 0.542 | 0.722 | 0.0 | 0.933 | 0.533 | 1.0 | 0.0 |  |  | 0.0 |  |
| qlora_nf4 / rag=off / raw | handwritten | 15 | 0.267 | 0.227 | 0.425 | 0.0 | 0.733 | 0.267 | 1.0 | 0.067 |  |  | 0.0 |  |
| qlora_nf4 / rag=off / raw | terminology | 15 | 0.4 | 0.371 | 0.4 | 0.333 | 0.467 | 0.4 | 1.0 | 0.133 |  |  | 0.0 |  |
| int8 / rag=on / raw | test | 15 | 0.6 | 0.634 | 0.679 | 0.5 | 1.0 | 0.6 | 1.0 | 0.0 | 0.897 | 0.267 | 0.0 | 1.0 |
| int8 / rag=on / raw | handwritten | 15 | 0.533 | 0.422 | 0.5 | 0.333 | 1.0 | 0.533 | 1.0 | 0.0 | 0.779 | 0.133 | 0.0 | 0.867 |
| int8 / rag=on / raw | terminology | 15 | 0.467 | 0.456 | 0.381 | 0.556 | 0.8 | 0.4 | 1.0 | 0.133 | 0.604 | 0.467 | 0.0 | 1.0 |
| qlora_nf4 / rag=on / raw | test | 15 | 0.933 | 0.875 | 0.833 | 1.0 | 1.0 | 0.933 | 1.0 | 0.0 | 1.0 | 0.733 | 0.0 | 1.0 |
| qlora_nf4 / rag=on / raw | handwritten | 15 | 0.8 | 0.733 | 0.708 | 0.762 | 1.0 | 0.8 | 1.0 | 0.0 | 0.97 | 0.8 | 0.0 | 0.867 |
| qlora_nf4 / rag=on / raw | terminology | 15 | 0.667 | 0.618 | 0.831 | 0.333 | 0.733 | 0.667 | 1.0 | 0.267 | 0.733 | 0.733 | 0.0 | 1.0 |
| nf4 / rag=on / raw | test | 15 | 0.467 | 0.399 | 0.532 | 0.0 | 0.533 | 0.4 | 0.933 | 0.267 | 0.55 | 0.067 | 0.0 | 1.0 |
| nf4 / rag=on / raw | handwritten | 15 | 0.533 | 0.533 | 0.375 | 0.714 | 0.533 | 0.4 | 1.0 | 0.267 | 0.3 | 0.4 | 0.0 | 0.867 |
| nf4 / rag=on / raw | terminology | 15 | 0.333 | 0.39 | 0.517 | 0.222 | 0.4 | 0.267 | 1.0 | 0.467 | 0.319 | 0.267 | 0.067 | 1.0 |
| nf4 / rag=off / raw | test | 15 | 0.2 | 0.181 | 0.241 | 0.0 | 0.867 | 0.2 | 1.0 | 0.0 |  |  | 0.333 |  |
| nf4 / rag=off / raw | handwritten | 15 | 0.133 | 0.093 | 0.175 | 0.0 | 0.667 | 0.133 | 0.867 | 0.133 |  |  | 0.2 |  |
| nf4 / rag=off / raw | terminology | 15 | 0.733 | 0.738 | 0.792 | 0.667 | 0.333 | 0.333 | 0.933 | 0.133 |  |  | 0.2 |  |
| fp16 / rag=off / raw | test | 15 | 0.267 | 0.193 | 0.258 | 0.0 | 0.867 | 0.267 | 1.0 | 0.0 |  |  | 0.0 |  |
| fp16 / rag=off / raw | handwritten | 15 | 0.2 | 0.152 | 0.161 | 0.143 | 0.733 | 0.2 | 0.933 | 0.0 |  |  | 0.0 |  |
| fp16 / rag=off / raw | terminology | 15 | 0.667 | 0.765 | 0.673 | 0.889 | 0.467 | 0.333 | 1.0 | 0.0 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=off / raw | test | 15 | 0.533 | 0.469 | 0.625 | 0.0 | 0.867 | 0.467 | 1.0 | 0.0 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=off / raw | handwritten | 15 | 0.267 | 0.189 | 0.354 | 0.0 | 0.733 | 0.267 | 1.0 | 0.133 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=off / raw | terminology | 15 | 0.6 | 0.599 | 0.631 | 0.556 | 0.6 | 0.6 | 1.0 | 0.067 |  |  | 0.0 |  |
| fp16 / rag=on / raw | test | 15 | 0.6 | 0.634 | 0.679 | 0.5 | 0.867 | 0.533 | 1.0 | 0.0 | 0.87 | 0.4 | 0.0 | 1.0 |
| fp16 / rag=on / raw | handwritten | 15 | 0.533 | 0.444 | 0.542 | 0.333 | 0.8 | 0.533 | 1.0 | 0.0 | 0.844 | 0.2 | 0.0 | 0.867 |
| fp16 / rag=on / raw | terminology | 15 | 0.6 | 0.546 | 0.539 | 0.556 | 0.867 | 0.533 | 0.933 | 0.067 | 0.716 | 0.533 | 0.0 | 1.0 |
| int8 / rag=off / raw | test | 15 | 0.333 | 0.283 | 0.378 | 0.0 | 0.867 | 0.333 | 0.933 | 0.0 |  |  | 0.067 |  |
| int8 / rag=off / raw | handwritten | 15 | 0.2 | 0.152 | 0.161 | 0.143 | 0.733 | 0.2 | 0.933 | 0.0 |  |  | 0.0 |  |
| int8 / rag=off / raw | terminology | 15 | 0.733 | 0.813 | 0.756 | 0.889 | 0.467 | 0.4 | 0.867 | 0.0 |  |  | 0.0 |  |
| lora_merged_nf4 / rag=on / raw | test | 15 | 0.267 | 0.225 | 0.3 | 0.0 | 0.267 | 0.267 | 1.0 | 0.733 | 0.285 | 0.267 | 0.0 | 1.0 |
| lora_merged_nf4 / rag=on / raw | handwritten | 15 | 0.067 | 0.067 | 0.0 | 0.143 | 0.067 | 0.067 | 1.0 | 0.933 | 0.133 | 0.067 | 0.0 | 0.867 |
| lora_merged_nf4 / rag=on / raw | terminology | 15 | 0.067 | 0.095 | 0.167 | 0.0 | 0.067 | 0.067 | 1.0 | 0.933 | 0.139 | 0.067 | 0.0 | 1.0 |

## Robustness: accuracy by prompt length

| config | handwritten | long1024 | long2048 | long512 | terminology | test |
|---|---|---|---|---|---|---|
| fp16 / rag=off / raw | 0.2 | 0.133 | 0.133 | 0.133 | 0.667 | 0.267 |
| fp16 / rag=on / raw | 0.533 | 0.533 | 0.667 | 0.6 | 0.6 | 0.6 |
| int8 / rag=off / raw | 0.2 | 0.133 | 0.267 | 0.2 | 0.733 | 0.333 |
| int8 / rag=on / raw | 0.533 | 0.533 | 0.667 | 0.6 | 0.467 | 0.6 |
| lora_fp16 / rag=off / raw | 0.333 | 0.267 | 0.2 | 0.333 | 0.6 | 0.6 |
| lora_fp16 / rag=on / raw | 0.8 | 0.733 | 0.6 | 0.867 | 0.533 | 0.8 |
| lora_merged_nf4 / rag=off / raw | 0.267 | 0.133 | 0.2 | 0.133 | 0.6 | 0.533 |
| lora_merged_nf4 / rag=on / raw | 0.067 | 0.533 | 0.533 | 0.133 | 0.067 | 0.267 |
| nf4 / rag=off / raw | 0.133 | 0.133 | 0.067 | 0.2 | 0.733 | 0.2 |
| nf4 / rag=on / raw | 0.533 | 0.667 | 0.4 | 0.6 | 0.333 | 0.467 |
| qlora_nf4 / rag=off / raw | 0.267 | 0.267 | 0.133 | 0.2 | 0.4 | 0.533 |
| qlora_nf4 / rag=on / raw | 0.8 | 0.733 | 0.6 | 0.867 | 0.667 | 0.933 |

## Robustness: OOD behaviour

| config | n | refusal_rate | false_answer_rate | dose_hallucination_rate | dose_refusal_rate |
|---|---|---|---|---|---|
| lora_fp16 / rag=off / raw | 15 | 0.933 | 0.067 |  |  |
| lora_fp16 / rag=on / raw | 15 | 1.0 | 0.0 |  |  |
| qlora_nf4 / rag=off / raw | 15 | 0.933 | 0.067 |  |  |
| int8 / rag=on / raw | 15 | 0.267 | 0.733 |  |  |
| qlora_nf4 / rag=on / raw | 15 | 0.933 | 0.067 |  |  |
| nf4 / rag=on / raw | 15 | 1.0 | 0.0 |  |  |
| nf4 / rag=off / raw | 15 | 0.933 | 0.067 |  |  |
| fp16 / rag=off / raw | 15 | 0.933 | 0.067 |  |  |
| lora_merged_nf4 / rag=off / raw | 15 | 0.933 | 0.067 |  |  |
| fp16 / rag=on / raw | 15 | 0.467 | 0.533 |  |  |
| int8 / rag=off / raw | 15 | 0.8 | 0.2 |  |  |
| lora_merged_nf4 / rag=on / raw | 15 | 1.0 | 0.0 |  |  |

## Decision layer (graph pipeline)

_not run_

## Efficiency

| variant | rag | gpu | gpu_mem_after_load_gb | disk_mb | n_params_m | n_prompts | ttft_ms_p50 | ttft_ms_p95 | tpot_ms_mean | latency_s_mean | tokens_in_mean | tokens_out_mean | throughput_tok_s_single | energy_wh_total | energy_wh_per_query | gpu_peak_gb_single | throughput_tok_s_batch8 | gpu_peak_gb_batch8 | cost_usd_per_1k_queries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nf4 | True | Tesla T4 | 0.457 | 457.349 | 315.119 | 20 | 104.619 | 111.038 | 55.7 | 4.503 | 931.35 | 79.9 | 17.743 | 1.057 | 0.053 | 0.625 | 80.456 | 1.71 | 0.438 |
| int8 | True | Tesla T4 | 0.631 | 631.525 | 494.033 | 20 | 342.907 | 360.66 | 166.896 | 13.295 | 931.35 | 78.6 | 5.912 | 2.682 | 0.134 | 0.798 | 26.678 | 1.888 | 1.293 |
| fp16 | True | Tesla T4 | 0.988 | 988.099 | 494.033 | 20 | 45.717 | 47.293 | 34.195 | 2.727 | 931.35 | 79.45 | 29.134 | 0.859 | 0.043 | 1.156 | 160.804 | 1.384 | 0.265 |
| qlora_nf4 | True | Tesla T4 | 0.493 | 539.265 | 323.918 | 20 | 187.439 | 200.495 | 101.084 | 10.381 | 931.35 | 101.85 | 9.811 | 2.254 | 0.113 | 0.66 | 52.468 | 1.746 | 1.009 |
| lora_merged_nf4 | True | Tesla T4 | 0.457 | 457.349 | 315.119 | 20 | 79.288 | 88.075 | 48.289 | 3.285 | 931.35 | 67.4 | 20.519 | 0.805 | 0.04 | 0.624 | 101.806 | 0.856 | 0.319 |
| lora_fp16 | True | Tesla T4 | 0.988 | 988.099 | 494.033 | 20 | 46.258 | 48.296 | 34.422 | 3.368 | 931.35 | 97.65 | 28.992 | 1.027 | 0.051 | 1.156 | 167.197 | 1.384 | 0.327 |

## Perplexity (held-out agri text)

| variant | source | n_texts | n_tokens | perplexity |
|---|---|---|---|---|
| lora_fp16 | data/heldout_agri_text.txt | 25 | 480 | 33.438 |
| lora_merged_nf4 | data/heldout_agri_text.txt | 25 | 480 | 52.321 |
| int8 | data/heldout_agri_text.txt | 25 | 480 | 33.155 |
| nf4 | data/heldout_agri_text.txt | 25 | 480 | 47.66 |
| qlora_nf4 | data/heldout_agri_text.txt | 25 | 480 | 46.299 |
| fp16 | data/heldout_agri_text.txt | 25 | 480 | 33.366 |

## Automatic findings: where the average hides failures

- **lora_fp16 / rag=off / raw**: head F1 0.79 vs rare F1 0.00 (gap 0.79); average accuracy 0.60 hides it.
- **lora_fp16 / rag=off / raw**: accuracy 0.60 on short prompts falls to 0.20 at ~2048 tokens.
- **lora_fp16 / rag=on / raw**: accuracy 0.80 on short prompts falls to 0.60 at ~2048 tokens.
- **qlora_nf4 / rag=off / raw**: head F1 0.72 vs rare F1 0.00 (gap 0.72); average accuracy 0.53 hides it.
- **qlora_nf4 / rag=off / raw**: accuracy 0.53 on short prompts falls to 0.13 at ~2048 tokens.
- **int8 / rag=on / raw**: head F1 0.68 vs rare F1 0.50 (gap 0.18); average accuracy 0.60 hides it.
- **int8 / rag=on / raw**: answers 73% of out-of-distribution questions instead of refusing.
- **qlora_nf4 / rag=on / raw**: accuracy 0.93 on short prompts falls to 0.60 at ~2048 tokens.
- **nf4 / rag=on / raw**: head F1 0.53 vs rare F1 0.00 (gap 0.53); average accuracy 0.47 hides it.
- **nf4 / rag=off / raw**: head F1 0.24 vs rare F1 0.00 (gap 0.24); average accuracy 0.20 hides it.
- **nf4 / rag=off / raw**: unsupported dose numbers in 33% of in-scope answers.
- **fp16 / rag=off / raw**: head F1 0.26 vs rare F1 0.00 (gap 0.26); average accuracy 0.27 hides it.
- **lora_merged_nf4 / rag=off / raw**: head F1 0.62 vs rare F1 0.00 (gap 0.62); average accuracy 0.53 hides it.
- **lora_merged_nf4 / rag=off / raw**: accuracy 0.53 on short prompts falls to 0.20 at ~2048 tokens.
- **fp16 / rag=on / raw**: head F1 0.68 vs rare F1 0.50 (gap 0.18); average accuracy 0.60 hides it.
- **fp16 / rag=on / raw**: answers 53% of out-of-distribution questions instead of refusing.
- **int8 / rag=off / raw**: head F1 0.38 vs rare F1 0.00 (gap 0.38); average accuracy 0.33 hides it.
- **int8 / rag=off / raw**: unsupported dose numbers in 7% of in-scope answers.
- **lora_merged_nf4 / rag=on / raw**: head F1 0.30 vs rare F1 0.00 (gap 0.30); average accuracy 0.27 hides it.