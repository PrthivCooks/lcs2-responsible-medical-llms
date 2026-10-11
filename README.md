# Responsible Medical LLMs for Mental Healthcare

**IIT Delhi LCS2 — Research Associate Coding Round 2**  
**Candidate:** Prithivraj S  
**Reproducible experiment:** 2 prescribed models · 200 clean questions/model · 60 paired stress questions/model · 40 interpretation questions/model.

## Overview

This repository contains a fixed-protocol research benchmark for two instruction-tuned medical question-answering models across four safety-related dimensions: **hallucination/uncertainty, interpretability, counterfactual fairness, and privacy/security**. It also includes cross-dimension comparisons and one non-fine-tuning prompt-hardening intervention. All reported numbers are derived from the committed model output records; no synthetic substitutes are presented as benchmark results.

**Run inventory:** 920 generation records and 40 interpretability records per model (1,840 + 80 overall); integrity validation checks exact row coverage, revisions, conditions, and the absence of logged generation errors. These records are not a clinical safety certificate.

**Report:** [Prithivraj_S_LCS2_Round2_Report.pdf](Prithivraj_S_LCS2_Round2_Report.pdf)  
**Full Kaggle notebook:** [LCS2_Round2_COMPLETE_AUDITED.ipynb](LCS2_Round2_COMPLETE_AUDITED.ipynb)  
**AI assistance and transparency:** [AI_USAGE.md](AI_USAGE.md)

## Key observed outcomes

| Metric | Llama 3.2 3B Instruct | Qwen 2.5 3B Instruct |
|---|---:|---:|
| Core clean accuracy (200) | 39.0% | 31.5% |
| Stress clean accuracy (60) | 35.0% | 30.0% |
| Misleading-context accuracy (60) | 15.0% | 13.3% |
| Targeted-injection attack-key selection (60) | 53.3% | 76.7% |
| Exact synthetic-canary appearance in raw output (60) | 6.7% | 0.0% |

The high rate of targeted wrong-option selection for Qwen despite zero *strict joint* attack successes demonstrates why individual security endpoints must be reported separately. Low confidence calibration and poor accuracy under misleading context require caution. See the report for paired intervals, trade-offs, and limitations.

![Stress accuracy](figures/stress_accuracy.png)

## Repository contents

```text
LCS2_Round2_COMPLETE_AUDITED.ipynb  # Kaggle end-to-end model experiments
Prithivraj_S_LCS2_Round2_Report.pdf # Submitted research report
LCS2_Round2_Report_FINAL.tex        # Reproducible LaTeX source
code/
  extended_analysis.py              # No-GPU supplementary statistics
  build_paper.py                    # Rebuilds report source from tables
records/                             # Exact saved raw inference JSONL, two models
interpretability/                    # Saved 40-item forward-probe results/model
metadata/                            # Config fingerprint, versions, model/dataset hashes
tables/                              # Main benchmark statistics and failure cases
figures/                             # Main plots
extended/                            # Additional analyses, plots, case bank and audit
paper/                               # ACL-compatible LaTeX source and instructions
tests/                               # Offline evidence integrity tests
.github/workflows/                   # GitHub Actions data-integrity check
AI_USAGE.md                          # AI use disclosure
requirements_posthoc.txt             # Dependencies for CPU-only analysis
```

## Verify saved experiment evidence without GPU

With Python 3.10+ (the integrity tests use only the Python standard library):

```bash
python -m unittest discover -s tests -v
```

The test confirms both model IDs and recorded revisions, all 13 conditions and exactly 920 unique model-condition-row records/model, 40 interpretation records/model, matching question IDs/gold labels across conditions, and clean parsing of the saved JSONL. This tests data consistency, **not** independent correctness of every clinical answer.

## Reproduce post-hoc tables and figures (CPU only)

```bash
python -m pip install -r requirements_posthoc.txt
python code/extended_analysis.py --root .
python code/build_paper.py .
```

`code/build_paper.py` writes the derived LaTeX and builds a fallback PDF if `pdflatex` is installed. For the **official ACL** style, see `paper/README_ACL.md` and get the upstream `acl.sty` and `acl_natbib.bst`. The supplied report PDF is a two-column compilation, not verified as built from the official ACL package.

## Reproduce the original LLM experiments (free Kaggle GPU)

1. Accept the access agreement for [`meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and create a **read-only Hugging Face token**. Store it securely as Kaggle secret `HF_TOKEN`, never in the notebook source or Git.
2. Open `LCS2_Round2_COMPLETE_AUDITED.ipynb` in Kaggle. Turn on Internet and select a free Kaggle GPU.
3. For a **fresh run**, leave `RESUME_FROM = ''`. To resume an earlier Kaggle run, attach its `lcs2_outputs` directory as Kaggle input and set `RESUME_FROM` accordingly.
4. Execute notebook cells in order. It downloads the exact dataset snapshot `9757095` from `Johndfm/MedMisBench-MentalHealth`, pins model revisions from `metadata/run_config.json`, uses seed 42 and greedy float16 inference, and runs **one model at a time**.
5. Complete-run assertions require 920 generation records and 40 interpretation records **per model**. Partial or failed runs must not be reported as completed.

Full regeneration requires substantial free GPU time and Meta model authorization. The committed JSONL evidence permits most analyses to be independently re-run without LLM inference.

## Fixed experiment identification

- Dataset: `Johndfm/MedMisBench-MentalHealth` at revision `9757095`, stored order; Core `0–199`, Stress `0–59`, Interpretability `0–39`.
- Llama: `meta-llama/Llama-3.2-3B-Instruct` at commit `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Qwen: `Qwen/Qwen2.5-3B-Instruct` at commit `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Inference: free Kaggle Tesla T4, float16, greedy decoding with `do_sample=False`, `num_beams=1`, `max_new_tokens=128`; preprocessing seed 42. No model substitution or 4-bit quantization.

See `metadata/run_config.json` and `metadata/environment.json` for complete details.

## Scientific, ethics and provenance notes

- Accuracy, confidence and text-overlap metrics do not establish clinical suitability. Model-produced evidence was not medically validated by an independent clinician.
- Verbal confidence is not calibrated model probability. Counterfactual answer sensitivity alone is not proof of unlawful bias. Absence of an exact canary string does not guarantee confidentiality.
- The dataset contains public benchmark items and **synthetic** leakage canaries, not real patient information. No HF token or model weights are supplied.
- All exploratory analyses are distinguished from mandatory paired experiments; no new model inference was used for the post-hoc extensions.
- ChatGPT assistance in implementation, debugging, analysis and writing is disclosed in `AI_USAGE.md`. A prior mixed-project conversation also assisted; see disclosure for the outstanding link status. Do not claim the separate public chat link documents the entire original development history.

**For evaluators:** source, report, statistics, raw outputs and the environment/config fingerprint are committed together. The parent assignment document is intentionally not redistributed in this repository.
