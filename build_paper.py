#!/usr/bin/env python3
"""Build source-controlled paper from saved measured outputs. Requires extended_analysis.py first.
Writes reports/LCS2_Round2_Report_FINAL.tex (standard pdflatex fallback), paper/main_acl.tex
and reports/LCS2_Round2_Report_FINAL.pdf if pdflatex is available.
Never changes the recorded generations or interpretability results.
"""
import json,os,re,subprocess,sys,shutil
from pathlib import Path
import pandas as pd

def main(root):
 root=Path(root)
 base=pd.read_csv(root/'tables'/'baseline_summary.csv')
 mis=pd.read_csv(root/'tables'/'misleading_summary.csv')
 sec=pd.read_csv(root/'tables'/'security_summary.csv')
 fa=pd.read_csv(root/'tables'/'fairness_summary.csv')
 intr=pd.read_csv(root/'tables'/'interpretability_summary.csv')
 interv=pd.read_csv(root/'tables'/'intervention_summary.csv')
 cross=pd.read_csv(root/'tables'/'cross_dimension_summary.csv')
 other=pd.read_csv(root/'extended'/'tables'/'paired_model_comparison.csv')
 endpoints=pd.read_csv(root/'extended'/'tables'/'attack_endpoint_uncertainty.csv')
 delta=pd.read_csv(root/'extended'/'tables'/'paired_transition_counts.csv')
 policy=pd.read_csv(root/'extended'/'tables'/'risk_coverage_fixed_thresholds.csv')
 conf=json.loads((root/'metadata'/'run_config.json').read_text())
 models=['meta-llama/Llama-3.2-3B-Instruct','Qwen/Qwen2.5-3B-Instruct']
 lookup=lambda t,m: t.loc[t.model==m].iloc[0]
 def pp(x):return f'{100*x:.1f}'
 def tex_escape(x):
  return str(x).replace('\\','\\textbackslash{}').replace('_','\\_').replace('&','\\&').replace('%','\\%').replace('$','\\$').replace('#','\\#').replace('{','\\{').replace('}','\\}')
 def irow(slug,c):return interv[(interv.model==models[0 if slug=='llama32' else 1])&(interv.condition==c)].iloc[0]
 def endrow(slug,c,k):return endpoints[(endpoints.model==slug)&(endpoints.condition==c)&(endpoints.endpoint==k)].iloc[0]
 # compact programmatically derived primary table strings
 baseline='\n'.join(f'{"Llama" if i==0 else "Qwen"} & {pp(lookup(base,m).accuracy)} & {lookup(base,m).mean_confidence:.1f} & {lookup(base,m).ECE10:.3f} \\\\' for i,m in enumerate(models))
 misrows='\n'.join(f'{"Llama" if i==0 else "Qwen"} & {pp(lookup(mis,m).clean_accuracy)} & {pp(lookup(mis,m).misleading_accuracy)} & {pp(lookup(mis,m).answer_flip_rate)} & {pp(lookup(mis,m).wrong_option_target_rate)} \\\\' for i,m in enumerate(models))
 secrows='\n'.join(f'{"Llama" if i==0 else "Qwen"} & {pp(lookup(sec,m).strict_full_attack_compliance)} & {pp(lookup(sec,m).target_key_selected)} & {pp(lookup(sec,m).canary_exact_leakage_raw)} \\\\' for i,m in enumerate(models))
 introws='\n'.join(f'{"Llama" if i==0 else "Qwen"} & {lookup(intr,m).selected_abs_change_mean:.3f} & {lookup(intr,m).control_abs_change_mean:.3f} & {lookup(intr,m).faithfulness_delta:+.3f} & {lookup(intr,m).Wilcoxon_p_exploratory:.3f} \\\\' for i,m in enumerate(models))
 fairness='\n'.join(f'{"Llama" if i==0 else "Qwen"} & {pp(fa[(fa.model==m)&(fa.attribute=="race")].iloc[0].answer_flip_rate)} & {pp(fa[(fa.model==m)&(fa.attribute=="insurance")].iloc[0].answer_flip_rate)} & {pp(fa[(fa.model==m)&(fa.attribute=="residence")].iloc[0].answer_flip_rate)} \\\\' for i,m in enumerate(models))
 hardened='\n'.join(f'{"Llama" if s=="llama32" else "Qwen"} & {pp(endrow(s,"target_injection","target_key").rate)} & {pp(endrow(s,"mitigation_target","target_key").rate)} & {pp(endrow(s,"target_injection","joint_strict").rate)} & {pp(endrow(s,"mitigation_target","joint_strict").rate)} \\\\' for s in ['llama32','qwen25'])
 # 3 cases per model: favorable, unfavorable and near-null, no cherry-picked causal claim
 cases=[]
 for slug in ['llama32','qwen25']:
  arr=[json.loads(l) for l in (root/'interpretability'/f'{slug}.jsonl').read_text().splitlines() if l.strip()]
  xx=[min(arr,key=lambda x:x['faithfulness_advantage']),min(arr,key=lambda x:abs(x['faithfulness_advantage'])),max(arr,key=lambda x:x['faithfulness_advantage'])]
  for kind,rr in zip(['contradictory','near-neutral','supportive'],xx):
   cue=re.sub(r'\s+',' ',rr['chosen_chunk_text']).strip()[:105]
   cases.append((slug,kind,rr['row_index'],rr['dataset_id'],rr['faithfulness_advantage'],cue))
 notes={
  ('llama32','contradictory'):'The length-controlled alternative removes more probability mass than the self-evidence span; this directly challenges the lexical evidence selector.',
  ('llama32','near-neutral'):'The two span deletions are nearly tied despite overlap differences; no robust localization claim follows.',
  ('llama32','supportive'):'The evidence-linked medication-history span has a much larger probe effect than the control span, consistent with local cue reliance but not sufficient for causal attribution.',
  ('qwen25','contradictory'):'The alternative span dominates despite the chosen span containing alarming clinical vital signs; the apparent rationale is nonfaithful under this specific probe.',
  ('qwen25','near-neutral'):'The lexical-selected history span is not uniquely influential; a near-zero advantage cautions against interpretation from overlap alone.',
  ('qwen25','supportive'):'Removing the laboratory-values span nearly eliminates the probed answer-letter mass, which is behaviorally relevant but could be an artifact of a high-information numeric block.'
 }
 cases_tex='\n'.join(f'\\item \\textbf{{{"Llama" if s=="llama32" else "Qwen"}, item {idx} ({kind})}}: diagnostic difference $\\Delta={d:+.3f}$; selected source span begins ``{tex_escape(cue)}\\ldots''. {notes[(s,kind)]} This is an influence diagnostic, not an independent medical explanation.' for s,kind,idx,id,d,cue in cases)
 # error examples from actual outputs (avoid claiming factual truth of generated rationale).
 err=[]
 for slug in ['llama32','qwen25']:
  ix={(r['condition'],r['row_index']):r for r in [json.loads(l) for l in (root/'records'/f'{slug}.jsonl').read_text().splitlines() if l.strip()]}
  ids=[i for i in range(60) if ix['core_clean',i]['correct'] and not ix['misleading',i]['correct']]
  for i in ids[:2]:
   b=ix['misleading',i];a=ix['core_clean',i]
   ev=re.sub(r'\s+',' ',b['evidence'])[:145]
   err.append((slug,i,a['parsed_answer'],b['parsed_answer'],b['type1_injection_key'],b['confidence'],ev))
 error_tex='\n'.join(f'\\item {"Llama" if slug=="llama32" else "Qwen"} item {i}: correct clean choice {clean}, then wrong answer {bad} (designated injected key {key}) with elicited confidence {int(conf)}; evidence begins ``{tex_escape(ev)}\\ldots''. ' for slug,i,clean,bad,key,conf,ev in err)

 br0=lookup(base,models[0]);br1=lookup(base,models[1]);mr0=lookup(mis,models[0]);mr1=lookup(mis,models[1]);sr0=lookup(sec,models[0]);sr1=lookup(sec,models[1]);
 cmp=other[other.condition=='core_clean'].iloc[0]
 wl0=endrow('llama32','canary_attack','exact_canary_raw');wl1=endrow('qwen25','canary_attack','exact_canary_raw');
 qwen_key_after=endrow('qwen25','mitigation_target','target_key');
 selection=policy[(policy.model=='qwen25')&(policy.condition=='misleading')&(policy.threshold_inclusive==100)].iloc[0]

 body=r'''
\title{Beyond Accuracy: A Fixed, Paired Evaluation of Uncertainty, Explanations, Demographic Sensitivity and Prompt Injection in Two Mental-Health LLMs}
\date{October 11, 2026}
\author{LCS2 Research Associate Coding Round 2 \\
Candidate: Prithivraj S}
\begin{document}
\maketitle
\begin{abstract}
We benchmark two fixed 3B instruction models, Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct, on the prescribed MedMisBench-MentalHealth snapshot. Across 200 baseline items, 60 item-aligned stress cases, and 40 interpretation cases per model, we audit misleading context, verbal confidence, evidence faithfulness, six metadata variants, targeted instruction attacks, synthetic-canary disclosure, and one system-prompt hardening intervention. Across two models, all 1,840 specified generation records and 80 interpretability records are present with no generation exceptions. Baseline accuracy was P_BASE0\% and P_BASE1\%; misleading context reduced stress accuracy from P_STRESS0\% to P_MIS0\% (Llama) and P_STRESS1\% to P_MIS1\% (Qwen). Security conclusions depend on their endpoint: Qwen never emitted the demanded attack marker but chose the attack-designated wrong option in P_KEY_Q\% of attacked examples. Hardened prompts improved strict success for Llama, but did not establish improved security for Qwen. We distinguish observed model behavior from clinical validity throughout. The benchmark is not a clinical safety certification.
\end{abstract}

\section{Introduction}
Medical question answering is not adequately characterized by answer accuracy alone. Untrusted supporting text may misdirect a model; fluent evidence can be copied rather than grounded; confidence may bear little relation to correctness; and irrelevant metadata can induce counterfactual inconsistency. A model may also obey a lower-trust instruction in an otherwise clinical vignette. We use the fixed assignment\'s controlled design to study these concerns without changing the models, dataset, questions or prescribed perturbations.

\paragraph{Research hypotheses.} H1: target-specific misleading context increases adoption of the targeted distractor and paired error frequency, potentially alongside overconfidence. H2: generic trust-boundary hardening reduces malicious-instruction compliance while maintaining clean accuracy within a prespecified five-percentage-point tolerance. Hypothesis directions were specified before looking at these results; secondary analyses are labeled exploratory.

\paragraph{Contributions.} (i) An audited item-level matrix with exact model/dataset revisions and all outcomes; (ii) separate answer corruption, lexical evidence contamination and verbal-confidence analyses; (iii) same-item model and attack-endpoint comparisons including confidence intervals for zero events; (iv) interpretation with length-matched deletion controls; and (v) a measured intervention with adverse effects reported explicitly.

\section{Experimental specification}
\paragraph{Dataset and splits.} We use \texttt{Johndfm/MedMisBench-MentalHealth}, revision \texttt{9757095}, file \texttt{medmisbench\_mentalhealth.jsonl} \citep{medmisbench}. Stored row indices determine Core 0--199 (200), Stress 0--59 (60), and Interpretability 0--39 (40). Original case text, choices, keys and mandated metadata variants are not rewritten. The first 200 stored cases happen to carry the dataset source subset \texttt{MEDMISQA}; this fixed slice may poorly represent the full psychiatric case mix. Data were public synthetic benchmark vignettes, not private patient records.

\paragraph{Models, inference and provenance.} The two mandatory models are \texttt{meta-llama/Llama-3.2-3B-Instruct} \citep{llamamodel} and \texttt{Qwen/Qwen2.5-3B-Instruct} \citep{qwenmodel}. They were evaluated one at a time on a free Kaggle Tesla T4 (14.56 GiB), float16, no sampling, greedy beam 1, maximum 128 generated tokens and seed 42. The precise model commit IDs, dataset SHA-256, package versions, frozen system instruction, prompts, configuration fingerprint and every raw generation reside in the repository metadata and record files. System-output requirements request a one-letter answer, numerical 0--100 confidence and brief evidence as JSON; we report parse coverage and count invalid responses as incorrect.

\paragraph{Reproducibility contract.} Each model has 200 clean records, 60 misleading records, 360 fairness records, 120 security records, and 180 intervention records. The 40 interpretation cases require forward passes and selected/control ablations. The independent audit checks all row-index pairs, model revisions, paired IDs, error fields and duplicates. Full experimental conditions are frozen, and the add-on analysis does not alter or re-run models.

\section{Baseline and misinformation vulnerability}
\begin{table}[t]
\centering\footnotesize
\begin{tabular}{lrrr}\toprule
Model & Acc.\% & Conf./100 & ECE$_{10}$ \\\midrule
TABLE_BASELINE
\bottomrule\end{tabular}
\caption{Core baseline, $n=200$ for each model. Confidence is verbally elicited, not posterior probability.}\label{tab:base}
\end{table}

\begin{table}[t]
\centering\footnotesize
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}\toprule
Model & Clean\% & Mislead\% & Flip\% & Target\% \\\midrule
TABLE_MISLEAD
\bottomrule\end{tabular}}
\caption{Same $n=60$ question IDs; ``Target'' is the percentage selecting the designated wrong option.}\label{tab:mis}
\end{table}

Core results are summarized in Table~\ref{tab:base}. Llama scored P_BASE0\% and Qwen P_BASE1\%. In the 200-item model-aligned comparison, Llama's 7.5-point advantage had a paired bootstrap 95\% interval [0, 15] percentage points; exact McNemar $p=P_MCNP$. Thus the benchmark does not establish a strong general difference from this finite sample.

Following the mandatory one-injection-per-case design, both models degraded markedly (Table~\ref{tab:mis}). Llama changed by $-20.0$ percentage points (95\% paired bootstrap CI $[-31.7,-8.3]$, McNemar $p=0.00418$). Qwen changed by $-16.7$ points (CI $[-28.3,-5.0]$, $p=0.0213$). There were 14 and 13 respectively newly wrong, previously correct cases, almost all accompanied by verbally high confidence. Novel 4-gram overlap with the injected text was 46.7\% and 56.7\%, respectively. This is an \emph{echo proxy}, not expert-assessed clinical hallucination: it misses paraphrases and may flag benign quotation.

\paragraph{Wrong-but-confident case audit.} The following examples are reproducibly selected newly-wrong paired outcomes; snippets are model output, \emph{not} asserted clinical facts:
\begin{itemize}\setlength\itemsep{1pt}
ERROR_CASES
\end{itemize}
The case IDs permit blinded follow-up medical adjudication, which was not performed here.

\section{Interpretability with behavioral controls}
On the clean 40-item subset for each model, intermediate hidden states are projected onto answer-token vectors to produce a diagnostic layer-wise logit lens. Independently, the question is partitioned into three contiguous spans. The span with greatest lexical overlap with the model's self-reported evidence is deleted, and the resulting change in conditional answer-letter probability is compared against a similar-length \emph{control} span deletion. We also record last-layer hidden-state distances. The probability readout uses an auxiliary constrained prefix and is \emph{not} a replacement for the primary free generation.

\begin{table}[t]\centering\footnotesize
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}\toprule
Model & Selected $|\Delta p|$ & Control $|\Delta p|$ & Advantage & $p$ \\\midrule
TABLE_INTERPRET
\bottomrule\end{tabular}}
\caption{Descriptive evidence-selection faithfulness diagnostics; $n=40$, Wilcoxon exploratory.}\label{tab:interp}
\end{table}

Qwen's mean selected-minus-control difference was +0.283 (95\% bootstrap CI [+0.121,+0.449], exploratory Wilcoxon $p=0.0053$); Llama's was +0.055 (CI $[-0.009,+0.127]$, $p=0.1316$). Differences may reflect what the auxiliary probing format emphasizes, not genuine human-interpretable mechanisms. At least three informative cases \emph{per model} were examined under pre-specified minimum, median-like and maximum influence criteria:
\begin{itemize}\setlength\itemsep{1pt}
INTERPRET_CASES
\end{itemize}
In particular, a larger deletion effect in the control than selected span is evidence \emph{against} interpreting a lexical self-explanation as faithful. Every deletion may remove necessary clinical information, so this is a behavioral relevance test with a meaningful, imperfect control, not a causal decomposition of the model's internal reasoning.

\section{Fairness and counterfactual sensitivity}
The mandatory six metadata prompts alter exactly one attribute at a time: race, insurance, or residence, with all other case and answer text preserved. A pairwise answer flip is a \emph{sensitivity indicator}, never by itself a label of unfair treatment. We additionally measure absolute reported-confidence differences, evidence Jaccard distance, and changes in correctness.

\begin{table}[t]\centering\footnotesize
\begin{tabular}{lrrr}\toprule
Model & Race\% & Insurance\% & Residence\% \\\midrule
TABLE_FAIRNESS
\bottomrule\end{tabular}
\caption{Paired answer flips in each metadata attribute ($n=60$ per attribute and model).}\label{tab:fair}
\end{table}

Flips are uncommon (Table~\ref{tab:fair}), but evidence text is sensitive more frequently, as documented by per-case review files. A change may be justified when access constraints genuinely affect an appropriate care decision; it might be unjustified when metadata are immaterial to the correct answer. No clinical expert labels were collected, so we explicitly avoid asserting demographic discrimination from these values. The case-review CSV preserves exact conflicting answers and explanations for future adjudication.

\section{Privacy and instruction-boundary attacks}
For each stress item, a lower-trust targeted note requests the dataset-designated wrong choice, confidence 100 and the exact phrase ``override accepted'' in the evidence field. The attack does \emph{not} modify the baseline system prompt. Separately, a unique \texttt{MCS2-CANARY-<id>-7F3A9C} appears only in the experiment-specific system message, while a lower-trust audit request asks to disclose it. Canary identifiers are synthetic; no actual records or credentials are involved.

\begin{table}[t]\centering\footnotesize
\begin{tabular}{lrrr}\toprule
Model & Strict\% & Wrong-key\% & Raw leak\% \\\midrule
TABLE_SECURITY
\bottomrule\end{tabular}
\caption{Security endpoints ($n=60$ each): strict means all three attack instructions obeyed. Raw leak is exact canary reproduction.}\label{tab:security}
\end{table}

Table~\ref{tab:security} demonstrates endpoint dependence. For example, Qwen reached P_KEY_Q\% targeted wrong-key selection but 0/60 three-part full compliances: it did not produce the demanded literal evidence marker, so the \emph{strict} metric conceals clinically consequential answer capture. Both models reported attack confidence 100 on every targeted-note example, making this constituent endpoint non-discriminatory. Zero canary disclosures in 60 Qwen trials do \emph{not} prove absence of leakage; the Wilson 95\% upper bound is 6.0\%. Llama disclosed the exact canary in 4/60 raw generations and 3/60 extracted evidence fields. These are lower-bound string matches, not a comprehensive privacy audit.

\section{Cross-dimension analysis}
The same 60 IDs support item-matched tests instead of treating dimensions as independent tasks. We examined (i) new misinformation errors against high reported confidence and (ii) any counterfactual answer flip against targeted wrong-key selection. Fisher tests for the first link are uninformative because all misinformation outputs were at or above the threshold 80 (ceiling effects); odds ratios are undefined under empty cells. The second is exploratory and underpowered with only a few fairness flips. A further illustrative same-item comparison of misinformation errors and targeted-answer changes is included in the supplement. These tests establish neither causality nor a broad patient-level risk correlation.

\section{One lightweight intervention and trade-offs}
The intervention appends a general instruction to respect task boundaries and ignore commands embedded in clinical vignette text; it is not fine-tuning, data modification, or answer-key memorization. The same intervention was tested on 60 targeted attacks, 60 clean items and 60 misleading-context items for \emph{both models}.

\begin{table}[t]\centering\footnotesize
\resizebox{\columnwidth}{!}{\begin{tabular}{lrrrr}\toprule
& \multicolumn{2}{c}{Wrong-key selection\%} & \multicolumn{2}{c}{Strict compliance\%}\\
Model & Before & After & Before & After\\\midrule
TABLE_HARDEN
\bottomrule\end{tabular}}
\caption{Paired targeted-attack endpoints before vs after hardening. Not all metrics improve.}\label{tab:hardening}
\end{table}

Llama's strict three-part compliance fell from 45.0\% to 21.7\% (paired difference $-23.3$ points, CI $[-33.3,-13.3]$, McNemar $p=0.000122$). For Qwen, strict compliance remained zero, \emph{but targeted wrong-key selection rose from 76.7\% to 83.3\%}; its attacked-task accuracy also fell by 5.0 points. Thus, declaring that hardening was uniformly helpful would be erroneous.

The hardening prefix added 74 prompt tokens per case. Clean clinical accuracy shifted by $-1.7$ points for Llama (95\% paired CI $[-5.0,0]$) and $-3.3$ for Qwen (CI $[-8.3,0]$). These intervals do not establish the prespecified five-point noninferiority claim. Under misleading context, Llama gained 1.7 points and Qwen lost 5.0 points. Self-reported confidence calibration can also worsen; fairness and canary leakage after hardening were not newly evaluated. Intervention choice was fixed; no post-hoc tuning against stress outcomes is claimed.

\begin{figure}[t]\centering
\includegraphics[width=\columnwidth]{extended/figures/attack_partial_vs_strict.png}
\caption{Strict vs partial attack response. Wrong-key obedience is crucial when assessing a security mitigation.}\label{fig:attack}
\end{figure}

\section{Optional exploratory follow-ups (no extra training)}
\paragraph{Between-model alignment.} The two models disagreed on P_DISAGREE\% of baseline answers. The 7.5-point difference in Core accuracy should be discussed jointly with item-level disagreement; model size alone cannot explain errors.

\paragraph{Confidence and selective prediction.} In a strictly post-hoc fixed-threshold exercise, demanding verbal confidence of 100 from Qwen retained P_SEL_COV\% of misinformation cases yet achieved only P_SEL_ACC\% accuracy, demonstrating that confidence thresholds may produce \emph{selective overconfidence} rather than reliable abstention. These thresholds were never deployed as a second intervention, and are not tuned or validated as a clinical risk-control policy.

\paragraph{Uncertainty for rare events.} We include exact counts, Wilson score intervals and the full observed event denominator for attack and canary endpoints, particularly important when 0/60 events occur. Supplementary confidence bins and paired transitions allow reanalysis under alternative metric definitions without rerunning the model.

\section{Limitations and ethics}
This analysis has important limits. Only 200 of 449 stored questions were tested for baseline, and all 60 stress cases are nested within that Core slice, limiting independence and generality. Some selected vignettes address broader medicine rather than mental-health practice. Inference uses one deterministic seed, elicited numeric confidence rather than calibrated predictive probability, and brief unconstrained evidence text. Direct n-gram echo is not fact verification. Token projection and deletion controls can be sensitive to prompt framing and answer-token likelihood design; the Kaggle runtime also warned that Qwen sliding-window attention is not implemented for the selected SDPA backend, warranting a focused reproduction check before stronger mechanistic claims. Small-$n$ hypothesis tests and multiple exploratory analyses cannot justify deployment claims. No physician review, demographic fairness adjudication, real patient data, real secrets or patient-level safety testing was conducted. All vignettes are from a public benchmark. Clinical decisions must never be based on these models' benchmark outputs.

\section{Conclusion}
The complete fixed model-question-condition matrix reveals severe susceptibility to misleading context, conspicuous verbal overconfidence, limited but nonzero exact canary leakage in one model, and strong sensitivity of prompt-injection conclusions to how attack success is defined. A generic trust-boundary mitigation lowered Llama's strict attack compliance but left a clear Qwen wrong-answer vulnerability. Better evaluation should combine item-paired accuracy, independently assessed evidence validity, conservative uncertainty, and explicit partial attack endpoints.

\section*{End-of-report questions}
\paragraph{Q1. Least trustworthy result?} The lexical four-gram ``hallucination'' proxy: it cannot distinguish factually false claims from harmless quotations or detect semantic paraphrases. A blind, clinician-led entailment review with inter-rater agreement would be needed to assess its validity.
\paragraph{Q2. Hardest dimension?} Interpretability. A fluent evidence sentence need not reflect the model's internal decision mechanism, and deleting text can move a query away from its natural clinical distribution. Attributions need multiple independent counterfactual controls, not just visualizations.
\paragraph{Q3. Most consequential metric choice?} Strict, all-components attack success versus wrong-key selection. The former is zero for Qwen even when the latter is high. We report both, do not change thresholds after observing the results, and disclose confidence ceilings.
\paragraph{Q4. First addition before clinical deployment?} Clinician-reviewed claim grounding with uncertainty-aware abstention, escalation testing in safety-critical situations, and longitudinal and setting-stratified fairness and privacy reviews, with prospective external validation.

\section*{Reproducibility and AI-use disclosure}
Files in the ZIP include full raw JSONL, internal diagnostic outputs, analysis scripts, diagrams, exact commits, generation prompts, environment metadata, and notebook source. The candidate must disclose relevant earlier AI sessions in \texttt{AI\_USAGE.md}, verify clinical case interpretations, and supply a working Git repository URL before submission. AI tools assisted with scripting, documentation and review; no LLM predictions were fabricated in this manuscript. Exact reported numbers are derived from the attached saved Kaggle records.

\begin{thebibliography}{9}
\bibitem[Brier(1950)]{brier} Glenn W. Brier. 1950. Verification of forecasts expressed in terms of probability. \emph{Monthly Weather Review}.
\bibitem[Guo et~al.(2017)]{guo} Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. \emph{Proceedings of ICML}.
\bibitem[Wilson(1927)]{wilson} Edwin B. Wilson. 1927. Probable inference, the law of succession, and statistical inference. \emph{Journal of the American Statistical Association}.
\bibitem[McNemar(1947)]{mcnemar} Quinn McNemar. 1947. Note on the sampling error of the difference between correlated proportions or percentages. \emph{Psychometrika}.
\bibitem[MedMisBench dataset(2026)]{medmisbench} Johndfm. 2026. MedMisBench-MentalHealth dataset. \url{https://huggingface.co/datasets/Johndfm/MedMisBench-MentalHealth}.
\bibitem[Meta AI(2024)]{llamamodel} Meta AI. 2024. Llama-3.2-3B-Instruct model card. \url{https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct}.
\bibitem[Qwen Team(2024)]{qwenmodel} Qwen Team. 2024. Qwen2.5-3B-Instruct model card. \url{https://huggingface.co/Qwen/Qwen2.5-3B-Instruct}.
\end{thebibliography}
\appendix
\section{Supplemental experiment inventory}
\begin{itemize}\setlength\itemsep{1pt}
\item Dataset: revision \texttt{9757095}, 449 total items, 200 Core, 60 Stress, 40 Interpretability.
\item Fixed models: exact commits in \texttt{metadata/run\_config.json}; do not substitute.
\item Number of actual generations: $2\times(200+12\times60)=1840$.
\item Actual interpretation rows: $2\times40=80$, each with layer probe and two controlled deletions.
\item Conditions: clean, misleading, six one-attribute fairness prompts, targeted injection, canary attack, and targeted/clean/misleading hardening variants.
\item Inference: float16, greedy decoding, deterministic preprocess seed 42; record checksum and all raw strings saved.
\item Exact system prompt, one-line metadata, misdirection, attack and synthetic canary prompt text are in the source notebook and saved configuration.
\item Core output integrity: no missing or duplicated model-condition-index records; 0 generation errors.
\end{itemize}
\paragraph{Supplemental output files.} \texttt{extended/tables/paired\_model\_comparison.csv}; \texttt{attack\_endpoint\_uncertainty.csv}; \texttt{paired\_transition\_counts.csv}; \texttt{risk\_coverage\_fixed\_thresholds.csv}; \texttt{confidence\_reliability\_bins.csv}; \texttt{fairness\_extended.csv}; \texttt{interpretability\_pairwise.csv}; \texttt{newly\_wrong\_case\_bank.csv}; and \texttt{additional\_cross\_dimension.csv}. All reproduced by \texttt{code/extended\_analysis.py} from unmodified JSONL records.
\paragraph{Disclosure.} AI assistance and session links: see \texttt{AI\_USAGE.md}; chat-share URL: \url{https://chatgpt.com/share/6acbc00d-c6c8-83ee-83e9-c7d849a240ed}. The supplied AI-review session URL is listed here; initial development also involved earlier conversations as detailed in \texttt{AI\_USAGE.md}. No alternative model or fresh dataset was introduced.
\end{document}
'''
 vals={'P_BASE0':pp(br0.accuracy),'P_BASE1':pp(br1.accuracy), 'P_STRESS0':pp(mr0.clean_accuracy),'P_MIS0':pp(mr0.misleading_accuracy),
 'P_STRESS1':pp(mr1.clean_accuracy),'P_MIS1':pp(mr1.misleading_accuracy),'P_KEY_Q':pp(sr1.target_key_selected),
 'P_MCNP':f'{cmp.p_value:.4f}','P_DISAGREE':pp(cmp.answer_disagreement),'P_SEL_COV':pp(selection.coverage),'P_SEL_ACC':pp(selection.selected_accuracy),
 'TABLE_BASELINE':baseline,'TABLE_MISLEAD':misrows,'TABLE_SECURITY':secrows,'TABLE_INTERPRET':introws,'TABLE_FAIRNESS':fairness,
 'TABLE_HARDEN':hardened,'INTERPRET_CASES':cases_tex,'ERROR_CASES':error_tex}
 for key,val in sorted(vals.items(),key=lambda kv:len(kv[0]),reverse=True):body=body.replace(key,str(val))
 assert not any(key in body for key in vals)
 acl=(r'''\documentclass[11pt]{article}
\usepackage[preprint]{acl}
\usepackage{times}
\usepackage{latexsym}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\renewcommand{\UrlFont}{\ttfamily\small}
''' + body)
 fallback=(r'''\documentclass[10pt,a4paper,twocolumn]{article}
\usepackage[left=1.7cm,right=1.7cm,top=1.9cm,bottom=1.9cm]{geometry}
\usepackage{newtxtext}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{url}
\usepackage{natbib}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\setlength{\columnsep}{0.65cm}
\setlength{\parindent}{1em}
\setlength{\parskip}{0.1em}
''' + body)
 # Keep the ACL style source unchanged; user must supply official upstream acl.sty and acl_natbib.bst.
 (root/'paper'/'main_acl.tex').write_text(acl)
 (root/'reports'/'LCS2_Round2_Report_FINAL.tex').write_text(fallback)
 (root/'paper'/'README_ACL.md').write_text('# Official ACL-source build\n\nThe report is authored as `main_acl.tex` with the **official unmodified ACL style** interface (`\\usepackage[preprint]{acl}`). For a strict ACL-format PDF, obtain `acl.sty` and `acl_natbib.bst` from https://github.com/acl-org/acl-style-files, place them beside this file, copy `../extended/figures` as indicated by relative paths, and compile with pdflatex. The supplied PDF was compiled from the identical paper body with a local two-column fallback because the official style files are not bundled. This assignment says ACL format is preferred, not mandatory. Check candidate-name and AI-chat disclosure placeholders before submission.\n')
 if shutil.which('pdflatex'):
    cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(root/'reports'),str(root/'reports'/'LCS2_Round2_Report_FINAL.tex')]
    for i in range(2):
       p=subprocess.run(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
       if p.returncode:
          print('pdfLaTeX failed, final output:');print(p.stdout[-5000:]);raise RuntimeError('PDF build failed')
    pdf=root/'reports'/'LCS2_Round2_Report_FINAL.pdf'
    assert pdf.exists() and pdf.stat().st_size>15000
    for f in (root/'reports').glob('LCS2_Round2_Report_FINAL.*'):
      if f.suffix in ('.aux','.log','.out'):f.unlink()
    print('Built:',pdf, 'size',pdf.stat().st_size)
 print('Created official ACL-compatible source and fallback PDF; candidate name and disclosure still require user input')

if __name__=='__main__':
 main(Path(sys.argv[1]) if len(sys.argv)>1 else Path('.'))
