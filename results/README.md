# Experiment Results

## Current Item-Level Method

The current animation, gesture and emoji vectors use per-item Wilcoxon
signed-rank tests, not regression LRTs. Original human R analyses and regression
exports remain reference reproductions; they are not the current vector tests.
Older human-minus-model benchmark figures are in
`animation/inference/effect_profiles/legacy_benchmark/`.

For SI and Supplement, each item supplies
`d = (TargetPremise/Target - TargetPremise/Baseline) - (ControlPremise/Target - ControlPremise/Baseline)`.
For Presupposition and Homogeneity, `d = Target - Baseline`.
Emoji co-text interactions use the same factorial contrast with Existential
as the target reading and Universal as the comparison reading; simple co-text
effects use Target minus Control material. Pro-/post-text use Target minus
Baseline inference. Pooled pro-text averages environments within each emoji
before testing, rather than counting environments as independent items.

SciPy `wilcoxon` uses exhaustive sign permutations via `PermutationMethod`,
two-sided, dropping zero differences. All-zero contrasts are assigned p=1;
constant nonzero contrasts have valid exact p-values. Missing item cells remain
unavailable, not zero. Subtraction noise is rounded to 10 decimal places before
ranking ties. BH families remain seven tests per animation/gesture model-format
and 19 tests per emoji model. Binary 1 means an effect detected, NOT human-model
equivalence or a significant human-model difference. Continuous values are the
mean item contrast, not a p-value or Wilcoxon W. Human vectors use the same
item-level analysis of human cell means; Lyn's participant-level tests are retained
separately and their published significance is not being reinterpreted.

There are only 2-3 animation items, 2-4 gesture items, and 5/9 emoji items per
test. With three nonzero items, minimum two-sided p is .25; with four, .125.
All observed animation/gesture binary profiles are consequently zero at .05,
even before BH. Their zero-distance trees are explicitly marked unresolved,
not evidence that all models have the same semantic abilities. Missing gesture
Supplement2 human controls and one Qwen3-VL-2B emoji answer remain missing.

Each study's `human_tests/` now contains `per_item_scores.csv`,
`per_item_contrasts.csv` and `item_wilcoxon_tests.csv`. Yair's requested single-model
example is `animation/human_tests/yair_gemma_split_per_item_scores.csv` (48 cells).
Current animation/gesture profiles are in `inference/effect_profiles/item_effects/`.
Continuous figures/tables use `mean_item_contrasts_*`; significance profiles use
`wilcoxon_detection_*`; tests are `item_wilcoxon_tests.csv`. The plot and metrics
modules are `plot_item_effects.py` and `item_effect_metrics.py`. Genuine regression
reference outputs retain their regression names; current outputs no longer do.
Regenerate using `build_binary_effect_profiles.py --study animation`, the same
command with `--study gesture`, and `build_emoji_effect_profiles.py`.

`animation/analysis/inference_perception/input_format_comparison.csv` contains
exploratory paired-model exact Wilcoxon comparisons of split versus pure video,
with BH over five measures. Model-family dependence limits generalization.
The subsequent historical regression sections document the superseded methods.

## Per-Test Perception and MDS

`plot_description_judgments.py` now includes both saved split and pure-video
judge files by default, deduplicates model/format/video entries, and exports
`animation/perception/charts/description_judge_by_test.csv`, a diagnostic heatmap,
and a response-format/repetition audit. Component scores and material counts are
retained. This reuses existing judge outputs; no new API calls or score corrections
are implied. See `animation/perception/diagnostics/qwen25_3b_video_review.md`.

Metric MDS uses `sklearn.manifold.MDS`, two dimensions, eight starts, seed 42,
and normalized stress. CSV coordinates and stress accompany each PNG/PDF.
Animation/gesture `mean_item_contrasts_mds_*` use RMS Euclidean distances over the seven
signed mean item contrasts (rating points), including Human. No uncertainty
normalization is applied; interaction coefficients have a wider possible range.
They can include contrasts without statistically detected effects.
Mean-contrast heatmaps include an adjacent average-linkage tree using these
same RMS distances; heatmap rows follow the dendrogram leaf order. Each figure
has accompanying `_distances.csv`, `_features.csv`, and `_order.csv` files.
P-values do not determine contrast-clustering membership or feature selection.
Binary-profile
MDS uses the same Hamming distances/shared observed features as its tree.
Emoji effect-profile MDS and clustering now also use RMS mean item-contrast distances
and average linkage. They use the original emoji scripts' 19 tests, including
reading interactions, pooled pro-text and pooled post-text effects. New post-text-control MDS instead
uses RMS distances of the 20 original/control-premise ratings, without Human.
MDS axes have no intrinsic semantic meaning, and 2D distances only approximate
the original distances. Dependency versions are listed in `requirements_analysis.txt`.

There are two study folders. Raw model outputs are preserved; profile figures
and tables are replaced in place when their existing scripts are rerun.

## Layout

```text
animation/
  inference/
    ratings/          Original model JSONL responses
    comparison/       Rating tables, raw errors, and descriptive effects
    bar_charts/       Human/model endorsement bar plots
    scatter/          Human/model scatter plots and correlations
    effect_profiles/  Participant-level benchmark LRTs, profiles, and clustering
    regression/
      raw/
      effect/
  perception/
    descriptions/
    judgments/
    charts/
    diagnostics/
    pilots/
  analysis/
    cross_test/
    model_size/
    inference_perception/
  human_tests/        Reproduction of the original paper's R tests
  gesture/            Separate gesture pilot outputs
emoji/
  inference/
    ratings/
    comparison/
    bar_charts/
    effect_profiles/
  pilots/
  logs/
```

## Current VLM Performance: Fixed-Only Models

Following Yair's recommendation, the reproduction/profile entry points now default
to fitting each VLM's own ratings, not human-minus-VLM discrepancies:

- Scalar implicature/Supplement: `value ~ InferenceC * GestureC`.
- Presupposition/Homogeneity: `value ~ InferenceC`.

`value` is the VLM rating. Target is -0.5 and baseline/control is +0.5.
There are no random effects, no Story blocking factor, no repeated-seed participants,
and no averaging across conditions. Each successful Story/premise/inference cell
contributes once; the latest successful retry is used. The two-factor interaction
is the diagnostic for scalar implicature/Supplement; the Inference coefficient is
the diagnostic otherwise. Single-factor coefficients are baseline minus target;
interaction coefficients are target-minus-baseline under the target premise minus
the same contrast under the control premise. Coefficients retain rating-point units.

Run locally from the project root:

```powershell
& "D:\Program Files\R\R-4.5.1\bin\Rscript.exe" --vanilla .\code\reproduce_original_human_tests.R
& "D:\Program Files\R\R-4.5.1\bin\Rscript.exe" --vanilla .\code\reproduce_original_human_tests.R --gesture
python .\code\build_significant_difference_profiles.py --study animation
python .\code\build_significant_difference_profiles.py --study gesture
```

Lyn's original human scripts still run unchanged and retain their mixed-effects
models and likelihood-ratio tests. VLM-only Gaussian linear models now also use
maximum-likelihood LRTs via `lmtest::lrtest(reduced, full)`, not coefficient
t-tests/F-tests. The reduced model removes only the diagnostic interaction or
Inference term. Each output records both log likelihoods, chi-square, df, and
p-value. BH correction is applied
over seven diagnostics per VLM/format, and separately over the seven human
diagnostics. These procedures have different uncertainty/power: comparing their
significance decisions is **not** a direct human-VLM difference test.

Current outputs under **each** study:

- `human_tests/llm_fixed_effect_tests.csv`: coefficients, SEs, LRT chi-square/df/p, confidence
  intervals, formulas and availability flags.
- `human_tests/llm_fixed_effect_cells.csv`: the exact scored cells and human means.
- `inference/effect_profiles/item_effects/`: mean item-contrast heatmaps,
  BH-adjusted effect-presence tables, heatmaps and available clustering.
- `inference/bar_charts/`: refreshed individual and paginated human/model charts.

Binary values here mean **1 = effect detected**, **0 = no effect detected**,
not the previous difference-from-humans interpretation. Perfect/constant fits
retain their coefficients but have unavailable significance (NA), not p=0 or
binary 0. Animation text/video have no diagnostic with estimable p-values for
every model, so their binary heatmaps explicitly omit the tree. Split clustering
has only one shared diagnostic (scalar implicature/positive) and is not an
informative seven-diagnostic profile comparison. Trees use only common observed features, listed in the exported feature
CSV. Rating bar error bars are SEM across story means, not across repeated model
runs; these are descriptive and separate from regression uncertainty.

All available animation checkpoints/formats and the five completed gesture
checkpoints are included. The gesture analysis uses the corrected original-design
questions and full-folder runs, not the old incomplete pilot comparison.
The small story counts and OLS independence/homoscedasticity assumptions limit
inference; repeated deterministic runs would not add independent observations.
LRT p-values use an asymptotic chi-square reference, which can be poorly calibrated
with these small story counts. Zero-variance fits remain non-regular: changing
from t-tests to LRTs does not justify assigning them p=0. Confidence intervals
remain ordinary OLS coefficient intervals; binary decisions use only LRT p/q-values.
The additional R dependency is `lmtest` (with `zoo`); a project-local installation
under `tmp/R-library/` is supported, as is a normal R library installation.
The gesture human response CSV has no `S9_Test_1`/`S10_Test_1` columns (the two
Supplement2 control questions). VLM fits still use all 64 design cells; their
`n_cells_with_human_data` flags this coverage difference. No missing human ratings
are invented, and Supplement human/model coefficients do not use identical item
coverage. This matters when interpreting their descriptive comparison.

## Current Downstream Analyses

All three folders under `animation/analysis/` now use the current fitted effect
vectors, not the older discrepancy-model significance decisions:

`effect_rms_distance = sqrt(mean((VLM coefficient - human coefficient)^2))`.

Whole-model comparisons use seven inference-type/environment combinations;
cross-test comparisons use the one or two combinations belonging to each family.
This is the same RMS geometry used in coefficient clustering/MDS, in rating-point
units, without SD normalization. Lower distance is descriptively closer to humans;
it is not a human-VLM significance/equivalence test. Interaction and single-factor
effects have different possible ranges. An unavailable LRT does not discard an
available coefficient; missing coefficients prevent a complete summary. LRT
availability and detected-effect counts are exported separately in the cross-test
table and never ranked as accuracy.

```powershell
python .\code\analyze_cross_test_performance.py
python .\code\analyze_model_size.py
python .\code\compare_inference_perception.py
```

Existing output tables/figures are replaced in place. Raw-rating MAE/correlations
and perception score definitions are retained as secondary measures, with current
input files. Cross-test effect correlations use SciPy pairing permutations, capped
at 9999 with seed 42 (exact enumeration when feasible), and the existing BH
families; raw-rating correlations remain asymptotic. Model-size regression and
quadratic tests retain their previous methods; their outcome is now coefficient
distance. These are exploratory checkpoint-level associations, not participant
replicates or causal estimates of size.

Perception-inference ranks use the intersection of available inference and
perception checkpoints, not ranks taken from a larger unmatched pool.
The added-model perception extension covers Qwen2.5-Omni-3B/7B,
Qwen3-Omni-30B-A3B-Instruct and MiniCPM-o-4.5, targeting the same 24 original
materials in split and pure-video formats. Its metadata and progress are in
`animation/perception/added_models_run.json`. Every association reports its actual sample size. Size
uses total parameters: Qwen3-Omni 30B-A3B = 30B, MiniCPM-o 4.5 = 9B.

## Previous Human-LLM Discrepancy Profiles

The following describes the older, distinct discrepancy analysis. Its files remain
for reference and are **not** regenerated or relabelled as VLM-only performance.
Use `--benchmark` on the reproduction/profile scripts to request it explicitly.
The downstream analyses now consume the new coefficient table instead, as described
above. Older MAE regressions remain separate and are not silently converted into
VLM performance models.

Run from the project root (R is installed locally at the path below):

```powershell
& "D:\Program Files\R\R-4.5.1\bin\Rscript.exe" --vanilla .\code\reproduce_original_human_tests.R --benchmark
python .\code\build_significant_difference_profiles.py --benchmark
```

`build_binary_effect_profiles.py` is an alternative entry point, also defaulting
to VLM fixed-only performance; use `--benchmark` for the older comparison below.
The reproduction script executes Lyn's original R script directly, preserving
her human-only models and plotting code. The added human-LLM tests retain
every individual human response and define:

`discrepancy = human rating - matching LLM rating`.

The LLM rating is a fixed benchmark, not a new observation for each human.
There is no `SourceC` factor or invented LLM participant ID. Each discrepancy
model inherits the fixed effects, contrast coding, and final random-effects
structure from Lyn's corresponding human-only model, replacing only the outcome:

- Scalar implicature/Supplement: test `InferenceC:GestureC` against zero.
- Presupposition: test `InferenceC`; Homogeneity: test `InferenceTypeC`.

These tests ask whether the human diagnostic coefficient differs from the fixed
LLM benchmark for the tested materials. The models use ML (`REML=FALSE`) and
the bobyqa optimizer with a higher evaluation limit; their formulas and fit
warnings are recorded rather than silently simplifying random effects.

Each reduced model removes only the diagnostic term; `anova(full, reduced)`
provides the likelihood-ratio chi-square and p-value. The source results are
`animation/human_tests/llm_fixed_benchmark_tests.csv`.

**Scope:** human ratings are not averaged. Human participant random effects
remain, and Story random effects remain wherever present in Lyn's final formula.
Uncertainty is over human responses conditional on the fixed model run and tested
materials, not over stochastic LLM runs or an unrestricted population of new items.
There are only 2-3 stories per diagnostic, and singular fits remain common.
Inspect the recorded convergence/singularity flags before interpreting LRTs.

Score parsing accepts a leading 0-100 rating, optionally followed by an
explanation. Gemma's known standalone `thought` line before the score is removed
for parsing only; raw responses are not edited. Arbitrary prose containing a
number is not treated as a score, and invalid/out-of-range outputs remain NA.

Animation binary profiles use 1 = detected difference, 0 = no detected difference,
after BH correction within each model/input format (seven diagnostics).
Failed/unscored tests remain NA, including in the plots; they are never coded 0.
Missing tests count conservatively as p = 1 for BH, then their q-values stay NA.
No detected difference is not evidence of equivalence.

The main vectors are **binary**, not coefficients, MAE, or signed LRT roots.
Each `binary_condition_profiles_{pure_text,pure_video,split}.csv` has seven model
rows and seven diagnostic columns. A value is 1 exactly when its BH-adjusted
q-value is below .05. `model_human_significance_tests.csv` contains the underlying
p/q-values, threshold, fit flags, and binary decisions.

Clustering uses **Hamming distance and average linkage** on binary vectors.
The similarity matrix is 1 minus Hamming distance, ranging from 0 to 1. Shared
zeros count as matching decisions but do not establish equivalence with humans.
The per-format `binary_*_clustering` files and concatenated
`significant_difference_clustering` files include distance/similarity matrices,
leaf orders, and the features actually used. Only features observed for every
model are used; missing values are never imputed as zero.

The all-format aggregate is the **proportion of three binary decisions**, not a
new binary test. It stays missing unless all three decisions are available.
Its clustering uses average linkage and mean absolute proportion distance;
its similarity matrix is 1 minus that distance. The concatenated all-format
matrix is exported separately as `binary_all_modes_profiles.csv`.

The earlier downstream analyses summarized fractions of detected human-LLM
differences. Those downstream files have been replaced by the current coefficient
distance analyses; the older discrepancy test/profile files themselves remain
separate. Constant binary profiles still have undefined correlations, not zero.

For emojis, run `reproduce_emoji_tests.R` followed by
`build_emoji_effect_profiles.py`. The R wrapper executes the original human
mixed-effects models and fits their fixed-only VLM counterparts with the same
contrast coding. Full/reduced ML LRTs test each effect; BH correction covers
19 tests per model. Unavailable p-values remain missing (counted conservatively
as p=1 only when calculating the family adjustment). Coefficients stay available
for clustering even when an LRT cannot be estimated. Binary values indicate
effect detection, not significant human-model differences or equivalence.

Original emoji regression reference: `emoji/human_tests/emoji_fixed_effect_tests.csv`.
Current emoji outputs:
`emoji/inference/effect_profiles/{item_wilcoxon_tests.csv,emoji_mean_item_contrast_vectors.csv,
emoji_wilcoxon_detection_vectors.csv,emoji_mean_item_contrasts_clustering.png}` with PDF,
distance, order, feature and MDS companions. Superseded t-vector/heatmap files
have been removed, without changing raw ratings or descriptive bar charts.

The expanded emoji run matches human item coverage: five co-text, nine pro-text,
and five post-text items, with 190 judgments per model. It reuses the original
38 pilot judgments and adds 152 judgments for each of seven models. Historical
`emoji_pilot` paths remain for compatibility; job metadata are in
`emoji/full_item_run.json`. Run `compare_emoji_pilot.py --require-complete` with
the original, remaining-item and post-text-control rating files before refitting.
The current run has one unresolved Qwen3-VL-2B format failure: 189/190 usable
ratings. Added checkpoints also receive all 190 original questions and ten
separate post-text controls; see `emoji/added_models_run.json`. To reproduce these outputs,
omit `--require-complete` and run `reproduce_emoji_tests.R --allow-incomplete`.
The two affected coefficients are withheld, not estimated from incomplete cells;
the heatmap keeps 19 tests and marks these NA. Clustering/MDS use 17 features
complete for every model. The affected bar is NA rather than a four-item average.
Constant VLM responses can still yield unavailable LRTs. Do not interpret
missing significance as non-significance. Original pooled tests and follow-ups
overlap, so the 19 clustering features are not independent.
The explicit new post-text premise controls lack human data and remain separate.

Descriptive MAE tables and older MAE regression outputs remain unchanged;
they are not fixed-benchmark tests and were not regenerated as if they were.

The obsolete `continuous_*` figures/data and standalone
`diagnostic_coefficient_vectors_*` files have been removed from the main
profile folder. Diagnostic coefficients remain in the raw benchmark fit table
for auditing, but are not the values used in the binary similarity matrices.

## Random-Effects Sensitivity

This is a separate analysis; it does not change the main formulas, decisions,
or clustering. Run:

```powershell
& "D:\Program Files\R\R-4.5.1\bin\Rscript.exe" --vanilla .\code\reproduce_original_human_tests.R --sensitivity
python .\code\build_significant_difference_profiles.py --sensitivity .\results\animation\human_tests\sensitivity
```

Four predefined candidates are applied to every model/format within a diagnostic:

- `lyn_final`: Lyn's final random-effects structure.
- `lyn_with_story`: the same structure with a Story intercept added where absent.
- `uncorrelated_slopes`: diagonal participant random effects for all design-based
  slopes, plus a Story intercept. SI/Supplement includes the interaction slope;
  other diagnostics include the Inference slope. This can restore slopes omitted
  in Lyn's final simplified models; it is not only a change to correlations.
- `intercepts_only`: participant and Story intercepts, as a deliberately simple
  reference. Dropping within-participant slopes can underestimate uncertainty.

Fixed effects, contrast coding, observations, ML and optimizer settings are held
constant. Each diagnostic LRT retains the same random structure in its full and
reduced model. Identical candidate formulas reuse a fit rather than refitting it.
No LRT is used to select random-effect variances, whose nulls lie on a boundary.

Outputs are under `animation/human_tests/sensitivity/`: detailed fits, BH decisions,
a diagnostic-level summary, distance stability, and two overview figures. AIC
differences compare full models only within the same model/format/diagnostic;
they are not comparisons of different data sets. AIC still includes singular
fits, so it must be read alongside convergence and singularity flags. Residual
RMSE, outlier fractions and residual/fitted associations are descriptive screens.

BH correction remains seven tests per variant/model/format. Unconverged fits are
unavailable for sensitivity decisions; singularity is reported rather than hidden.
Clustering stability uses shared observed features across all variants. It compares
pairwise Hamming and cophenetic distances, not arbitrary leaf reordering under ties.
The summaries do not automatically choose a formula or promote a candidate based
on its p-values.

## Gesture audio-video inference

`code/test_gesture_omni.py` also runs the gesture study (TSC Experiment 1), using
only `materials/gestures/OSF Stimuli Anonymized/*.mp4`. Target and baseline
statements come from `materials/stories/TSC_EXP_1_Stories.rtf`; the RTF's
"Control inference" is the baseline question, not a ControlPremise video.
No written story or gesture gloss is included in the model prompt.

The complete anonymized folder has 36 videos: 6 Homogeneity, 6 Presupposition,
16 Scalar Implicature, 4 Supplement and 4 Iconicity. The inference runner skips
Iconicity because it requires a different task: 32 videos / 64 ratings per model.
The initial pilot used only 14 inference videos / 28 ratings.

Supported audio-video models: Qwen2.5-Omni 3B / 7B, Qwen3-Omni 30B-A3B-Instruct,
and Gemma 4 12B. These Omni models are different checkpoints from Qwen-VL.
The runner verifies audio tracks and uses greedy generation, 3 fps, and a
32-token answer budget. Qwen decoding is capped at 307200 pixels per frame;
actual sampled fps is passed to its processor with resampling disabled.
Gemma uses its native visual resolution and a separate temporary 16 kHz mono
soundtrack from the same video. No original media is changed.

Outputs are saved and flushed per question under `results/gesture/inference/`.
`--resume` skips successful scores; failures remain visible and produce a nonzero
job exit. Logs are under `logs/gesture-*.log`. This is an incomplete gesture-study
subset; do not compare it to animation human data as though it were Experiment 2.

Example on Oberon, from the project directory (create `logs` before submission):

```bash
sbatch --job-name=gesture-omni7b job_omni.sh --model-id Qwen/Qwen2.5-Omni-7B --resume
```

### Gesture Perception Check

Use the same script and Oberon wrapper with `--task describe`. It processes each
of the 14 inference videos once, using their original audio and video. This mode
does not read the story RTF and does not supply an inference question. It asks for
literal speech transcription and physical gesture descriptions in one ordered
timeline, allowing speech and gestures to overlap. Left/right is viewer-relative;
timestamps can be null rather than guessed. Silent gestures must not be completed
as invented spoken words.

Results: `results/gesture/perception/descriptions/*_descriptions.jsonl`.
The answer budget defaults to 2048 tokens; malformed or incomplete timeline JSON
is recorded as an error and is retried with `--resume`. Valid JSON does not imply
accurate perception. A later judge must use an independently checked speech and
gesture reference, not assume that the unverified RTF is an exact transcript.
The original animation perception and inference outputs are unchanged.

```bash
sbatch --job-name=gesture-desc7b job_omni.sh --model-id Qwen/Qwen2.5-Omni-7B --task describe --resume
```

For a technical check before judging, use `--task sanity --limit 1`. This uses a
short plain-text speech/gesture prompt, without the structured description schema
or timestamps. Original description outputs are preserved. Outputs are under
`results/gesture/perception/sanity/`. It records input tensor shapes, nonempty
finite audio/visual features, decoded waveform duration and RMS, Qwen sampled
frame shapes/fps, generated-token count, prompt-prefix agreement before trimming,
and whether the answer reached its token budget. No judge API is called.
These technical checks do not certify transcription, gesture or alignment accuracy.

Pipeline version `gesture-av-2` fixes Gemma's default 32-frame setting taking
precedence over requested fps. Earlier Gemma runs sampled 32 frames uniformly
over each complete clip, rather than honoring fps. Old results are not modified;
resuming them with the corrected Gemma sampler is rejected to avoid mixing settings.
Use a new output file for a rerun. Qwen sampling is unchanged.

### Complete Folder Rerun With Mask Note

`code/commands.txt` contains the eight full rerun commands: inference and
perception for each of the four audio-video models. Inference jobs use `--design
materials/gestures/human_data/TSC_EXP_1_Design.csv` to take the exact original
human questions, including crosses for Homogeneity2, instead of relying on the
RTF. Only the question is supplied, never the story narration or gesture gloss.

Both tasks include this neutral system-prompt note:

> The video shows a person speaking and making hand gestures. The blur over the
> person's face is an anonymization mask. Do not infer hidden facial details.

New outputs, kept separate from the old pilot:

- `results/gesture/inference/full_mask_note/`: 64 ratings per model.
- `results/gesture/perception/descriptions/full_mask_note/`: 32 plain-text
  speech/gesture descriptions per model, using `--task sanity` without a limit.

Perception uses the simpler pilot prompt (no JSON/timestamps), retains full raw
responses and input/token audits, and does not call a judge. Prompt versions and
video SHA-256 hashes are saved; resuming with changed prompts or media is rejected.
Compared with the pilot, the media set, mask note, exact inference question wording
and corrected Gemma sampling have changed. These are not mask-only comparisons.

Gemma's upstream float-step frame sampler can occasionally emit an index equal
to the frame count. The gesture adapter filters out-of-range indices only; valid
sampling cases are unchanged. Recovery job IDs are stored in the run manifest.
Raw failed attempts are retained when successful retries are appended. Summaries
must select the latest successful response per video/question rather than count
every JSONL line as a separate observation. Repetition and output-budget hits in
plain perception responses are retained as model outcomes, not resampled with
`--resume`; only missing/error responses are retried.

## Matched Audio-Video Checkpoints Across Studies

The animation runner also supports Qwen2.5-Omni (3B/7B), Qwen3-Omni
30B-A3B-Instruct, and MiniCPM-o 4.5. Animation runs keep the existing questions,
greedy decoding and three formats; they supply no audio. Split Omni inputs retain
each clip's sampled timing. Outputs go to `animation/inference/ratings/`.

MiniCPM-o uses the separate `$WORK/venvs/minicpmo45` environment described by
`requirements_minicpmo.txt`; the main environment is unchanged. Its model revision
is pinned in `code/omni_runners.py`. Native preprocessing differs from Qwen's:
gesture audio-video inputs use aligned one-second segments with stacked
within-second frames, while animations use sampled video frames. Image area is
capped without cropping; native visual token compression remains enabled.
The input context limit is 32768 tokens, not the native chat helper's shorter default.

Gesture MiniCPM outputs use the existing `full_mask_note/` inference and perception
directories. Perception remains an unjudged technical sanity check, not a measured
accuracy score. Shared checkpoints permit cross-study comparisons, but the studies
still differ in their materials and available modalities. Run commands are in
`code/commands.txt`; no statistical analyses are automatically replaced by these runs.
