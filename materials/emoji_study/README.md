# Emoji inference experiments

## Current Vector Tests

Run `python .\code\build_emoji_effect_profiles.py` after refreshing the joined
ratings. Current profiles use exact per-item Wilcoxon tests with BH correction;
continuous heatmaps show mean item contrasts. See `results/README.md` for the
contrast definitions and small-item-count limitations. The original R outputs
below remain reference analyses, not the current vector significance tests.
Current cell scores, item contrasts and tests are in
`results/emoji/human_tests/{per_item_scores.csv,per_item_contrasts.csv,item_wilcoxon_tests.csv}`.

## Mean Item-Contrast Profiles

Every checkpoint is asked the same 190 original questions. The known
Qwen3-VL-2B format failure leaves 189 usable ratings for that checkpoint.
One Phone / Control / Exactly-one / Existential response never produced a
number, including unchanged-prompt retries at 128 and 1024 tokens. All attempts
are retained. `results/emoji/full_item_run.json` records the exception.
The affected co-text existential and interaction coefficients are withheld;
the 19-column heatmap shows NA, while clustering/MDS use 17 shared complete
features. The affected bar is also NA, not an average over fewer items or zero.

The added checkpoints are Qwen2.5-Omni-3B, Qwen2.5-Omni-7B,
Qwen3-Omni-30B-A3B-Instruct and MiniCPM-o-4.5. They also receive the ten
post-text-control questions, kept separate from the human comparison.
Their run metadata are in `results/emoji/added_models_run.json`.

After refreshing the joined ratings, run:

```powershell
Rscript .\code\reproduce_emoji_tests.R
python .\code\build_emoji_effect_profiles.py
```

For this known missing response, use `Rscript .\code\reproduce_emoji_tests.R
--allow-incomplete` to withhold its affected coefficients. Without that flag,
the R script rejects incomplete cells. Likewise `compare_emoji_pilot.py
--require-complete` rejects the 189/190 model; omit it to rebuild the available
ratings explicitly, without imputing the missing answer.

The R wrapper runs the original human mixed models, then fixed-only VLM models
with the same contrasts. It maps local paths, skips demographic-only reporting
(those files are unavailable), and defines the missing `post` subset in the
distributed Experiment 3 script. Original source files remain unchanged.
Install R packages `lme4`, `lmtest`, `plyr`, `ggplot2` and `sciplot` as needed.

Vectors contain the original 19 tests: 12 co-text (including reading interactions),
six pro-text (including the pooled test), and one pooled post-text test. The
adjacent tree and MDS use RMS mean item-contrast distances with average linkage for
the tree. Color represents mean item contrasts in rating points, not t statistics or
human-model errors; stars mark effect detection by ML LRT with BH correction.
Missing LRTs stay unavailable, not zero. These overlapping tests are not
independent clustering features.

The sequence builder now includes all five co-text, nine pro-text and five
post-text items, matching the original human fits. Constant VLM responses can
still make significance unavailable. New post-text premise controls remain
separate because human ratings are absent.

This experiment uses Experiments 1–3 from Tieu et al. (2023), *Experimental
evidence for a semantic typology of emoji*. Source materials, participant
responses, and analysis scripts were downloaded from the authors' OSF project:
https://osf.io/cejaq/?view_only=24b25303e6324f1883b00b6444c75b9b

The original-item set contains:

- Experiment 1: all five co-text items in every environment and reading,
  under both target and control material conditions.
- Experiment 2: all nine pro-text items in every environment, with their
  target and control inferences.
- Experiment 3: all five post-text items, with target and control inferences.

This yields 140 emoji-text materials and 190 model judgments per model.
The historical `emoji_pilot` file/folder names are retained for compatibility.
The earlier pilot had 28 materials and 38 judgments; these responses are reused
unchanged alongside 152 new judgments per model. Use `--previous-index PATH`
when preparing materials to export `remaining_index.jsonl` for unchanged prior
materials; also verify that their inference questions and saved responses match.
Experiment 4 is
not included because it is an acceptability task and requires a different
prompt and outcome analysis.

## Explicit Post-Text Premise Controls

`prepare_emoji_pilot.py` additionally writes five control materials and ten
questions to `processed/emoji_pilot/posttext_controls_index.jsonl` and
`posttext_control_statements.json`. The manifest is
`materials/emoji_study/posttext_controls_manifest.csv`.

Lyn's cigarette wording is used exactly: "If the employee steps out onto the
balcony to do this: [cigarette emoji], he will miss a phone call."
The toilet item also uses "to do this:"; the umbrella and phone items use
"because of this:", and the plane item uses "using this:". These four extensions
are proposed analogous controls, not independently approved wording from Lyn.
Original target materials and target/baseline questions are unchanged.

The new controls have **no human ratings**. Human n is zero and means are blank,
not copied from original materials or from the baseline inference question.
The same seven emoji checkpoints are run with the original pure-text rating
prompt, greedy decoding and eight-token budget. Their outputs are separate
`*_posttext_controls.jsonl` files; job IDs/commands are in
`results/emoji/posttext_control_run.json`.

Pass both original and new JSONL files to `compare_emoji_pilot.py`. It preserves
the original human-comparison tables and adds `posttext_premise_controls.csv`
and `posttext_premise_contrasts.csv`. The latter is descriptive, not a new
significance test or human-VLM discrepancy measure. `plot_emoji_pilot_barcharts.py`
also renders paginated `all_models_posttext_premise_controls_bar_charts` figures,
showing absent human control data explicitly.

Refresh locally after downloading the new ratings:

```powershell
$ratings = @(Get-ChildItem .\results\emoji\inference\ratings\*.jsonl | ForEach-Object FullName)
python .\code\compare_emoji_pilot.py @ratings --require-complete
python .\code\plot_emoji_pilot_barcharts.py
```

The original human-agreement statistics continue to use original materials only;
the new premise-control table does not create additional human observations.

Prepare the inputs and matched human means:

```powershell
python .\code\prepare_emoji_pilot.py
```

Example full-set run through the existing deterministic model adapter:

```bash
sbatch --export=ALL,WORK=/store/work/users/yzhou \
  --job-name=emoji-qwen3b \
  --partition=gpu-p1,gpu-p2 \
  --gres=gpu:1 \
  --time=00:20:00 \
  job_model.sh \
  Qwen/Qwen2.5-VL-3B-Instruct \
  --input-mode pure-text \
  --sequence-index processed/emoji_pilot/index.jsonl \
  --statements code/emoji_pilot_statements.json \
  --allow-missing-statements \
  --max-new-tokens 8 \
  --output results/emoji/inference/ratings/qwen__qwen2.5-vl-3b-instruct_emoji_full.jsonl
```

Compare the model ratings with the matched participant means:

```powershell
$ratings = @(Get-ChildItem .\results\emoji\inference\ratings\*.jsonl | Sort-Object Name | ForEach-Object FullName)
python .\code\compare_emoji_pilot.py @ratings --require-complete
```
