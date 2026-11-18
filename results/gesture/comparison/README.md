# Preliminary Matched Human Comparison

This folder contains the **older incomplete pilot** only. Current full-study
fixed-only VLM fits are in `../human_tests/llm_fixed_effect_tests.csv`, with
current item-level plots in `../inference/effect_profiles/item_effects/` and
`../inference/bar_charts/`. The exclusions below do not describe those new runs.

`human_model_ratings.csv` compares the three completed audio-video models with
the original gesture-study human ratings, using only matching available stories.
`matched_item_audit.csv` retains every completed rating, both inference statements,
the human mean, participant count, and the inclusion decision.

Preprocessing follows the original Experiment 1 R script: remove the first two
Qualtrics metadata records, reshape the `_Test_1` response columns, join the
original design by `single_name`, and convert ratings to numbers. There are
103 human ratings per matched item. Story means are averaged equally, keeping
environment, premise and inference type separate.

The model trial is matched by family, story, environment, premise and inference
type, using Lyn's factor definitions. RTF questions sometimes omit temporal
modifiers or use slightly different wording, so these are descriptive rather
than exact-prompt replications. No significance tests are claimed here.

Homogeneity story 2 is excluded: the RTF-derived model questions mention coins,
whereas the original human design asks about crosses. Its four scores per model
remain in the audit but must not be interpreted as matching human judgments.
The comparison includes stories 1 and 3 for Homogeneity and Presupposition,
and story 1 for Scalar Implicature. Supplement is unavailable in this subset.
Qwen3-Omni has no successful scores and is excluded, not assigned zero.

No original model responses or inference statements have been changed. The
matched comparison contains 24 ratings per successful model (72 total).
