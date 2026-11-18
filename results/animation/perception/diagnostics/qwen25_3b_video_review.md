# Qwen2.5-VL-3B Pure-Video Perception Review

## Existing Full Run

The existing run has 24 descriptions and 24 successful judge responses. Of the
descriptions, 23 parse as JSON. Four have no parsed visual entries (including the
unparseable response); nine repeat at least one visual description. These are
screening counts, not a manual annotation of all errors.

Mean visual-only judge score is 16.67 for pure video versus 45.42 for split.
Per-diagnostic means and sample sizes are in
`../charts/description_judge_by_test.csv`; component scores are retained there.
The visual-only score uses the existing four visual components and hallucination
penalty, not text transcription accuracy.

The first Homogeneity negative response describes aliens firing lasers twice,
apparently converting the introductory narration into visual events. It omits
the gray stars/triangles and blue spot. Similar narrative grounding and repeated
event descriptions warrant inspection before interpreting inference failures.

## Judge Limitation

The first Homogeneity negative judgment says that "At tonight's game, there will
be..." was omitted, although that sentence is present in the description. Thus
individual judge explanations can be wrong. The existing scores have not been
silently corrected or presented as human-verified perception accuracy. Gold
descriptions are text-derived references, not independently annotated video gold.

## Fresh Technical Pilot

Oberon job 91421 reruns the two Homogeneity1 videos, with a 2048-token budget and
frame/timing logging. Prompt version 4 corrects a stale navigation reference from
`visual_events` to the actual `timeline` schema. This is a technical replication,
not a controlled estimate of a prompt effect; it changes the budget and uses the
current environment.

The negative video supplies 66 frames of 476 x 616 at sampled fps 2.964; the
positive video supplies 68 frames at sampled fps 2.952. The first new response
still describes lasers from the introductory text. This reproduces the semantic
grounding problem despite video frames being supplied; it does not establish that
every frame/preprocessing choice in the earlier run was correct.

No new judge API calls were made. Raw pilot outputs are kept separately from the
old descriptions and are not mixed into the old score averages.

An additional input audit (job 91422) confirms that preprocessing supplies a
`pixel_values_videos` tensor of shape (49368, 1176), video grid (33, 34, 44), and
12342 video tokens for the negative video. Thus the decoded frames are not merely
read and then omitted from the model inputs. This long visual context may be a
factor for a small checkpoint, but size versus prompt versus sampling is not
causally isolated by this audit.
