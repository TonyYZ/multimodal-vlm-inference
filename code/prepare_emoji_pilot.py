from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "materials" / "emoji_study" / "source_cache"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "emoji_pilot"
MANIFEST = PROJECT_ROOT / "materials" / "emoji_study" / "emoji_pilot_manifest.csv"
STATEMENTS = PROJECT_ROOT / "code" / "emoji_pilot_statements.json"

RATING_INTRO = (
    "Based only on the emoji-text material above, to what degree does it suggest "
    "the following statement?"
)

EMOJI = {
    "Cigarette": "🚬",
    "Umbrella": "☂️",
    "Phone": "📱",
    "Plane": "✈️",
    "Toilet": "🚽",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all original inference items from Tieu et al. Experiments 1-3."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--statements", type=Path, default=STATEMENTS)
    parser.add_argument("--previous-index", type=Path,
                        help="Also write remaining_index.jsonl, excluding unchanged materials in this earlier index.")
    return parser.parse_args()


def replace_emoji_name(text: str, name: str) -> str:
    return str(text).replace(name.upper(), EMOJI[name])


def load_human_responses(path: Path, response_type: str, response_value: str) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    data = data[data[response_type] == response_value].copy()
    data["Response"] = pd.to_numeric(data["Response"], errors="coerce")
    return data[data["Response"].notna()]


def sequence(source: str, sentence: str) -> dict:
    content = [{"type": "text", "text": sentence}]
    return {
        "source_video": source,
        "rating_prompt_intro": RATING_INTRO,
        "qwen_content": content,
        "qwen_content_pure_text": content,
    }


def add_manifest_row(
    rows: list[dict],
    *,
    source: str,
    experiment: str,
    emoji_type: str,
    emoji: str,
    environment: str,
    material_condition: str,
    inference_type: str,
    statement_type: str,
    sentence: str,
    inference: str,
    responses: pd.Series,
) -> None:
    rows.append(
        {
            "source_video": source,
            "experiment": experiment,
            "emoji_type": emoji_type,
            "emoji": emoji,
            "environment": environment,
            "material_condition": material_condition,
            "inference_type": inference_type,
            "statement_type": statement_type,
            "sentence": sentence,
            "inference": inference,
            "human_mean": responses.mean(),
            "human_n": responses.count(),
        }
    )


def build_exp1(source_dir: Path, sequences: list[dict], statements: dict, rows: list[dict]) -> None:
    design_path = source_dir / "exp1_design.xlsx"
    human_by_condition = {
        "Target": load_human_responses(
            source_dir / "exp1_target.csv", "Zone Type", "response_slider_endValue"
        ),
        "Control": load_human_responses(
            source_dir / "exp1_control.csv", "Zone Type", "response_slider_endValue"
        ),
    }

    for condition, sheet in (("Target", "target"), ("Control", "control")):
        design = pd.read_excel(design_path, sheet_name=sheet)
        design["Environment"] = design["Environment"].fillna("None")
        human = human_by_condition[condition]

        for _, item in design.iterrows():
            emoji_name = str(item["Emoji"]).strip()
            environment = str(item["Environment"])
            inference_type = str(item["Inference.Type"])
            source = (
                "materials\\emoji_study\\"
                f"Exp1_CoText_{emoji_name}_{condition}_{environment}_{inference_type}.txt"
            )
            sentence = replace_emoji_name(item["Sentence"], emoji_name)
            inference = str(item["Inference"])
            sequences.append(sequence(source, sentence))
            statements[source] = {"target": inference, "baseline": ""}

            matches = human[
                (human["Emoji"].str.strip() == ("Smoke" if emoji_name == "Cigarette" else emoji_name))
                & (human["Environment"].fillna("None") == environment)
                & (human["Inference.Type"] == inference_type)
            ]["Response"]
            add_manifest_row(
                rows,
                source=source,
                experiment="Experiment 1",
                emoji_type="Co-text",
                emoji=emoji_name,
                environment=environment,
                material_condition=condition,
                inference_type=inference_type,
                statement_type="target",
                sentence=sentence,
                inference=inference,
                responses=matches,
            )


def build_paired_experiment(
    *,
    design: pd.DataFrame,
    human: pd.DataFrame,
    experiment: str,
    emoji_type: str,
    source_prefix: str,
    sequences: list[dict],
    statements: dict,
    rows: list[dict],
) -> None:
    design = design.copy()
    design["Environment"] = design.get("Environment", pd.Series(index=design.index)).fillna("None")

    group_columns = ["Emoji", "Environment", "Sentence"]
    for (emoji_name, environment, raw_sentence), group in design.groupby(group_columns, sort=False):
        emoji_name = str(emoji_name).strip()
        source = (
            "materials\\emoji_study\\"
            f"{source_prefix}_{emoji_name}_{environment}.txt"
        )
        sentence = str(raw_sentence)
        if emoji_name in EMOJI:
            sentence = replace_emoji_name(sentence, emoji_name)
        sequences.append(sequence(source, sentence))
        statements[source] = {"target": "", "baseline": ""}

        for _, item in group.iterrows():
            condition = str(item["Condition"])
            statement_type = "target" if condition == "Target" else "baseline"
            inference = str(item["Inference"])
            statements[source][statement_type] = inference

            human_emoji = emoji_name
            matches = human[
                (human["human_emoji"] == human_emoji)
                & (human["human_environment"] == str(environment))
                & (human["human_condition"] == condition)
            ]["Response"]
            add_manifest_row(
                rows,
                source=source,
                experiment=experiment,
                emoji_type=emoji_type,
                emoji=str(emoji_name),
                environment=str(environment),
                material_condition="Same material",
                inference_type="Target" if condition == "Target" else "Control",
                statement_type=statement_type,
                sentence=sentence,
                inference=inference,
                responses=matches,
            )


def build_exp2(source_dir: Path, sequences: list[dict], statements: dict, rows: list[dict]) -> None:
    design = pd.read_excel(source_dir / "exp2_design.xlsx")
    human = load_human_responses(
        source_dir / "exp2_protext.csv", "Zone Type", "response_slider_endValue"
    )
    human["human_emoji"] = human["Emoji"]
    human["human_environment"] = human["Environment"].fillna("None")
    human["human_condition"] = human["Condition"]
    build_paired_experiment(
        design=design,
        human=human,
        experiment="Experiment 2",
        emoji_type="Pro-text",
        source_prefix="Exp2_ProText",
        sequences=sequences,
        statements=statements,
        rows=rows,
    )


def build_exp3(source_dir: Path, sequences: list[dict], statements: dict, rows: list[dict]) -> None:
    design = pd.read_excel(source_dir / "exp3_design.xlsx")
    design = design[design["Emoji.Type"] == "Post-text"].copy()
    design["Environment"] = "Unembedded"
    human = load_human_responses(source_dir / "exp3_task.csv", "Response Type", "response")
    human["human_emoji"] = human["Spreadsheet: Emoji"]
    human["human_environment"] = "Unembedded"
    human["human_condition"] = human["Spreadsheet: Condition"]
    build_paired_experiment(
        design=design,
        human=human,
        experiment="Experiment 3",
        emoji_type="Post-text",
        source_prefix="Exp3_PostText",
        sequences=sequences,
        statements=statements,
        rows=rows,
    )


def posttext_controls(sequences: list[dict], statements: dict, rows: list[dict]):
    # Purpose, cause and means controls make the emoji's attachment explicit
    # without replacing its meaning with a verbal gloss.
    connectors = {"Cigarette": " to do this:", "Toilet": " to do this:",
                  "Umbrella": " because of this:", "Phone": " because of this:",
                  "Plane": " using this:"}
    controls, questions, manifest = [], {}, []
    for item in sequences:
        source = item["source_video"]
        matches = [r for r in rows if r["source_video"] == source and r["experiment"] == "Experiment 3"]
        if not matches:
            continue
        name = matches[0]["emoji"]
        sentence = item["qwen_content"][0]["text"]
        marker = " -- " + EMOJI[name]
        if sentence.count(marker) != 1:
            raise ValueError(f"Expected one post-text emoji attachment: {source}")
        control = sentence.replace(marker, connectors[name] + " " + EMOJI[name], 1)
        control_source = source.removesuffix(".txt") + "_ControlPremise.txt"
        controls.append(sequence(control_source, control))
        questions[control_source] = dict(statements[source])
        manifest.extend({**r, "source_video": control_source, "target_source": source,
                         "sentence": control, "material_condition": "ControlPremise",
                         "human_mean": "", "human_n": 0, "human_available": False,
                         "control_version": "explicit-attachment-1"} for r in matches)
    return controls, questions, manifest


def main() -> None:
    args = parse_args()
    sequences: list[dict] = []
    statements: dict[str, dict[str, str]] = {}
    manifest_rows: list[dict] = []

    build_exp1(args.source_dir, sequences, statements, manifest_rows)
    build_exp2(args.source_dir, sequences, statements, manifest_rows)
    build_exp3(args.source_dir, sequences, statements, manifest_rows)

    frame = pd.DataFrame(manifest_rows)
    if frame["human_mean"].isna().any() or frame.duplicated(["source_video", "statement_type"]).any():
        raise ValueError("Every original question must have one unique human-data match.")
    if len(sequences) != len(statements):
        raise ValueError("Duplicate material identifiers.")
    previous = {}
    if args.previous_index:
        previous = {item["source_video"]: item for item in
                    (json.loads(line) for line in args.previous_index.read_text(encoding="utf-8").splitlines() if line.strip())}
        current = {item["source_video"]: item for item in sequences}
        if any(current.get(source) != item for source, item in previous.items()):
            raise ValueError("Earlier materials changed or are missing; do not silently reuse their ratings.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sequences),
        encoding="utf-8",
    )
    args.statements.write_text(
        json.dumps(statements, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.manifest, index=False)
    if args.previous_index:
        remaining = [item for item in sequences if item["source_video"] not in previous]
        (args.output_dir / "remaining_index.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining), encoding="utf-8")
        print(f"Wrote {len(remaining)} additional materials to remaining_index.jsonl")

    controls, questions, control_rows = posttext_controls(sequences, statements, manifest_rows)
    (args.output_dir / "posttext_controls_index.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in controls), encoding="utf-8")
    (args.output_dir / "posttext_control_statements.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(control_rows).to_csv(args.manifest.with_name("posttext_controls_manifest.csv"), index=False)

    print(f"Wrote {len(sequences)} materials to {index_path}")
    print(f"Wrote {sum(bool(v) for item in statements.values() for v in item.values())} judgments to {args.statements}")
    print(f"Wrote human comparison data to {args.manifest}")
    print(f"Wrote {len(controls)} new control premises / {len(control_rows)} questions; no human data assigned.")


if __name__ == "__main__":
    main()
