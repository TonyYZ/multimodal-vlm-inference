"""Two editable report slides from the current animation tests and regressions."""
import csv
from pathlib import Path

import fitz
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from reportlab.pdfgen import canvas

from build_significant_difference_profiles import MODEL_ORDER, TEST_ORDER, short_model

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/animation"
OUT = BASE / "analysis/report_slides"
COLORS = {-1: "E6A3A3", 0: "E6E6E6", 1: "A8DBAF"}


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(16), Inches(9)
    pdf = canvas.Canvas(str(OUT / "animation_methods_and_regression.pdf"), pagesize=(1152, 648))

    def slide(title, subtitle):
        page = prs.slides.add_slide(prs.slide_layouts[6])
        text(page, title, 32, 28, 1088, 46, 30)
        text(page, subtitle, 32, 81, 1088, 32, 17)
        return page

    def text(page, value, x, y, w, h, size=16):
        box = page.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_top = frame.margin_right = frame.margin_bottom = 0
        for i, line in enumerate(value.split("\n")):
            para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            para.text, para.font.size, para.font.name = line, Pt(size), "Arial"
            pdf.setFont("Helvetica", size)
            pdf.drawString(x, 648 - y - size - i * size * 1.25, line)

    def table(page, rows, x, y, widths, row_height, colors=None, size=15):
        shape = page.shapes.add_table(len(rows), len(widths), Pt(x), Pt(y),
                                      Pt(sum(widths)), Pt(row_height * len(rows)))
        for col, width in zip(shape.table.columns, widths):
            col.width = Pt(width)
        for r, row in enumerate(rows):
            left = x
            for c, value in enumerate(row):
                color = colors[r][c] if colors else ("F0F0F0" if r == 0 else "FFFFFF")
                cell = shape.table.cell(r, c)
                cell.text = str(value)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor.from_string(color)
                cell.margin_left, cell.margin_top = Pt(5), Pt(4)
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.name, run.font.size = "Arial", Pt(size)
                        run.font.color.rgb = RGBColor(0, 0, 0)
                pdf.setFillColor("#" + color)
                pdf.setStrokeColor("#FFFFFF")
                pdf.rect(left, 648 - y - (r + 1) * row_height, widths[c], row_height, fill=1)
                pdf.setFillColor("#000000")
                pdf.setFont("Helvetica", size)
                pdf.drawString(left + 5, 648 - y - r * row_height - size - 4, str(value))
                left += widths[c]

    tests = read(BASE / "inference/effect_profiles/item_effects/item_wilcoxon_tests.csv")
    lookup = {(r["model"], r["test"]): r for r in tests if r["input_format"] in ("human", "split")}
    models = ["Human", *[m for m in MODEL_ORDER if (m, TEST_ORDER[0]) in lookup]]
    assert len(models) == 12, "Expected Human plus all 11 models."
    headings = ["Model", "SI +", "SI -", "Pres Q", "Pres Neg", "Supp", "Hom +", "Hom -"]
    rows, colors, export = [headings], [["FFFFFF"] * 8], []
    for model in models:
        values = []
        for test in TEST_ORDER:
            row = lookup[model, test]
            value = (1 if float(row["estimate"]) > 0 else -1) if float(row["q_value"]) < .05 else 0
            values.append(value)
            export.append(dict(model=model, test=test, signed_detection=value,
                               estimate=row["estimate"], p_value=row["p_value"], q_value=row["q_value"]))
        rows.append([short_model(model), *values])
        colors.append(["FFFFFF", *[COLORS[v] for v in values]])
    with (OUT / "animation_signed_detection_split.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    page = slide("From item contrasts to effect profiles", "Animation / split format / exact two-sided Wilcoxon tests with BH correction")
    table(page, rows, 32, 125, [290] + [114] * 7, 27, colors)
    text(page, "+1 green: significant positive    0 grey: not significant    -1 red: significant negative", 32, 489, 1088, 27)
    text(page, "Hamming distance = proportion of the 7 test labels that differ between two rows.", 32, 525, 1088, 27)
    text(page, "Only 2-3 items per test: all cells are 0, so Hamming distances are 0 (no resolved clustering).", 32, 559, 1088, 27)
    text(page, "The continuous heatmap instead uses RMS distances between mean item contrasts, with average linkage.", 32, 595, 1088, 26, 15)
    pdf.showPage()

    summaries = {}
    for level in ("raw", "effect"):
        prefix = "human_likeness_regression" if level == "raw" else "effect_human_likeness_regression"
        for interaction in (False, True):
            stem = prefix + ("_with_interactions" if interaction else "")
            lines = (BASE / f"inference/regression/{level}/{stem}_summary.txt").read_text().splitlines()
            summaries[level, interaction] = dict(line.split(": ", 1) for line in lines if ": " in line)
    page = slide("Updated regression: all 11 animation models", "OLS predicts human likeness from model family, log2 size, input format and experimental condition")
    rows = [["Outcome", "N rows", "Adjusted R2", "With interactions"]]
    for level, label in (("raw", "Raw-rating likeness"), ("effect", "Item-contrast likeness")):
        main, interaction = summaries[level, False], summaries[level, True]
        rows.append([label, main["N"], f"{float(main['Adjusted R-squared']):.3f}",
                     f"{float(interaction['Adjusted R-squared']):.3f}"])
    table(page, rows, 32, 140, [420, 140, 230, 298], 48, size=20)
    text(page, "Outcome = 100 - |model value - human value|, for each condition and input format.", 32, 312, 1088, 32, 18)
    text(page, "Interactions: family x inference group, family x format, and format x inference group.", 32, 357, 1088, 32, 18)
    coefficients = read(BASE / "inference/regression/effect/effect_human_likeness_regression_with_interactions_coefficients.csv")
    strongest = min((r for r in coefficients if r["term"] != "Intercept"), key=lambda r: float(r["p_value"]))
    text(page, "Strongest contrast-profile interaction: video x scalar implicature", 32, 411, 1088, 32, 21)
    text(page, f"beta = {float(strongest['estimate']):.1f}, p = {float(strongest['p_value']):.4f}, BH q = {float(strongest['q_value']):.3f}", 32, 450, 1088, 32, 20)
    text(page, "This interaction indicates lower likeness relative to the split/Homogeneity reference pattern.", 32, 493, 1088, 32, 17)
    text(page, "Exploratory: repeated conditions share models, and checkpoints share families. Conventional OLS", 32, 554, 1088, 26, 17)
    text(page, "p-values are not confirmatory evidence from independent model replications.", 32, 582, 1088, 26, 17)
    pdf.save()
    prs.save(OUT / "animation_methods_and_regression.pptx")
    doc = fitz.open(OUT / "animation_methods_and_regression.pdf")
    for i, page in enumerate(doc):
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(OUT / f"slide_{i + 1}.png")
    print(f"Wrote two slides to {OUT}")


if __name__ == "__main__":
    build()
