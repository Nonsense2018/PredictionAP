#!/usr/bin/env python3
"""Build Appendix — Code.docx from the five source files."""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).parent
OUT  = Path.home() / "Downloads" / "Appendix — Code.docx"

SECTIONS = [
    (
        "Appendix A — Air Quality Data Collection",
        "This script queries the EPA AirNow API to retrieve daily AQI and PM2.5 readings for each of the eight SJV counties using a ±0.25° bounding box centered on each county centroid.",
        ROOT / "scripts/air/fetch_airnow_history.py",
    ),
    (
        "Appendix B — Meteorological Data Collection",
        "This script queries the Open-Meteo Historical Weather API to retrieve daily maximum temperature, minimum temperature, total precipitation, and maximum wind speed for each county.",
        ROOT / "scripts/met/fetch_openmeteo_history.py",
    ),
    (
        "Appendix C — Wildfire Proximity Data Collection",
        "This script queries the NASA EONET v3 API to retrieve wildfire event locations and computes the active fire count and minimum distance to fire within a 150-kilometer radius for each county-day observation.",
        ROOT / "scripts/fire/fetch_eonet_wildfire_data.py",
    ),
    (
        "Appendix D — Feature Engineering",
        "This script merges the three data sources by county and date and constructs all predictive features including AQI lag features, three-day and seven-day rolling means, meteorological features, and wildfire proximity features.",
        ROOT / "src/features/build_features.py",
    ),
    (
        "Appendix E — Model Training and Evaluation",
        "This script trains and evaluates Logistic Regression, Random Forest, and Persistence Baseline models on the county-day dataset using a chronological train/validation/test split and saves all performance metrics.",
        ROOT / "src/models/train_models.py",
    ),
]


def add_page_number(doc):
    """Add bottom-center page numbers to the default footer."""
    section = doc.sections[0]
    footer  = section.footer
    para    = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.text = "PAGE"
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)


def main():
    doc = Document()

    # 1-inch margins
    for sec in doc.sections:
        sec.top_margin    = Inches(1)
        sec.bottom_margin = Inches(1)
        sec.left_margin   = Inches(1)
        sec.right_margin  = Inches(1)

    add_page_number(doc)

    for idx, (title, desc, code_path) in enumerate(SECTIONS):
        # Page break before every section except the first
        if idx > 0:
            doc.add_page_break()

        # Header
        h = doc.add_paragraph()
        h.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = h.add_run(title)
        run.bold      = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)

        # Description
        d = doc.add_paragraph()
        run = d.add_run(desc)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)

        doc.add_paragraph()  # small spacer

        # Code block
        code = code_path.read_text(encoding="utf-8")
        for line in code.splitlines():
            p    = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after  = Pt(0)
            run  = p.add_run(line if line else " ")
            run.font.name = "Courier New"
            run.font.size = Pt(10)

    doc.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
