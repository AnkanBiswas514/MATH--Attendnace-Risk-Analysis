"""Generate PS04 (PS2) PDF report from the text parts + figures."""
from __future__ import annotations

import os
import glob
import re
from fpdf import FPDF

PROJECT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJECT, "outputs")
FIG_DIR = os.path.join(OUT, "figures")
OUT_PDF = os.path.join(OUT, "ps2_report.pdf")

TEXT_PARTS = sorted(glob.glob(os.path.join(OUT, "ps2_report_part*.txt")))
FIGS = sorted(glob.glob(os.path.join(FIG_DIR, "ps2_*.png")))


def sanitize(s: str) -> str:
    repl = {
        "\u2014": "-", "\u2013": "-", "\u00b7": "|", "\u2192": "->",
        "\u2264": "<=", "\u2265": ">=", "\u2248": "~", "\u00b2": "^2",
        "\u00d7": "x", "\u00df": "ss",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\x09\x0a\x0c\x20-\x7e]", "?", s)
    return s


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "MFAD Datathon CSE1 - RR_CSE_B - Team 9", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "PS04 - Attendance Threshold Investigation - Problem Statement 2",
                    ln=True, align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    for part in TEXT_PARTS:
        with open(part, "r") as f:
            text = f.read()
        for line in text.split("\n"):
            if len(line) > 100:
                line_iter = [line[i:i + 100] for i in range(0, len(line), 100)]
            else:
                line_iter = [line]
            for chunk in line_iter:
                chunk = sanitize(chunk)
                pdf.set_x(pdf.l_margin)
                if chunk.strip() == "":
                    pdf.ln(2)
                elif chunk.startswith("="):
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.multi_cell(0, 6, chunk.strip("=").strip())
                    pdf.set_font("Helvetica", "", 10)
                    pdf.ln(1)
                else:
                    pdf.multi_cell(0, 5.5, chunk)
        pdf.ln(4)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Figures", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for fig in FIGS:
        pdf.image(fig, x=15, w=180)
        pdf.ln(4)

    pdf.output(OUT_PDF)
    print("Wrote", OUT_PDF)
    print("Text parts:", len(TEXT_PARTS), "Figures:", len(FIGS))


if __name__ == "__main__":
    main()
