# paper/

Manuscript and supporting material for **DetailHOI**.

> **Start with [`HANDOFF.md`](HANDOFF.md)** — the full project handoff: what the method
> is, how the repo is laid out, the complete experimental record with provenance for
> every number, known defects in the code, and the open work in priority order.

## Contents

| File | What it is |
|---|---|
| `HANDOFF.md` | Project handoff. The single source of truth for provenance and open work. |
| `DetailHOI.docx` | Word manuscript, ~6,400 words, figures and 5 tables embedded. |
| `detailhoi.tex` | Same paper as self-contained LaTeX. `pdflatex detailhoi.tex` twice; no BibTeX run needed. |
| `detailhoi.html` | Web version, figures inlined. Published at https://claude.ai/code/artifact/406b1935-0320-4af4-8f11-6da737136123 |
| `figures/` | Fig. 1 (flowchart), Fig. 2 (architecture), Fig. 3 (convergence), Fig. 4 (ablation), Fig. 5 (attention decomposition), as PDF + PNG, plus `curves.json`. |
| `mkfigs.py`, `mkarch.py`, `mkflow.py` | Regenerate the drawn figures: `python3 mkfigs.py && python3 mkarch.py && python3 mkflow.py` |
| `mkattn.py` | Rebuilds Fig. 5 from the two source crops in `figures/_attn_source/` (extracted from the Progress Review deck's media) — not needed unless that figure must change. |
| `Detailed human object interation.docx` | The author's original draft. Superseded, but some numbers exist only here. |

The three manuscript formats carry identical content. **If you change a number, change it
in all three.**

## Formatting note (2026-08-24)

All three formats were reformatted to match the journal-submission style of the companion
**VideoHOI** manuscript (`../VideoHOI/Manuscript_IVC_new_figures.docx`), at the user's
request. No result, number, or wording in the body was changed — only:

- **Page/typography**: `detailhoi.tex` moved from a 10pt two-column conference layout to a
  12pt single-column A4 layout with 2.5cm margins, 1.5 line spacing, and continuous line
  numbers (`lineno`); `DetailHOI.docx` got the same page setup plus a centered page-number
  footer. `detailhoi.html` is unaffected (VideoHOI has no HTML counterpart to mirror).
- **Byline**: Chao Chen and Michael Yu Wang were promoted from a "Supervisors:" line to
  co-authors with a shared affiliation superscript, matching how they appear as co-authors
  on the VideoHOI paper. A corresponding-author mark, e-mail list, ORCID line and
  "Declarations of interest: none." line were added — **this is a real authorship/byline
  change, not just typography; confirm it's what you want before submission.**
- **New back-matter sections**, copied in structure (not content) from VideoHOI: CRediT
  authorship contribution statement, Declaration of competing interest, Declaration of
  generative AI and AI-assisted technologies, Funding, Data availability, Acknowledgements.
  These contain **placeholders that must be filled in before submission**: Chao Chen's and
  Michael Yu Wang's e-mail addresses, all three ORCID iDs, the funding statement, the
  repository URL for code/checkpoints, the GenAI-tool disclosure (or its removal, if none
  was used), and any acknowledgements.
- Section headings were renumbered `N. Title` / `N.M. Title` and figure captions
  standardised to the "Fig. N." abbreviation, matching VideoHOI's convention.

Pre-reformat versions of all three files are kept in `_before_reformat_<timestamp>/` (this
repo has no version control — see `HANDOFF.md` §"Read this before touching anything").

## Content added from the Progress Review deck (2026-08-24)

At the user's request, material from `Progress Review .pptx` (the candidature-review talk)
was folded into all three manuscript formats, renumbering figures/equations throughout:

- **Fig. 1 (new), Introduction.** A simplified before/after schematic ("what changes, at a
  glance") adapted from the deck's slide-10 flow chart. It is **not** a reproduction of that
  slide — the slide's "Face Direction" input and "Human Object matcher" enhancement box
  describe a head-orientation extension that `HANDOFF.md` §10 explicitly documents as **cut
  from the evaluated system** ("do not re-introduce the claim" unless the code is found). The
  new figure shows only the pipeline actually implemented and reported; the head-orientation
  idea stays exactly where it already was, in §6's "Toward head-orientation priors" paragraph.
- **New equation, end of §3.1.** The standard four-term cross-attention decomposition
  (content–content / content–position / position–content / position–position, after
  Conditional DETR), rewritten in this paper's own notation. Shifts every later equation
  number by one (old (5)–(8) → (6)–(9) in the .docx/.tex; the .html was renumbered to match).
- **Fig. 5 (new), §5.3 "Qualitative: what the positional term attends to".** The deck's
  attention-decomposition visualisation (slide 14 / its Fig. 7), reassembled from the two
  source crops in `figures/_attn_source/`. Framed explicitly as a single-example, qualitative
  illustration, not a quantitative result.
- **§3.3 (positional budget) enrichment.** Added the `ratio1`/`ratio3` shorthand (matching the
  wandb run names in `HANDOFF.md` §6) and a sentence on the K=6 variable split, from slide 13.

**Deliberately not applied** — the deck's slide 9 gives a different human-clarity filter
(RTMPose-L; trainval 5,814→5,404, test 4,532→4,332) than the paper's YOLOv8-n filter
(trainval 4,969→4,404, test 4,532→3,971). `HANDOFF.md` §5 marks the paper's numbers
**"verified from logs"** against the actual filter script and the annotation JSONs, and every
ablation mAP in the paper was measured on that YOLOv8-n subset. The deck's numbers were not
applied — doing so would have contradicted the verified code/logs and made the results tables
inconsistent with the stated method. If the deck's numbers are in fact the current ones, this
needs the underlying re-verified before the paper is touched, not a text swap.

## Two things to know before editing

1. **Not everything in the paper is reproducible from this repo.** The subset ablation
   (Table 4, Figs. 3–4), the dataset counts and the parameter counts are verified from
   logs and code. Every HICO-DET number and the full-set V-COCO baseline come from the
   original draft with no logs behind them. `HANDOFF.md` §6 has the per-number table.

2. **V-COCO mAP here is not `AP_role`.** It comes from the in-repo diagnostic evaluator
   and runs much higher than the official protocol. It is valid only for comparing
   configurations trained and evaluated identically. `HANDOFF.md` §8.
