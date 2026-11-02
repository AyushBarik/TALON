# FramesToTokens

Source of the CVPR 2027 submission **Moving from Frames to Tokens for Video Retrieval-Augmented Generation** (TALON).

Built on the official CVPR 2027 author kit (github.com/cvpr-org/author-kit); `cvpr.sty`, `ieeenat_fullname.bst` and
the kit part of `preamble.tex` are unchanged. Project packages and macros are appended at the end of `preamble.tex`.

- Main file: `main.tex`. Sections: `sec/0_abstract.tex` ... `sec/5_conclusion.tex`; proofs in `sec/X_suppl.tex`
  (supplementary pages, appended after the references: remove them from the submission PDF).
- Figure 1: `fig/overview.tex` (TikZ), images in `fig/frames/`.
- Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (Overleaf: set `main.tex` as the main document).
