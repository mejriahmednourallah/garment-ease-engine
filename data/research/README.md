# Research data and published results

These files are manually transcribed **aggregate values**, not participant-level measurements, model predictions, training data, or results produced by this project. They cannot reproduce or validate either study. The dashboard continues to use the separately labeled synthetic `sample_garments.csv`; the real jacket chart below lacks hip measurements and is not compatible with its three-region input.

## Dik et al. (2023)

`dik_2023_table5_overall.csv` contains the overall R² entries for the four garment types in Table 5, plus the article's reported 70 training and 50 test rows. It omits psychographic subgroup results. These are author-reported scores, not independently reproduced; the small test set and study population limit their interpretation. The article reports 120 subjects and four garment categories. Its data availability statement says data will be made available on request; no participant-level dataset was obtained. The publisher version is licensed CC BY-NC-ND 4.0; check rights for any further reuse and do not treat article access as a dataset license.

Source: Dik et al., *Heliyon* 9(7), e17916, Table 5 and data availability statement, https://doi.org/10.1016/j.heliyon.2023.e17916 (full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC10362334/). Numerical values transcribed; publisher documents are linked below, not redistributed here.

## Zhang et al. (2026)

`zhang_2026_table3_jacket_a.csv` and `zhang_2026_table6_jacket_b.csv` contain the respective garment charts and **human fit model** specifications. Table 6's L fit-model bust value (85.5 cm) is transcribed as printed, despite the apparent inconsistency with the adjacent sizes and stated ideal ease; it has not been silently corrected. The human fit model is a design reference, not individual participants. The study used 270 female participants and two jacket styles. `zhang_2026_table9_results.csv` contains author-reported Top-1, Top-2 and Top-3 accuracy for the two jackets from Table 9. The prose later states 100.0% Top-3 for jacket A, conflicting with Table 9's 99.6%; the CSV deliberately records Table 9 and the abstract, not the contradictory sentence. The article's data statement refers inquiries to corresponding authors; no participant-level fitting data were obtained.

Source: Zhang et al., *Textiles* 6(3), 75, Tables 3, 6 and 9, https://doi.org/10.3390/textiles6030075. Transcribed and reformatted from the article under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); changes: column labels normalized, chart and aggregate results converted to CSV, not all article tables included.

## Reproduction and data requests

Neither CSV is a fitted model, a substitute for participant-level observations, or evidence of this application's accuracy. The two studies used different methods and different metrics; their scores are not directly comparable. Obtaining the underlying body measurements, garment specifications, labels, splits and applicable usage rights requires contacting the authors. Do not send them a request or disclose user information without separate approval. After obtaining permission, record consent, source version, field descriptions, missingness, licensing and participant-level train/test splits; evaluate the current baseline and any learned model on a held-out set. The 2026 paper does not explicitly model fabric elasticity; our stretch feature is an unvalidated heuristic.

## Source documents and supplementary materials

The 2023 paper links a [blank pre-survey questionnaire](https://ars.els-cdn.com/content/image/1-s2.0-S2405844023051241-mmc1.docx) and an [ethics clearance letter](https://ars.els-cdn.com/content/image/1-s2.0-S2405844023051241-mmc2.pdf). These are not participant responses. Its [full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10362334/fullTextXML) is also available through Europe PMC. The [2026 publisher PDF](https://mdpi-res.com/d_attachment/textiles/textiles-06-00075/article_deploy/textiles-06-00075.pdf) is available separately. Full documents are not copied into this repository; check their licenses and supplementary-material rights before reuse.
