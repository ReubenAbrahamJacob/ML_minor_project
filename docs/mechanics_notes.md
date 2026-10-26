# Mechanics notes (from the course PDFs in docs/references/)

Source: `Ch3_1_Rolling Bearing_2021.pdf` (rolling bearings), `Ch2_1Journal Bearings_2021.pdf` (journal bearings),
`Ch1_Lubrication`, `Ch2_1_Bearings_class 2024`. Slide numbers are PDF page numbers.

| Item | Formula used | Where |
|---|---|---|
| Rating life | L10 = revolutions 90% of identical bearings reach before first fatigue evidence; L50 = 5 L10 | rolling p11 |
| Life, ball / roller | L = (C/P)^3, L = (C/P)^(10/3) (millions of rev.) | rolling p14-15 |
| Life in hours | Lh = 10^6 L10 / (60 N), N rpm | rolling p15 |
| Equivalent load | P = (X V Fr + Y Fa) Ks Kt; Ks = 1 / 1.25 / 1.5 / 2 | rolling p14 |
| Reliability | ln(1/R) = (L/a)^b, a = 6.84, b = 1.17, L in multiples of L10 | rolling p20 |
| Petroff, Sommerfeld, McKee | **journal bearings only**, background, no features built from them | journal p9-11 |

## Discrepancy found (rule 5)
Slide p20 prints `L/L90 = 8.85 [log(1/R)]^(1/1.17)`. With a = 6.84 and b = 1.17 the coefficient is
`1 / ln(1/0.9)^(1/1.17) = 6.84`, and 6.84 is the value for which R = 0.9 gives L = L10. 8.85 does not follow from
a and b. The code uses 6.84 computed from a and b (`physics.l90_coefficient`, tested in `tests/test_core.py`).

## Second discrepancy (BSF)
CLAUDE.md section 5.6 gives BSF = D/(2d) fr (1 - (d cos)/D)^2, which is the ball *spin* frequency (2.357 fr for the
CWRU 6205). CWRU publishes 4.7135 fr, exactly twice this (a ball defect strikes both races per spin). The code keeps the
CLAUDE.md formula; the test asserts 2 x BSF equals the CWRU figure.
