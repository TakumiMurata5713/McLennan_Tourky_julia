# Julia vs Python (QuantEcon.py) comparison of `mclennan_tourky`

- Reference file: `python_reference.tsv`
- Number of games: 1000
- Settings: `epsilon = 0.001`, `max_iter = 200`, random pure initial actions
- Match criterion: identical best response path (the pure best response profile in every iteration), identical `converged` and `num_iter`, and `max |x_star_jl - x_star_py| <= 0.001`
- Julia 1.12.6, 7 thread(s)

## Result

**892 / 1000 games match.**

- Best response path mismatches: 82
- `converged` mismatches: 0
- `num_iter` mismatches: 0
- Converged games: 115 (all of them pass `is_nash(g, NE; tol=epsilon)`: true)

### Converged vs. non-converged games

| | games | match | BR path match | max abs diff of x_star |
|---|---|---|---|---|
| converged | 115 | 115 | 115 | 1.180e-05 |
| not converged (stopped at max_iter) | 885 | 777 | 803 | 1.000e+00 |

### Where the best response paths diverge

For the games whose best response paths differ, the first iteration at which the pure best response profiles differ:

| iteration | games |
|---|---|
| 1-50 | 0 |
| 51-100 | 0 |
| 101-150 | 3 |
| 151-200 | 79 |

Earliest: 139, median: 188. The paths agree exactly in the early iterations and diverge only after the ~1e-16 rounding differences have accumulated enough to flip a nearly tied best response.

### Closeness of `x_star`

Rounding differences of order 1e-16 between Numba and Julia accumulate over the iterations, so `x_star` is not bit-for-bit identical. Distribution of `max |x_star_jl - x_star_py|`:

| max abs diff | games | of which converged |
|---|---|---|
| = 0 (bit-for-bit) | 63 | 41 |
| <= 1e-14 | 6 | 6 |
| (1e-14, 1e-12] | 28 | 25 |
| (1e-12, 1e-10] | 30 | 12 |
| (1e-10, 1e-8] | 138 | 17 |
| (1e-8, 1e-6] | 300 | 10 |
| (1e-6, 1e-4] | 271 | 4 |
| (1e-4, 1e-3] | 58 | 0 |
| > 1e-3 | 106 | 0 |

Overall max: 1.000e+00; median: 4.857e-07

## By number of players

| N | games | match | BR path match | converged | max abs diff | mean time Python (s) | mean time Julia (s) |
|---|---|---|---|---|---|---|---|
| 3 | 64 | 61 | 61 | 54 | 7.03e-01 | 0.301 | 0.391 |
| 4 | 64 | 62 | 62 | 32 | 1.41e-01 | 0.815 | 1.185 |
| 5 | 64 | 58 | 60 | 11 | 1.00e+00 | 1.282 | 1.798 |
| 6 | 64 | 55 | 58 | 1 | 9.31e-01 | 1.838 | 2.177 |
| 7 | 64 | 55 | 59 | 2 | 7.74e-01 | 2.014 | 2.092 |
| 8 | 64 | 58 | 59 | 2 | 6.32e-01 | 2.245 | 2.150 |
| 9 | 64 | 53 | 55 | 2 | 8.13e-01 | 2.388 | 2.161 |
| 10 | 64 | 58 | 61 | 0 | 7.41e-01 | 4.849 | 2.341 |
| 11 | 64 | 52 | 53 | 1 | 1.00e+00 | 3.735 | 2.284 |
| 12 | 64 | 58 | 60 | 0 | 7.44e-01 | 6.143 | 2.724 |
| 13 | 64 | 57 | 61 | 0 | 8.85e-01 | 3.026 | 2.397 |
| 14 | 64 | 55 | 57 | 0 | 9.39e-01 | 4.962 | 2.312 |
| 15 | 64 | 59 | 59 | 0 | 8.95e-01 | 7.195 | 2.555 |
| 16 | 64 | 57 | 57 | 0 | 9.26e-01 | 13.842 | 2.233 |
| 17 | 40 | 37 | 38 | 0 | 6.76e-01 | 26.517 | 1.589 |
| 18 | 30 | 26 | 27 | 0 | 6.35e-01 | 54.011 | 4.390 |
| 19 | 20 | 17 | 17 | 2 | 8.76e-01 | 109.629 | 6.210 |
| 20 | 14 | 14 | 14 | 8 | 3.54e-04 | 204.364 | 25.016 |

## Mismatches

| game_id | N | BR path match | first divergence | converged (py/jl) | num_iter (py/jl) | max abs diff |
|---|---|---|---|---|---|---|
| 5 | 3 | false | 169 | false/false | 200/200 | 1.776e-01 |
| 16 | 3 | false | 191 | false/false | 200/200 | 3.246e-01 |
| 41 | 3 | false | 170 | false/false | 200/200 | 7.032e-01 |
| 66 | 4 | false | 200 | false/false | 200/200 | 1.407e-01 |
| 94 | 4 | false | 186 | false/false | 200/200 | 7.315e-02 |
| 136 | 5 | true | - | false/false | 200/200 | 5.015e-03 |
| 145 | 5 | false | 197 | false/false | 200/200 | 2.079e-01 |
| 160 | 5 | false | 152 | false/false | 200/200 | 1.000e+00 |
| 163 | 5 | false | 199 | false/false | 200/200 | 4.035e-01 |
| 166 | 5 | true | - | false/false | 200/200 | 1.669e-03 |
| 186 | 5 | false | 186 | false/false | 200/200 | 5.561e-01 |
| 204 | 6 | true | - | false/false | 200/200 | 4.095e-03 |
| 211 | 6 | false | 190 | false/false | 200/200 | 3.760e-05 |
| 219 | 6 | false | 184 | false/false | 200/200 | 5.087e-01 |
| 220 | 6 | true | - | false/false | 200/200 | 2.251e-03 |
| 230 | 6 | false | 191 | false/false | 200/200 | 1.439e-01 |
| 233 | 6 | false | 191 | false/false | 200/200 | 4.765e-01 |
| 237 | 6 | false | 179 | false/false | 200/200 | 9.306e-01 |
| 238 | 6 | true | - | false/false | 200/200 | 1.359e-03 |
| 245 | 6 | false | 172 | false/false | 200/200 | 8.482e-01 |
| 257 | 7 | false | 188 | false/false | 200/200 | 7.279e-01 |
| 258 | 7 | false | 199 | false/false | 200/200 | 5.391e-01 |
| 267 | 7 | true | - | false/false | 200/200 | 1.178e-03 |
| 272 | 7 | true | - | false/false | 200/200 | 1.150e-03 |
| 274 | 7 | true | - | false/false | 200/200 | 1.258e-02 |
| 288 | 7 | true | - | false/false | 200/200 | 1.345e-03 |
| 290 | 7 | false | 181 | false/false | 200/200 | 5.308e-01 |
| 291 | 7 | false | 184 | false/false | 200/200 | 6.026e-01 |
| 304 | 7 | false | 186 | false/false | 200/200 | 7.735e-01 |
| 320 | 8 | false | 198 | false/false | 200/200 | 6.297e-01 |
| 331 | 8 | true | - | false/false | 200/200 | 6.053e-03 |
| 338 | 8 | false | 198 | false/false | 200/200 | 4.352e-01 |
| 361 | 8 | false | 198 | false/false | 200/200 | 3.817e-01 |
| 364 | 8 | false | 195 | false/false | 200/200 | 1.608e-01 |
| 370 | 8 | false | 185 | false/false | 200/200 | 6.316e-01 |
| 384 | 9 | false | 184 | false/false | 200/200 | 1.206e-01 |
| 387 | 9 | false | 200 | false/false | 200/200 | 3.881e-03 |
| 388 | 9 | false | 139 | false/false | 200/200 | 6.029e-01 |
| 395 | 9 | false | 151 | false/false | 200/200 | 3.753e-01 |
| 397 | 9 | true | - | false/false | 200/200 | 1.780e-03 |
| 398 | 9 | false | 187 | false/false | 200/200 | 3.066e-01 |
| 406 | 9 | false | 193 | false/false | 200/200 | 3.605e-01 |
| 422 | 9 | false | 177 | false/false | 200/200 | 7.293e-01 |
| 428 | 9 | false | 164 | false/false | 200/200 | 8.125e-01 |
| 433 | 9 | false | 192 | false/false | 200/200 | 1.412e-01 |
| 447 | 9 | true | - | false/false | 200/200 | 4.133e-03 |
| 458 | 10 | true | - | false/false | 200/200 | 3.181e-03 |
| 462 | 10 | true | - | false/false | 200/200 | 2.429e-03 |
| 490 | 10 | false | 195 | false/false | 200/200 | 4.306e-01 |
| 495 | 10 | false | 171 | false/false | 200/200 | 4.899e-01 |
| 505 | 10 | false | 199 | false/false | 200/200 | 7.412e-01 |
| 508 | 10 | true | - | false/false | 200/200 | 1.008e-03 |
| 516 | 11 | false | 196 | false/false | 200/200 | 7.256e-01 |
| 520 | 11 | false | 193 | false/false | 200/200 | 2.873e-01 |
| 528 | 11 | false | 180 | false/false | 200/200 | 5.471e-01 |
| 530 | 11 | false | 183 | false/false | 200/200 | 6.588e-01 |
| 532 | 11 | true | - | false/false | 200/200 | 1.555e-03 |
| 534 | 11 | false | 190 | false/false | 200/200 | 1.000e+00 |
| 545 | 11 | false | 149 | false/false | 200/200 | 5.685e-01 |
| 547 | 11 | false | 179 | false/false | 200/200 | 4.155e-01 |
| 548 | 11 | false | 182 | false/false | 200/200 | 5.000e-02 |
| 551 | 11 | false | 185 | false/false | 200/200 | 3.133e-01 |
| 554 | 11 | false | 193 | false/false | 200/200 | 6.341e-01 |
| 558 | 11 | false | 189 | false/false | 200/200 | 1.774e-01 |
| 584 | 12 | false | 200 | false/false | 200/200 | 1.501e-01 |
| 590 | 12 | false | 197 | false/false | 200/200 | 4.797e-01 |
| 597 | 12 | true | - | false/false | 200/200 | 1.356e-03 |
| 604 | 12 | false | 166 | false/false | 200/200 | 7.438e-01 |
| 632 | 12 | false | 176 | false/false | 200/200 | 5.870e-01 |
| 637 | 12 | true | - | false/false | 200/200 | 1.841e-03 |
| 642 | 13 | false | 195 | false/false | 200/200 | 8.850e-01 |
| 645 | 13 | true | - | false/false | 200/200 | 4.148e-03 |
| 660 | 13 | true | - | false/false | 200/200 | 9.780e-03 |
| 673 | 13 | false | 187 | false/false | 200/200 | 1.659e-01 |
| 689 | 13 | true | - | false/false | 200/200 | 2.310e-03 |
| 697 | 13 | false | 165 | false/false | 200/200 | 2.533e-01 |
| 698 | 13 | true | - | false/false | 200/200 | 1.535e-03 |
| 711 | 14 | false | 169 | false/false | 200/200 | 5.536e-01 |
| 712 | 14 | false | 200 | false/false | 200/200 | 8.223e-03 |
| 716 | 14 | false | 186 | false/false | 200/200 | 4.928e-01 |
| 719 | 14 | true | - | false/false | 200/200 | 1.319e-03 |
| 740 | 14 | false | 192 | false/false | 200/200 | 5.902e-01 |
| 760 | 14 | true | - | false/false | 200/200 | 1.766e-03 |
| 764 | 14 | false | 197 | false/false | 200/200 | 6.171e-01 |
| 765 | 14 | false | 191 | false/false | 200/200 | 9.390e-01 |
| 767 | 14 | false | 182 | false/false | 200/200 | 2.258e-01 |
| 772 | 15 | false | 189 | false/false | 200/200 | 4.744e-01 |
| 775 | 15 | false | 198 | false/false | 200/200 | 8.955e-01 |
| 804 | 15 | false | 167 | false/false | 200/200 | 7.314e-01 |
| 818 | 15 | false | 163 | false/false | 200/200 | 8.755e-01 |
| 823 | 15 | false | 179 | false/false | 200/200 | 6.900e-01 |
| 849 | 16 | false | 198 | false/false | 200/200 | 2.916e-01 |
| 862 | 16 | false | 196 | false/false | 200/200 | 2.610e-01 |
| 877 | 16 | false | 175 | false/false | 200/200 | 6.064e-01 |
| 882 | 16 | false | 199 | false/false | 200/200 | 5.065e-01 |
| 888 | 16 | false | 162 | false/false | 200/200 | 6.068e-01 |
| 893 | 16 | false | 200 | false/false | 200/200 | 6.418e-03 |
| 895 | 16 | false | 145 | false/false | 200/200 | 9.258e-01 |
| 903 | 17 | false | 185 | false/false | 200/200 | 6.761e-01 |
| 931 | 17 | false | 188 | false/false | 200/200 | 4.855e-01 |
| 935 | 17 | true | - | false/false | 200/200 | 1.101e-03 |
| 936 | 18 | true | - | false/false | 200/200 | 1.255e-03 |
| 941 | 18 | false | 197 | false/false | 200/200 | 6.350e-01 |
| 961 | 18 | false | 197 | false/false | 200/200 | 1.587e-01 |
| 963 | 18 | false | 189 | false/false | 200/200 | 6.152e-01 |
| 966 | 19 | false | 169 | false/false | 200/200 | 8.765e-01 |
| 976 | 19 | false | 200 | false/false | 200/200 | 1.237e-02 |
| 978 | 19 | false | 200 | false/false | 200/200 | 5.912e-05 |

Times are wall-clock seconds per game; both sides were run with several games in parallel, so they are indicative only.
