# Generic

Denoised Monte Carlo (DMC) from [arXiv:2402.12528](https://arxiv.org/abs/2402.12528) — variance reduction via control variate π and Legendre quadrature.

Two implementations: C++ (original) and Python (AADC wheel).

---

## Python version

### Requirements

```
pip install aadc numpy scipy    # Python 3.9+
```

### Usage

```bash
python generic_python.py Examples/Heston.json
python generic_python.py Examples/SabrScalar.json
python generic_python.py Examples/HestonMultiDim.json
python generic_python.py Examples/SABR.json
python generic_python.py Examples/SABR2.json
```

### What it does

1. Records AADC kernel: simulates one MC path (Heston/SABR), computes payoff + FD Hessian at Legendre quadrature points
2. Replays kernel via `aadc.evaluate()`: batch pricing + AAD Greeks (delta)
3. DMC correction: `price = E[payoff] - E[quadrature]`

### Supported models

| Process | Dimensions | JSON |
|---|---|---|
| Heston | 1 | `Heston.json` |
| SABR (scalar) | 1 | `SabrScalar.json` |
| Heston (multi-asset) | 10 | `HestonMultiDim.json` |
| SABR (multi-asset) | 3, 10 | `SABR.json`, `SABR2.json` |

### Payoffs

EuropeanCall, EuropeanCallAsian, LookbackCall, DownAndOutEuropCallPayoff, BasketCall, RainbowCallOnMax

### Control variates (Pi)

BlackScholes, BachelierAsian, BlackLookbackCall, DownAndOutEuropCallPrice

### Performance

| Process | d | Steps | Recording | Replay 50K paths |
|---|---:|---:|---:|---:|
| Heston | 1 | 200 | 0.13s | 0.59s |
| SabrScalar | 1 | 200 | 0.13s | 0.61s |
| HestonMultDim | 10 | 200 | 0.48s | 4.28s |
| SABR | 10 | 200 | 0.43s | 4.77s |

Recording: one-time (Python ~50× slower than C++ due to interpreter). Replay: identical speed (same AADC compiled kernel).

### Validation

All 53 test cases across 5 JSON configs: AAD delta matches bump-and-reval (diff ~1e-4).

---

## C++ version

### Build

1. Provide correct links in CMakeLists.txt to Nlohmann (JSON), Eigen and AADC.
2. Parallel/Non-parallel version switched via `#define PARALLEL_IMPLEMENTATION` in generic.cpp (parallel on by default).

```bash
cd build
make
./Generic/par ../Generic/Examples/Heston.json
```

### Bug fixes

Branch `fix-paper-bugs`: 6 bugs found and fixed while reproducing paper tables. See [generic_fixes_v2](http://dev.matlogica.com:8888/generic_fixes_v2.html).
