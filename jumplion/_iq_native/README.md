# iq MaxLFQ, ported to a standalone C++/pybind11 extension

`_iq_native/src/maxlfq_core.h` is a direct port of the `ion_table` class and
grouping loop that live inside `iq_MaxLFQ()` in tvpham/iq's `src/iq.cpp`.
The only thing removed is the R marshaling layer (SEXP unpack/pack, Rprintf
progress messages, R interrupt checks) — the grouping logic and the
least-squares MaxLFQ solve are unchanged. Verified against the worked
example in the iq README (2-ion, 3-sample case → `[11, 13, 9]`), and against
a 2-protein case to confirm proteins are handled independently and correctly.

Source package: https://github.com/tvpham/iq (BSD-3-Clause). Keep the
license/citation notice if you redistribute this.

## Files

- `_iq_native/src/utils.h` — median helper, copied unmodified (already pure C++)
- `_iq_native/src/maxlfq_core.h` — the ported algorithm (`iq_native::run_maxlfq`)
- `_iq_native/src/bindings.cpp` — pybind11 glue, numpy in/out via `pybind11/eigen.h`
- `_iq_native/thirdparty/eigen/` — vendored Eigen headers (MPL2, trimmed to the
  dense modules used here — see `LICENSE.txt` inside). Vendored so the build
  needs **no system C++ dependency on any platform**, including Windows,
  where there's no `apt`-equivalent for a header-only C++ lib.
- `maxlfq.py` — thin Python wrapper matching R's `fast_MaxLFQ()` interface
  (factorizes protein/sample labels, calls the extension, maps back to labels)
- `setup.py`, `pyproject.toml.snippet` — build config to merge into your package

## Integrating into your existing package

1. Copy `_iq_native/` (including `thirdparty/eigen/`) into your package
   directory, and `maxlfq.py` next to it.
2. In `bindings.cpp`, the module is registered as `_iq_native` — rename the
   `PYBIND11_MODULE(...)` call and the `Pybind11Extension("yourpackage._iq_native", ...)`
   name in `setup.py` to match your actual package path.
3. Merge `pyproject.toml.snippet` into your `pyproject.toml`'s `[build-system]`.
4. Merge the `ext_modules=[...]` from `setup.py` into your own `setup.py`
   (or keep it standalone if you don't have one).
5. `pip install -e .` — no Eigen install step needed; `setup.py` points at
   the vendored copy by default. Set `EIGEN_INCLUDE` to override with your
   own Eigen install if you'd rather not vendor it.
6. Works with MSVC (Windows), GCC, and Clang — `setup.py` picks compiler-
   appropriate flags (`/openmp` + `/std:c++17` via `cxx_std=17` on MSVC,
   `-fopenmp` on GCC/Linux). OpenMP is optional; the code runs correctly
   single-threaded if you'd rather drop the OpenMP build dependency
   entirely (e.g. plain Clang on macOS without libomp) — just remove the
   relevant lines in `setup.py`.

## Usage

```python
from yourpackage.maxlfq import fast_maxlfq

result = fast_maxlfq(
    protein_list=[...],  # one entry per observation
    sample_list=[...],
    ion_id=[...],         # fragment/peptide/precursor id, the grouping key
    quant=[...],          # log2 intensities, pre-filtered & normalized
)

result.estimate        # (n_proteins, n_samples) numpy array
result.protein_names   # row labels
result.sample_names    # column labels
result.annotation      # per-protein connectivity annotation, same as R's $annotation
```

Works transparently with any number of proteins — pass your whole
long-format table in one call (the underlying C++ loop is already
parallelized across proteins with OpenMP) and index `result.estimate` /
`result.protein_names` afterward, rather than looping per-protein in Python:

```python
row = result.protein_names.index("P12345")
profile = result.estimate[row]   # this protein's per-sample abundance
```

Note: `quant` should already be filtered and (typically) median-normalized,
same expectation as R's `fast_MaxLFQ()` — the preprocessing step
(`fast_preprocess()` in R) isn't part of this port; that's ordinary
pandas/numpy work on your side, not something worth calling into C++ for.

## What wasn't ported

`iq_filter` and `iq_quant` (the other two `.Call`-registered functions in
iq.cpp) follow the same R-shell-around-plain-C++ pattern and could be
extracted the same way if you need them — not done here since the request
was scoped to MaxLFQ.
