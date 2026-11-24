# pe-omp-llm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24489050.svg)](https://doi.org/10.5281/zenodo.24489050)

Code, prompts, LLM answers and measurements for the paper

> **Heterogeneity-Blind? Evaluating LLM-Generated OpenMP Code on an Asymmetric P/E-Core Processor**
> Mohamed Qurrain Shayeeb and Mallegowda M, Ramaiah Institute of Technology, Bangalore, 2026.

We asked LLMs to parallelize 15 serial C kernels with OpenMP and measured their code on an
Apple M4 (4 performance + 6 efficiency cores). This repository holds everything needed to check
or extend the results: the kernels, the exact prompts, every raw model reply, the extracted code,
all timing and power logs, and the scripts that turn them into the paper's tables, figures and numbers.

## Layout

| Path | Contents |
|---|---|
| `kernels/` | The 15 serial C kernels (`k01_saxpy.c` … `k15_cutoff.c`) and the shared helper `common.h`. The loop to parallelize sits between `KERNEL BEGIN` and `KERNEL END`. |
| `prompts/` | The measurement-feedback (`fb/`) and explicit E-core request (`hwx/`) prompts. The plain and hardware-described (`hw`) prompts are built by `collect.py` (`COMMON` and `HW`); every raw reply in `llm_raw/` also stores the exact prompt that was sent. |
| `llm_raw/` | Every raw model reply as JSON, plus `manifest.jsonl` (model, prompt style, sample, time). Links to private chat sessions were replaced with `[private-chat-link-removed]`. |
| `variants/<kernel>/` | The code that was compiled and timed: expert baselines (static, dynamic, guided, calibrated split) and every LLM answer, named `<model>__<style>__s<sample>.c`. |
| `results/` | Raw measurements: `timings_*.csv` (one row per timed run), `power_*.csv` (parsed `powermetrics` CPU power and per-core activity), `power_raw/*.txt.gz` (unparsed `powermetrics` output), core-placement probes (`probe*`, `pe_speed*`), `schedules.csv`, `compile_errors/` and the run logs. |
| `analysis/` | `analyze.py` (tables, statistics, `paper/numbers.tex`), `figures.py` (figures, `paper/numbers_fig.tex`) and the manual classification files `variant_class*.csv`. |
| `paper/` | LaTeX source (IEEEtran), bibliography, figures and the compiled PDF. |
| `run.py` | Builds, checks correctness against the serial checksum and times every variant. |
| `collect.py`, `ds_collect.py`, `import_web.py`, `make_feedback.py`, `make_experts.py`, `make_calib.py` | Collect answers from the Gemini and DeepSeek APIs, import answers saved from the web chats, and build the feedback, expert and calibrated variants. |
| `overnight*.sh`, `probe*.sh`, `pe_speed.py`, `parse_power.py` | Measurement drivers: overnight benchmark runs, core-placement probes and the `powermetrics` parser. |

Measurement runs: run 1 = `timings_lpmoff.csv`, run 2 = `timings_run2.csv` (histogram after the vectorizer fix, Claude),
run 3 = `timings_run3.csv` (feedback and explicit-request answers) and `timings_oversub.csv` (20 threads),
run 4 = `timings_run4.csv` (DeepSeek API). Each run includes its own same-night baselines.

## Reproduce the paper's numbers (any OS)

Requires Python 3.10+ with pandas, numpy, scipy and matplotlib.

```bash
pip install pandas numpy scipy matplotlib
python3 analysis/analyze.py     # tables + paper/numbers.tex
python3 analysis/figures.py     # paper/figs/ + paper/numbers_fig.tex
cd paper && latexmk -pdf main.tex
```

Every number in the paper is a LaTeX macro written by these two scripts; none is typed by hand.

## Re-run the measurements (macOS on Apple silicon)

Requires Apple clang and Homebrew `libomp` (`brew install libomp`). Close other apps, turn Low Power Mode off and plug in the charger.

```bash
python3 run.py calibrate                 # serial time of every kernel
caffeinate -i python3 run.py run         # full benchmark (several hours)
sudo powermetrics --samplers cpu_power -i 250 > results/power.txt   # in a second terminal, optional
```

Compiler flags: `-O2 -ffp-contract=off -Xpreprocessor -fopenmp` with Homebrew libomp; the histogram kernel is also built with `-fno-vectorize`
because Apple clang 21 miscompiles its vectorized form (see the paper).

Collecting new LLM answers needs your own API keys in `.gemini_key` / `.deepseek_key` (one line each, never commit them;
both are listed in `.gitignore`).

## Models

Gemma 4 31B and Gemini 3.8 Flash (Gemini API), ChatGPT (chatgpt.com, Go plan, Think off), DeepSeek (chat.deepseek.com,
DeepThink and Search off), DeepSeek API (`deepseek-flash`, V4.1-Flash-1017, thinking disabled) and Claude
(claude.ai, Opus 5.5, default effort). Collection dates are in `llm_raw/manifest.jsonl`.

## License

Code (`*.py`, `*.sh`, `*.c`, `*.h`): MIT, see `LICENSE`.
Data, prompts, model outputs, figures and paper text: CC BY 4.0, see `LICENSE-DATA`.
Model outputs are included for research reproducibility; each provider's terms of use also apply to them.

## Citation

See `CITATION.cff`, or cite the paper above. Archived on Zenodo: [10.5281/zenodo.24489050](https://doi.org/10.5281/zenodo.24489050) (always resolves to the latest version).
