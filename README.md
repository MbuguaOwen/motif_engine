# Motif Engine — Walk‑Forward with Class‑Aware Multivariate Motifs

This repo mines **multivariate shapelets (motifs)** from labeled data and uses them in a **walk‑forward simulation**. The mined artifacts (`.pkl`) act as the system’s informed memory of what “good” vs “bad” looked like in training; a configurable **gate** then decides whether to trade in the test month, and a clean **ATR‑based risk manager** simulates P\&L (SL/TP/BE/TSL).

---

## Pipeline at a glance

1. **Features & labels** are built from raw bars (derived from ticks).
2. **Mining** finds class‑aware shapelets for each horizon (macro/meso/micro) and persists them to `.pkl` artifacts.
3. **Walk‑forward** loads the artifacts, scores each new bar against motif banks, gates entries (trend‑aligned, K‑of‑N, BAD‑aware, discord‑veto), and simulates trade outcomes with your risk rules.

**Artifacts = memory of good/bad patterns.
Live features = what the market looks like now.
Gate + risk = how we act on that memory in real time.**

---

## Data expected

Place **tick CSVs** under `inputs/<SYMBOL>/` (UTC, ms timestamps):

```
inputs/
  BTCUSDT/
    BTCUSDT-ticks-2025-01.csv
    BTCUSDT-ticks-2025-02.csv
    ...
  ETHUSDT/
  SOLUSDT/
```

Columns: `timestamp,price,qty,is_buyer_maker`

Bars (1‑minute) are built internally during the **features** step.

---

## Quickstart (commands you actually run)

1. Make sure `configs/motifs.yaml` is set the way you want (features, horizons, labeling, gating, and risk).

```bash
# (A) Build features (also prints global label counts)
python -m run_motifs --config configs/motifs.yaml --mode features

# (B) Mine class‑aware, multivariate shapelets per fold + persist
python -m run_motifs --config configs/motifs.yaml --mode mine --persist-artifacts

# Quick sanity on what was mined (counts per bank + ε per horizon)
python -m pk_count

# (C) Walk‑forward using the persisted artifacts (no re‑mining)
python -m run_motifs --config configs/motifs.yaml --mode walkforward --use-artifacts --export-csv
```

Outputs land under `outputs/` with logs, artifacts per fold, and evaluation JSON/CSV.

---

## What’s inside a `.pkl` artifact?

Per **symbol**, **fold** (3 train months → next test month), and **horizon** (macro/meso/micro), the `.pkl` stores:

* **Class‑aware motif banks**

  * `classes.long.good`, `classes.long.bad`, `classes.short.good`, `classes.short.bad`
  * Each bank holds many **multivariate shapelets** (fixed feature list) and its **ε** tolerance.
* **Discords** (outlier patterns to veto).
* The **feature list** and **window length `L`** expected by those shapelets.

> Think of a fold’s artifact as: “what good/bad looked like in those training months” for each horizon.

---

## How “good” is defined (class labeling)

* **Triple‑Barrier (TB) Labels (micro)**: For each candidate start bar `t`, we check first touch of **up** vs **down** barrier within a timeout.

  * Up barrier at `close_t + up_mult * ATR_t`
  * Down barrier at `close_t - dn_mult * ATR_t`
  * TB = +1 if **up** hits first; TB = −1 if **down** hits first; 0 on timeout.
* **Macro alignment**: We tag motif windows relative to the macro trend sign.

  * If **TB=+1** and **macro>0** → goes to **`long.good`**
  * If **TB=−1** and **macro<0** → goes to **`short.good`**
  * Opposite‑sign outcomes land in the respective **`*.bad`** banks. (Timeouts ignored.)

Your TB multipliers come from `configs/motifs.yaml` under **`labels`**. If your **RiskManager** uses `tp_mult=60` and `sl_mult=20`, set **`labels.barrier_up_atr: 60`**, **`labels.barrier_dn_atr: 20`** to keep training labels aligned with trade geometry.

---

## What happens during walk‑forward

For each **test‑month** bar:

1. **Compute live features** at macro/meso/micro windows.
2. **Match** each horizon’s window against that horizon’s motif banks to get scores:

   * `S_good` (best match among good shapelets)
   * `S_bad` (best match among bad shapelets)
   * **Discord score** (if an outlier matches)
3. **Gate** the bar using your config:

   * **Trend alignment**: `pick_side` combines macro sign with `long_only_if_macro_up` / `short_only_if_macro_dn`.
   * **K‑of‑N horizons**: e.g., at least 2 of {macro, meso, micro} must clear their floors.
   * **Margin vs BAD**: require `(S_good - α·S_bad) ≥ margin_min`; `bad_max` is a hard veto.
   * **Discord veto**: if any horizon’s discord ≥ threshold, **skip**.
   * **Cooldown**, etc.
4. **Trade & manage risk** if passed: SL/TP, break‑even (BE), trailing stop (TSL) per **`risk`** in YAML.

---

## Reading the logs (what to look for)

**Features step**

```
[LABELS] phase=None used up=60.0 dn=20.0 timeout=2880 use_high_low=True | counts(all)={-1:…,0:…,1:…}
```

Confirms triple‑barrier labeling and global class counts.

**Mining step**

* TRAIN label counts per fold (non‑zero ±1 is essential).
* `[EPS]` blocks (chosen ε per horizon/side/kind).
* "persisted artifacts to artifacts/<SYMBOL>\_<train>\_\_<test>/shapelet\_artifacts.pkl".

`python -m pk_count`

* Compact table of counts per bank/horizon + the ε chosen per horizon.

**Walk‑forward step**

* First‑N decisions (if `ui.simulate_debug_first_n` is set) show why bars were accepted/rejected: `trend_guard`, `k_of_n`, `score`, `margin`, `bad`, `discord`, or `OK`.

**Fold summary example**

```json
{"symbol":"BTCUSDT","train_months":[...],"test_months":["2025-04"],
 "trades":13,"sum_R":3.94,"avg_R":0.30,"median_R":0.0}
```

**Decision summary example**

```
[SIMDBG] decision summary: {"ok":903,"score":6071,"bad":0,"margin":0,
                            "macro_req":0,"meso_req":0,"micro_req":6224,
                            "discord":0,"k_of_n":0,"trend_guard":11818}
```

Use these tallies to tune the gate:

* **Too much `k_of_n`** → ease per‑horizon floors or the K‑of‑N requirement.
* **Too much `score`** → lower `score_min` slightly.
* **Too much `bad/margin`** → tweak `alpha` / `margin_min` / `bad_max`.
* **Many `trend_guard` vetoes in an up month** → ensure macro sign aligns with your macro trend feature.

---

## Tuning to “milk more R” (safe levers)

**Agreement**

* Start with `require_at_least_k_of: 1` and soft floors (`score_min_per_horizon`: macro≈0.03, meso≈0.05, micro≈0.08).
* If too many trades, bump to **K=2** or raise `score_min` to 0.22–0.25.

**Selectivity vs BAD**

* `alpha` (0.45–0.55), `margin_min` (0.05–0.08), `bad_max` (0.65–0.75).
* Higher `alpha` and lower `bad_max` block shakier entries.

**Discords**

* If discord veto fires too often, ease `discord_block_min` slightly (e.g., 0.995 → 0.992).

**Risk (convert near‑wins & stretch runners)**

* Earlier BE: `be_at_R: 0.50–0.65`.
* Slightly tighter trail: `tsl.atr_mult: 18–20`.
* Allow fatter tails if stable: `tp_mult: 80–100`.

---

## Common gotchas

* **Features & order must match mining**. If you change `motifs.features` or a horizon’s `L`, re‑run **mine**.
* **Artifacts are fold‑specific** (encode the 3 train months used). The runner picks the right `.pkl` for each test month automatically.
* **Zero trades** usually means the gate is too strict:

  * K‑of‑N too high, per‑horizon floors too high,
  * `bad_max` / `alpha` / `margin_min` too harsh,
  * `trend_guard` eliminating the only available side.

---

## Outputs

* `outputs/trades.csv` — all fills with timestamps, side, price, SL/TP/TSL events, and R.
* `outputs/summary.json` — per‑fold summary with counts and P\&L aggregates.
* Console `[SIMDBG]` lines — first‑N decision reasons (if enabled).

---

## FAQ

**Q: Are we fully relying on `.pkl` artifacts to decide?**
**A:** Yes. With `--use-artifacts`, decisions come from matching live feature windows to the motif banks in the `.pkl`. Gate/risk then act on those scores. Labels are used only during mining to build class banks.

**Q: Do artifacts encode actual TP vs SL?**
**A:** Artifacts are classed by TB + macro alignment (`good`/`bad`). Realized TP/SL outcomes come from the RiskManager during simulation. To align labels with risk, set `labels.barrier_up_atr`/`labels.barrier_dn_atr` ≈ `risk.tp_mult`/`risk.sl_mult`.

**Q: When do I need to re‑mine?**
**A:** Whenever you change the feature set, window lengths `L`, labeling scheme, or mining ranges (`top_k`, `keep`, etc.).

---

## Configuration quick reference (YAML)

```yaml
labels:
  barrier_up_atr: 60       # keep aligned with risk.tp_mult (e.g., 60)
  barrier_dn_atr: 20       # keep aligned with risk.sl_mult (e.g., 20)
  timeout_bars: 2880       # e.g., 2 days on 1m bars

motifs:
  features: [ret_z, atr_z, donch_pos, kama_slope, r2, wick_up, wick_dn, clv]
  horizons:
    macro: { L: 40, candidate_stride: 1, top_k: 32 }
    meso:  { L: 60, candidate_stride: 1, top_k: 64 }
    micro: { L: 30, candidate_stride: 1, top_k: 128 }

  gating:
    require_at_least_k_of: 2       # K-of-N horizons must pass
    score_min_per_horizon:
      macro: 0.08
      meso:  0.10
      micro: 0.12
    alpha: 0.5                     # penalty factor on bad
    margin_min: 0.06               # require S_good - alpha*S_bad ≥ margin_min
    bad_max: 0.70                  # hard veto if S_bad ≥ bad_max
    discord_block_min: 0.995       # veto on strong discord

risk:
  tp_mult: 60.0
  sl_mult: 20.0
  be_at_R: 0.60
  tsl:
    enabled: true
    atr_mult: 20.0
    floor_from_median_mult: 1.0    # ATR floor so trail can’t collapse on compression
```

> If you prefer different multipliers by side (e.g., 60/20 for shorts only), extend the config with `tp_mult_long`, `sl_mult_long`, `tp_mult_short`, `sl_mult_short` and wire them in the RiskManager.

---

## Suggested run flow (copy/paste)

```bash
# (A) Build features (also prints global label counts)
python -m run_motifs --config configs/motifs.yaml --mode features

# (B) Mine class‑aware, multivariate shapelets per fold + persist
python -m run_motifs --config configs/motifs.yaml --mode mine --persist-artifacts

# Quick sanity on what was mined (counts per bank + ε per horizon)
python -m pk_count

# (C) Walk‑forward using the persisted artifacts (no re‑mining)
python -m run_motifs --config configs/motifs.yaml --mode walkforward --use-artifacts --export-csv
```

---

## Notes on internals

* **MASS** via NumPy FFT approximates matrix profiles for fast nearest‑neighbor distances.
* **Neighbor density** select‑rule: shapelets are medoids minimizing mean k‑nearest MASS distances; discords maximize it.
* **No external network deps**; core stack: `numpy`, `pandas`, `pyyaml` (plus `tqdm` for progress bars if enabled).
* **Deterministic**: given fixed inputs/config, mining and simulation are repeatable.

---

## High‑end defaults (illustrative)

You can start strict and relax if participation is too low:

* Labeling defaults (example): up barrier **60×ATR**, down barrier **20×ATR**, timeout **\~2 days**.
* Gate defaults (example): `K=2`, modest `score_min` per horizon, `alpha≈0.5`, `bad_max≈0.7`, `discord_block_min≈0.995`.
* Risk defaults (example): `tp_mult=60`, `sl_mult=20`, `BE@0.6R`, `TSL atr_mult≈20` with median‑ATR floor.

Tune using the **decision summary** tallies explained above.
