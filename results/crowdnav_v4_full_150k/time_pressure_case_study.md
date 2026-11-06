# Case Study: `time_pressure` LOFO Failure (CrowdNav v4)

**Campaign:** `crowdnav_v4_full_150k`  
**Surrogate:** ExtraTrees / `early+math+behavioral` / `success_rate`  
**Symptom:** Mean within-family Spearman ≈ **0.12**; Top-3 hit = **0** (only failing family of 12).

---

## 1. What the labels look like

Seed-mean success rates (3 seeds):

| candidate | mean SR | std | seed min–max |
|-----------|--------:|----:|--------------|
| time_pressure_03 | 0.692 | 0.088 | 0.60–0.78 |
| time_pressure_02 | 0.690 | 0.079 | 0.60–0.75 |
| time_pressure_01 | 0.665 | 0.058 | 0.63–0.73 |
| time_pressure_04 | 0.594 | 0.075 | 0.52–0.67 |
| time_pressure_05 | 0.573 | 0.051 | 0.54–0.63 |
| time_pressure_00 | 0.475 | 0.125 | 0.34–0.58 |

**True order (coarse):** `{03, 02, 01}` ≻ `{04, 05}` ≻ `{00}`  

Top-3 are **near-ties**: gaps of ~0.00–0.03 while seed SE is ~0.03–0.05. Ranking among 03/02/01 is only weakly identifiable.

---

## 2. What the surrogate predicts (OOF)

Typical repeat-0 scores:

| candidate | y_true | y_pred | issue |
|-----------|-------:|-------:|-------|
| _03 | 0.692 | 0.551 | true best underranked |
| _02 | 0.690 | 0.660 | OK-ish |
| _01 | 0.665 | 0.449 | **severely underranked** |
| _04 | 0.594 | 0.646 | **overranked** (often predicted top) |
| _05 | 0.573 | 0.577 | mid |
| _00 | 0.475 | 0.541 | overranked vs mid |

Predicted top-3 repeatedly centers on `_04 / _02 / _05` (or similar), missing true best `_03` and especially `_01` → Top-3 hit = 0 every repeat.

Spearman across repeats: 0.20 / 0.09 / 0.09.

---

## 3. Weight structure (why the family is hard)

Non-zero weights are dominated by **`goal`** (≈14–31) and **`collision`** (≈−5…−18).  
The namesake **`time`** weight is tiny (≈ −0.05…−0.20).

So within-family diversity is mostly “goal vs collision tradeoff,” not a clean time-pressure axis. Static/math features that emphasize `weight_time` ratios will not separate these candidates well.

---

## 4. Do features contain within-family signal?

Yes — locally:

| Feature (examples) | Spearman with SR inside family |
|--------------------|-------------------------------:|
| `early_t12000_path_length_iqr` | +0.94 |
| `early_t12000_min_clearance_p10` | −0.94 |
| several early length/clearance stats | \|ρ\| ≈ 0.83–0.89 |

**Conclusion:** early behavioral features *do* track success inside `time_pressure`.  
The LOFO ExtraTrees fit on the **other 11 families** does not transfer that local mapping — classic family-shift / representation mismatch, not “no signal in X.”

Many columns are globally constant (shared with other families); that alone does not explain the failure. The failure is **wrong ranking under transfer**, amplified by **near-tie labels**.

---

## 5. Diagnosis (evidence-backed)

| Hypothesis | Status for `time_pressure` |
|------------|----------------------------|
| Flat / non-identifiable family | **Partial** — family identifiable overall, but top-3 nearly tied + seed noise |
| Missing feature signal | **Unsupported** — strong early-feature correlations exist locally |
| Between-family model ignores local structure | **Supported** — OOF systematically overranks `_04`, underranks `_01/_03` |
| Misleading family geometry (goal≫time) | **Supported** — static “time pressure” axis is weak |

**Working diagnosis:**  
`time_pressure` is a **transfer failure on a near-tie top band**, not a dead family. Surrogate triage still finds “good enough” mid/high members (`_02`) but misses the exact best.

---

## 6. What to do (minimal, ordered)

1. **Reporting:** keep excluding this family from “healthy within-mean” narratives or footnote it as the sole Top-3 miss.  
2. **No new PPO required** for understanding — already explained.  
3. **Optional cheap fix tests (0 PPO):**  
   - ablate using only early features for a diagnostic LOFO score on this family;  
   - add goal/collision ratio features already present and check OOF for this fold only.  
4. **If regenerating the pool later:** widen `weight_time` range and shrink goal dominance so the family matches its name.  
5. **Adaptive seeds:** only if you care about separating `_03` vs `_02` vs `_01` scientifically; not needed for coarse triage.

---

## 7. Takeaway for the thesis

CrowdNav@150k surrogate quality is **not uniformly weak**.  
Global within-Spearman 0.75 / Top-3 0.92 is driven by 11 healthy families; headline variance is largely **one** pathological transfer case plus near-ties — consistent with the earlier 2D audit lesson (heterogeneous within-family errors).
