# Execution Plan — اصلاح بر اساس Audit

**هدف:** کوچک‌ترین مسیر قابل‌اعتماد برای بهبود within-family ranking و آماده‌سازی selection، بدون خراب کردن baseline قفل‌شده.

**Baseline قفل (دست نزن / overwrite نکن):**  
`experiments/model_comparison_confirm/` → ExtraTrees / early+math+behavioral ≈ Spearman 0.935 / within 0.648 / top3 0.833

**قانون طلایی:** تا وقتی Stage 0–1 تمام نشده، مدل جدید یا PPO جدید شروع نشود.

---

## نمای کلی اولویت‌ها

| Phase | نام | PPO؟ | خروجی کلیدی | توقف اگر… |
|-------|-----|------|-------------|-----------|
| **0** | ایمنی و provenance | خیر | آرشیو confirm + baseline درست | آرشیو نیست |
| **1** | تشخیص + تعریف estimand | خیر | جدول identifiable families + سیاست متریک | — |
| **2** | اصلاح متریک و بنچمارک منصفانه | خیر | معیار within قابل‌دفاع + مقایسه مدل‌ها | — |
| **3** | سیگنال (فیچر / ranking) | خیر | برندهٔ کم‌هزینه vs ET | Δ ضمن CI شامل 0 → متوقف شو |
| **4** | کیفیت برچسب (seed تطبیقی) | کم | seed plan + در صورت تأیید اجرا | order flip ناچیز |
| **5** | Selection / AL شبیه‌سازی | خیر→بعداً | regret–budget | شبیه‌سازی ضعیف → AL نرو |
| **6** | تعمیم (اختیاری) | جداگانه | pool جدید / CrowdNav | فقط با مجوز صریح |

---

## Phase 0 — ایمنی (روز ۱، قبل از هر چیز)

### 0.1 آرشیو confirm
- [x] کپی کامل `experiments/model_comparison_confirm/` و `results/model_comparison_confirm/` به  
  `experiments/archives/model_comparison_confirm_2026-11-05/`
- [x] در `docs/DECISIONS.md` یک خط: «baseline قفل = این آرشیو؛ مسیر زنده را بدون run_id جدید overwrite نکن»

### 0.2 تصحیح استناد اعداد
- [x] DECISIONS: اعداد تأییدشده 0.935/0.648 ثبت شد؛ 0.8448/0.5751 به‌عنوان 2D رد شد
- [ ] جاروی کامل docs/README برای هر ارجاع قدیمی باقی‌مانده (در صورت نیاز)

### 0.3 جلوگیری از overwrite تصادفی
- [x] سیاست در DECISIONS: بدون run_id جدید overwrite نکن
- [ ] سخت‌سازی کد `generate_candidates` / quality pipeline برای مسیر خروجی نسخه‌دار (Phase 2+)

**Done when:** آرشیو موجود است و تیم روی یک baseline عددی توافق دارد. ✅ (2026-11-05)

---

## Phase 1 — تشخیص (اولین کار علمی؛ ۰ PPO)

### 1.1 اسکریپت Identifiability (Task T1)
- [x] `scripts/analyze_within_family_identifiability.py`
- [x] خروجی: `results/diagnostics_within_family/{identifiability_table.csv,report.md,summary.json}`
- نتیجهٔ اجرا (κ=2, min_range=0.05): **11/12 identifiable**؛ فقط `distance_dense` flat  
  - Spearman all-12 ≈ **0.648** → identifiable ≈ **0.765**

### 1.2 سیاست متریک
- [x] `research_audit/METRIC_POLICY.md`

**Done when:** جدول + سیاست مکتوب؛ می‌دانیم کدام خانواده‌ها «قابل ranking» هستند. ✅ (2026-11-05)

**بعدی:** Phase 2 (فلگ متریک identifiable در pipeline + بنچمارک mean/Ridge/ET).

---

## Phase 2 — زیرساخت ارزیابی درست

### 2.1 متریک identifiable در pipeline (Task T2)
- [x] فلگ `within_family_identifiability` در `configs/model_comparison.yaml` (**default off**)
- [x] ستون‌ها در `evaluate_oof_bundle`: `within_family_spearman_identifiable_*`, `top3_hit_identifiable`, `n_families_identifiable`
- [x] تست‌ها در `tests/test_model_comparison.py`

### 2.2 Fair benchmark scaffold (Task T3)
- [x] `configs/benchmark.yaml` + `scripts/run_fair_benchmark.py` (refuse confirm path)
- [x] Quick run → `experiments/benchmark_phase2_base/` (mean / ridge / extra_trees)
- [ ] Full (non-quick) benchmark برای claim رسمی (اختیاری قبل از Phase 3)

**Done when:** یک run بنچمارک با ۳ مدل پایه روی همان LOFO سبز است. ✅ (quick 2026-11-05)

**بعدی:** Phase 3.1 فیچر ratios/deltas + prune — یا اول full benchmark اگر claim عددی می‌خواهی.

---

## Phase 3 — بهبود سیگنال (۰ PPO؛ موازی بعد از Phase 2)

ترتیب داخل Phase 3:

### 3.1 Feature cleanup اول (احتمال موفقیت بالاتر از ranker خام)
- [x] `features_ratios.csv` (51) + `features_early_deltas.csv` (12) via `scripts/build_phase3_feature_extras.py`
- [x] train-fold `min_within_family_std` در preprocessor
- [x] quick ablation → `experiments/benchmark_phase3_features/` + prune run
- **نتیجه quick:** هیچ set روی identifiable within از `early+math+behavioral` بهتر نشد؛ deltas ضرر کرد؛ prune بی‌اثر.
- **تصمیم:** operating features را نگه دار؛ ratios را promote نکن تا full run یا Phase 3.2.

### 3.2 Pairwise ranking دوم
- [x] `pairwise_logistic` (within-family-only, `min_abs_delta=0.02`) vs ExtraTrees
- [x] خروجی: `experiments/benchmark_phase3_ranking/` + `results/benchmark_phase3_ranking/`
- [x] Family bootstrap: mean Δ≈+0.07 ولی CI ۹۰٪ = [−0.13, +0.30] شامل صفر
- **تصمیم:** `fail_to_reject_keep_extra_trees` — مدل پیچیده‌تر نرو

**قانون توقف:** اگر CIِ Δ شامل ۰ بود → مدل پیچیده‌تر نرو؛ برو Phase 4. ✅ فعال شد

**Done when:** یا برندهٔ واضح در برابر ET، یا تصمیم مستند «ET کافی است». ✅ ET کافی است (تا اطلاع ثانوی)

**Phase 3 بسته شد. Phase 4 plan + Phase 5 sim انجام شد.**  
**توقف اجرایی:** برای PPOِ seedهای Phase 4 یا AL واقعی نیاز به مجوز صریح داری.

---

## Phase 4 — کیفیت برچسب (PPO فقط هدفمند)

### 4.1 فقط plan (بدون train)
- [x] `scripts/plan_adaptive_seeds.py`
- [x] خروجی: `results/adaptive_seed_plan/`
- نتیجه: **۱۲ کاندید / ۳۶ run** فقط از `risk_sensitive` + `nonlinear_progress` (۱۹۳ close pair؛ flatها skip)

### 4.2 اجرا فقط بعد از تأیید بودجه
- [ ] PPO هنوز اجرا نشده — نیاز به مجوز صریح
- سقف: ≤ 36 seed-run؛ root جدید؛ phase1 overwrite نشود
- بعد از اجرا: سنجش order-flip و LOFO مجدد

**Done when (plan):** لیست قابل‌اجرا آماده است. ✅  
**Done when (science):** بعد از PPO می‌دانی چند درصد ترتیب‌ها عوض می‌شوند.

---

## Phase 5 — Selection (Objective B)

### 5.1 شبیه‌سازی آفلاین (اجباری قبل از AL واقعی)
- [x] `scripts/simulate_surrogate_selection.py`
- [x] `results/selection_sim_phase5/`
- نتیجهٔ confirm OOF: در **k=5** surrogate به بهترین جهانی می‌رسد؛ در k=3 regret≈0.015؛ واضح بهتر از random

### 5.2 AL واقعی
- [ ] هنوز نه — شبیه‌سازی امیدوارکننده است ولی AL آنلاین نیاز به طراحی جدا + مجوز
- شرط: run_id جدید؛ hold-out family block

---

## Phase 6 — اختیاری / جداگانه

| کار | شرط |
|-----|------|
| pool ۴۸تایی جدید | مسیر جدید؛ آرشیو ۱۰۸ دست‌نخورده |
| CrowdNav | پروتکل و اعداد جدا؛ مجوز صریح |
| Deep net / multi-task سنگین | فقط اگر Phase 3 شکست سخت داد و فرضیه جدید داری |

---

## برنامهٔ پیشنهادی هفته‌ای (واقع‌گرایانه)

| بازه | کار |
|------|-----|
| روز ۱ | Phase 0 کامل |
| روز ۱–۲ | Phase 1 (اسکریپت + گزارش + سیاست متریک) |
| روز ۳–۴ | Phase 2 (فلگ متریک + بنچمارک mean/Ridge/ET) |
| روز ۵–۷ | Phase 3.1 فیچر |
| روز ۸–۱۰ | Phase 3.2 ranking (اگر 3.1 کافی نبود یا برای contrast) |
| بعد | Phase 4 فقط با بودجه تأییدشده؛ سپس 5.1 |

---

## چک‌لیست «الان دقیقاً چه کار کنیم»

1. آرشیو confirm  
2. اسکریپت identifiability + گزارش  
3. نوشتن سیاست primary metric  
4. بنچمارک منصفانه با ET به‌عنوان لنگر  
5. فیچر ratios/deltas + prune  
6. یک آزمایش ranking از پیش ثبت‌شده  
7. seed plan → (مجوز) PPO محدود  
8. selection simulation → (مجوز) AL  

---

## معیار موفقیت کلی پروژه (بعد از اصلاحات)

- Primary: ↑ identifiable within-family Spearman نسبت به ET قفل، با CI خانواده‌محور  
- Secondary: Top-3 hit / regret@3 پایدار روی repeats  
- عملیاتی: در شبیه‌سازی، همان کیفیت انتخاب با بودجه PPO کمتر از full eval  
- سخت: هیچ overwrite روی آرشیو baseline  

اگر فقط یک چیز این هفته انجام شود: **Phase 0 + Phase 1**.
