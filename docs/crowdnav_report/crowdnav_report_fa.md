<div dir="rtl" lang="fa">

# گزارش ممیزی CrowdNav: مقایسهٔ PPO 50k و 150k

**پروژه:** انتخاب تابع پاداش با مدل جانشین (surrogate) قبل از آموزش کامل PPO  
**محدوده:** فقط خروجی‌های موجود CrowdNav در مخزن؛ بدون تغییر کد/کانفیگ و بدون آموزش مجدد  
**تاریخ ممیزی:** 2026-11-06  
**محیط:** `CrowdNavPlusPlus-v0` (CrowdNav++ / ORCA humans)

## منابع اصلی عددها

| منبع | نقش |
|---|---|
| <code dir="ltr">experiments/crowdnav_v1_50k/dataset/reward_performance_dataset.csv</code> | برچسب‌ها و زمان آموزش ران 50k (دیتاست واقعی) |
| <code dir="ltr">experiments/crowdnav_v4_full_150k/dataset/reward_performance_dataset.csv</code> | برچسب‌ها و زمان آموزش ران 150k |
| <code dir="ltr">results/crowdnav_v1_50k/leaderboard.csv</code> | متریک‌های surrogate ران 50k (میانگین ۳ تکرار) |
| <code dir="ltr">results/crowdnav_v4_full_150k/leaderboard.csv</code> | متریک‌های surrogate ران 150k |
| <code dir="ltr">experiments/*/model_comparison/metrics_by_fold.csv</code> | Spearman داخل‌خانواده به‌ازای هر fold |
| <code dir="ltr">experiments/*/model_comparison/oof_predictions.csv</code> | پیش‌بینی OOF |
| <code dir="ltr">experiments/*/phase1_summary.json</code> / <code dir="ltr">phase1_manifest.json</code> | بودجهٔ PPO، شکست‌ها، زمان |
| <code dir="ltr">results/crowdnav_v4_full_150k/selection_sim/</code> | شبیه‌سازی انتخاب top-k (فقط 150k) |
| <code dir="ltr">docs/crowdnav_report/numbers.csv</code> | جدول ردیابی همهٔ عددهای استفاده‌شده |
| <code dir="ltr">docs/crowdnav_report/discovery_integrity.json</code> | ناسازگاری‌های کشف‌شده |

**هشدار یکپارچگی (بسیار مهم):** در درخت <code dir="ltr">crowdnav_v1_50k</code> فایل‌های <code dir="ltr">candidates.json</code> و <code dir="ltr">candidates_summary.json</code> با دیتاست/فیچر/OOF **هم‌خوان نیستند**. دیتاست واقعی خانواده‌های legacy دارد؛ JSON کاندید بعدها با پول CrowdNav-aware بازنویسی شده است. همهٔ عددهای 50k این گزارش از **دیتاست و model_comparison** گرفته شده‌اند، نه از <code dir="ltr">candidates.json</code>.

---

## ۱. سوال و معیار درست

هدف CrowdNav همان ۲D است: با ویژگی‌های ارزان، عملکرد PPO کامل را برای پاداش‌های دیده‌نشده پیش‌بینی کنیم تا فقط shortlist آموزش کامل ببیند.

سه معیار را نباید یکی کرد:

| معیار | معنی | خطر سوءبرداشت |
|---|---|---|
| **pooled Spearman** | ترتیب همهٔ کاندیدها با هم | با اختلاف میانگین خانواده‌ها مصنوعی بالا می‌رود |
| **within-family Spearman** | ترتیب داخل یک خانواده وقتی آن خانواده در train نبوده (LOFO) | نزدیک‌ترین معیار به تصمیم واقعی |
| **Top-3 hit** | آیا بهترین عضو واقعی خانواده داخل ۳ پیش‌بینی اول است | برای triage عملی |

در CrowdNav هم مثل ۲D باید **هر دو** pooled و within گزارش شوند.

---

## ۲. محیط، کاندیدها، خانواده‌ها، برچسب‌ها

### ۲.۱ محیط (هر دو ران، تا جایی که در خروجی ثبت شده)

| پارامتر | مقدار | منبع | قطعیت |
|---|---|---|---|
| env_id | `CrowdNavPlusPlus-v0` | ستون `env_id` دیتاست هر دو ران | قطعی |
| ارزیابی | 160 اپیزود | ستون `n_eval_episodes` دیتاست | قطعی |
| device | cpu | ستون `device` دیتاست | قطعی |
| seeds آموزش | 0,1,2 | `phase1_manifest.json` | قطعی |
| human_num | پیش‌فرض پایپلاین = 5 | `scripts/run_quality_pipeline.py --humans` | در status.json ران‌ها **یافت نشد** |
| comfort_radius | پیش‌فرض آداپتر = 1.0 | `rsr/environment/crowdnav_env.py` | در خروجی ران **یافت نشد** |
| robot.visible | پیش‌فرض فعلی **False** (انسان‌ها ربات را نادیده می‌گیرند؛ کمپین فعال v5) | `crowdnav_env.py` + `environment.yaml` | v4 با visible=True آموزش دیده بود |
| episode max_steps | در آداپتر از config؛ پیش‌فرض YAML ۲D برابر 200 | `configs/environment.yaml` + آداپتر | مقدار دقیق per-run در CSV **یافت نشد** |

### ۲.۲ شناسایی دو ران

| برچسب گزارش | مسیر آزمایش | timesteps | دیتاست | surrogate |
|---|---|---:|---|---|
| **50k** | <code dir="ltr">experiments/crowdnav_v1_50k</code> | 50000 | 216 ردیف / 72 کاندید | موجود |
| **150k** | <code dir="ltr">experiments/crowdnav_v4_full_150k</code> | 150000 | 216 ردیف / 72 کاندید | موجود |

سایر درخت‌ها:
- <code dir="ltr">crowdnav_v2_150k</code>: فقط candidates — دیتاست/surrogate **نیست**
- <code dir="ltr">crowdnav_v3_my_trial</code>: آزمایش ناقص
- <code dir="ltr">crowdnav_smoke</code>: دمو

### ۲.۳ خانواده‌ها (دیتاست واقعی)

**50k (legacy ۱۲ خانواده × ۶ = ۷۲):**  
`balanced`, `distance_dense`, `efficiency`, `energy_aware`, `near_zero`, `nonlinear_progress`, `potential_shaping`, `progress_heavy`, `risk_sensitive`, `safety_heavy`, `sparse_goal`, `weak_unbalanced`

**150k (CrowdNav-aware ۱۲ خانواده × ۶ = ۷۲):**  
`balanced`, `crowded_progress`, `efficiency`, `energy_aware`, `human_clearance`, `nonlinear_progress`, `progress_dense`, `progress_heavy`, `risk_sensitive`, `safety_moderate`, `time_pressure`, `weak_contrast`

فقط ۶ نام خانواده مشترک‌اند. **پول پاداش یکی نیست.**

### ۲.۴ اجزای پاداش

هر دو از `WeightedRewardFunction` استفاده می‌کنند. در 150k ستون‌های وزن انسانی هم در دیتاست هست (`weight_human_proximity`, `weight_human_intrusion`, `weight_crowded_progress`). در دیتاست 50k این سه ستون **وجود ندارد** (فقط ۱۱ وزن کلاسیک).

### ۲.۵ برچسب‌ها

اهداف موجود در `features_targets.csv` هر دو ران: `success_rate`, `collision_rate`, `timeout_rate`, `mean_goal_time`, `mean_path_length`.  
هدف اولیهٔ surrogate در هر دو: **`success_rate`** (تنها target در model_comparison).

دلیل انتخاب: collision در CrowdNav تقریباً صفر است و timeout مکمل success است؛ success مستقیم‌ترین معیار رسیدن به هدف است.

#### توزیع success_rate (میانگین ۳ seed)

| آمار | 50k | 150k | منبع |
|---|---:|---:|---|
| n کاندید | 72 | 72 | دیتاست |
| mean | 0.100 | 0.435 | seed-mean |
| median | 0.058 | 0.472 | |
| std | 0.132 | 0.207 | |
| min | 0.000 | 0.000 | |
| max | 0.508 | 0.808 | |
| mean collision | 0.0017 | 0.0028 | |
| mean timeout | 0.898 | 0.562 | |
| سهم کاندید با SR&lt;0.05 | 0.486 | 0.042 | |

#### میانگین success به‌ازای خانواده

**50k**

| خانواده | mean | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| progress_heavy | 0.440 | 0.058 | 0.358 | 0.508 | 6 |
| nonlinear_progress | 0.196 | 0.105 | 0.088 | 0.367 | 6 |
| balanced | 0.176 | 0.119 | 0.029 | 0.319 | 6 |
| risk_sensitive | 0.106 | 0.051 | 0.067 | 0.194 | 6 |
| efficiency | 0.090 | 0.037 | 0.046 | 0.144 | 6 |
| distance_dense | 0.069 | 0.025 | 0.033 | 0.104 | 6 |
| energy_aware | 0.068 | 0.079 | 0.000 | 0.206 | 6 |
| sparse_goal | 0.024 | 0.019 | 0.013 | 0.060 | 6 |
| safety_heavy | 0.016 | 0.010 | 0.006 | 0.029 | 6 |
| near_zero | 0.015 | 0.027 | 0.000 | 0.069 | 6 |
| potential_shaping | 0.001 | 0.002 | 0.000 | 0.004 | 6 |
| weak_unbalanced | 0.000 | 0.001 | 0.000 | 0.002 | 6 |

منبع: `docs/crowdnav_report/labels_50k_by_family.csv`

**150k** (منبع: `docs/crowdnav_report/labels_150k_by_family.csv`)

| خانواده | mean | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| progress_heavy | 0.702 | 0.074 | 0.598 | 0.765 | 6 |
| progress_dense | 0.647 | 0.120 | 0.463 | 0.808 | 6 |
| time_pressure | 0.615 | 0.085 | 0.475 | 0.692 | 6 |
| crowded_progress | 0.567 | 0.048 | 0.498 | 0.617 | 6 |
| balanced | 0.500 | 0.119 | 0.319 | 0.688 | 6 |
| human_clearance | 0.434 | 0.134 | 0.310 | 0.663 | 6 |
| nonlinear_progress | 0.379 | 0.109 | 0.225 | 0.513 | 6 |
| safety_moderate | 0.375 | 0.184 | 0.188 | 0.633 | 6 |
| efficiency | 0.313 | 0.124 | 0.133 | 0.438 | 6 |
| risk_sensitive | 0.298 | 0.149 | 0.125 | 0.531 | 6 |
| energy_aware | 0.201 | 0.185 | 0.000 | 0.481 | 6 |
| weak_contrast | 0.190 | 0.190 | 0.000 | 0.504 | 6 |

خانواده‌های تقریباً مرده در 50k: `potential_shaping`, `weak_unbalanced`, و بخش زیادی از `near_zero`/`safety_heavy`/`sparse_goal` (SR بسیار پایین). در 150k برچسب‌ها پهن‌تر و قابل‌رتبه‌بندی‌ترند.

#### نویز seed

| | 50k | 150k |
|---|---:|---:|
| میانگین std بین ۳ seed | 0.029 | 0.067 |
| میانه std | 0.017 | 0.059 |
| سهم std&gt;0.05 | 0.222 | 0.583 |
| سهم std&gt;0.10 | 0.042 | 0.194 |

در 150k نویز seed **بزرگ‌تر** است (برچسب‌ها از کف timeout بلند شده‌اند و واریانس واقعی ظاهر شده). این به معنی «بدتر بودن 150k» نیست؛ در 50k بسیاری از کاندیدها نزدیک صفر قفل شده بودند.

#### کیفیت داده

| مورد | 50k | 150k |
|---|---|---|
| NaN در success_rate | مشاهده نشد در seed-agg | مشاهده نشد |
| failed jobs | 0 از 324 در phase1_summary | 0 از 216 |
| outlier seeds | 0 | 0 |
| n_eval | 160 ثابت | 160 ثابت |
| candidates.json با دیتاست | **ناسازگار** | سازگار |
| وزن‌های مشترک با ران دیگر | 0 از 36 ID مشترک یکسان نیستند | — |

---

## ۳. تاریخچهٔ آزمایش‌ها (زمانی)

1. **Smoke / آداپتر CrowdNav++** — اتصال محیط و ریوارد پروژه.  
2. **v1 @ 50k** — آموزش روی پول legacy (در عمل 108 جاب در `runs/`، دیتاست فیلترشدهٔ 72). میانگین SR≈0.10؛ timeout غالب. Surrogate ExtraTrees ثبت شد.  
3. **بازنویسی `candidates.json` داخل همان درخت v1** با پول CrowdNav-aware — باعث ناسازگاری آرشیو شد (این ممیزی آن را صریح ثبت می‌کند).  
4. **v2_150k** — فقط تولید کاندید؛ بدون تکمیل PPO.  
5. **v3 trial** — آزمایش محدود.  
6. **v4_full_150k** — کمپین کامل 72×3 با PPO 150k، early 5k/12k/25k، surrogate، selection sim، dataset-size curve، case study `time_pressure`.

---

## ۴. نتایج ران 50k

### ۴.۱ هزینهٔ PPO

| کمیت | مقدار | منبع |
|---|---:|---|
| timesteps | 50000 | phase1_manifest / status |
| early checkpoints | 2000, 5000, 12000 | status.json |
| جاب phase1 | 324 (108×3) | phase1_manifest |
| ردیف دیتاست surrogate | 216 (72×3) | dataset |
| failed | 0 | phase1_summary |
| مجموع train همهٔ 324 جاب | 9.94 CPU-h | phase1_summary |
| مجموع train دیتاست 72 | 6.59 CPU-h | dataset |
| duration_sec لاگ‌شده | 1316 s ≈ 0.37 h | فقط batch آخر resume — **دیوارزمان کامل نیست** |
| workers | 6 | manifest |

### ۴.۲ Surrogate (تنها تنظیم موجود)

تنها مدل×فیچر ثبت‌شده: **ExtraTrees / `early+math+behavioral` / success_rate / LOFO / 3 repeats / bootstrap families n=2000**.

| متریک | مقدار | منبع |
|---|---:|---|
| pooled Spearman | 0.860 | `results/crowdnav_v1_50k/leaderboard.csv:spearman` |
| within Spearman mean | 0.533 | `...:within_family_spearman_mean` |
| within Spearman std | 0.578 | `...:within_family_spearman_std` |
| Top-3 hit | 0.833 | `...:within_family_top3_hit` |
| within pairwise | 0.769 | |
| MAE | 0.037 | |
| R² | 0.795 | |
| top10 overlap | 0.900 | |
| top10 regret | 0.009 | |
| boot pooled CI | [0.672, 0.940] | |
| boot within CI | [0.292, 0.822] | |
| needs_ppo_steps (گزارش‌شده) | 12000 | early cost alias |

تکرارها (از `metrics_summary.csv`): within ≈ 0.539 / 0.529 / 0.531 — پایدار روی میانگین، ولی std بین‌خانواده بالاست.

#### within به‌ازای خانواده (میانگین ۳ تکرار)

| خانواده | within Spearman | تفسیر |
|---|---:|---|
| sparse_goal | 0.981 | خوب (ولی برچسب‌ها نزدیک صفر) |
| energy_aware | 0.971 | خوب |
| balanced | 0.943 | خوب |
| risk_sensitive | 0.928 | خوب |
| nonlinear_progress | 0.924 | خوب |
| efficiency | 0.870 | خوب |
| near_zero | 0.850 | خوب روی بازهٔ کوچک |
| safety_heavy | 0.628 | متوسط |
| weak_unbalanced | 0.581 | متوسط/محدود |
| distance_dense | −0.181 | بد |
| potential_shaping | −0.463 | بد + تقریباً flat |
| progress_heavy | −0.638 | بد (بهترین خانواده از نظر SR!) |

منبع: `docs/crowdnav_report/within_by_family_50k.csv`

**نکته:** Top-3 hit کل 0.833 است، اما برای `progress_heavy` و `potential_shaping` از OOF برابر 0 است. یعنی میانگین Top-3 موفقیت خانواده‌های آسان را می‌پوشاند و شکست روی بهترین خانواده را پنهان می‌کند.

### ۴.۳ نشت اطلاعات (leakage)

| ریسک | وضعیت در این ران |
|---|---|
| family one-hot | در فیچرهای استفاده‌شده دیده نشد؛ LOFO روی `family_id` |
| برچسب در فیچر | early features از warmup PPO همان کاندید می‌آیند (هزینه‌دار اما مجاز در پروتکل) — نه از eval نهایی 160ep |
| random split به‌جای LOFO | خیر؛ outer=LOFO |
| دیدن خانوادهٔ تست در train | خیر (LOFO) |

**یافت نشد:** شواهد leakage ممنوع. early features عمداً بخشی از هزینهٔ پیش‌بینی‌اند.

### ۴.۴ Selection sim

برای 50k در مخزن **یافت نشد** (برخلاف 150k).

---

## ۵. نتایج ران 150k

### ۵.۱ هزینهٔ PPO

| کمیت | مقدار | منبع |
|---|---:|---|
| timesteps | 150000 | manifest / status |
| early checkpoints | 5000, 12000, 25000 | status.json |
| جاب | 216 (72×3) | phase1 |
| failed | 0 | |
| wall duration_sec | 22283.5 ≈ **6.19 h** | phase1_summary |
| مجموع train دیتاست | 22.39 CPU-h | dataset |
| workers | 4 | manifest |

### ۵.۲ Surrogate

همان تنظیم: ExtraTrees / early+math+behavioral / LOFO / 3 repeats.

| متریک | مقدار | منبع |
|---|---:|---|
| pooled Spearman | 0.653 | `results/crowdnav_v4_full_150k/leaderboard.csv` |
| within Spearman mean | 0.746 | |
| within Spearman std | 0.234 | |
| Top-3 hit | 0.917 | |
| within pairwise | 0.823 | |
| MAE | 0.124 | |
| R² | 0.402 | |
| top10 overlap | 0.333 | |
| top10 regret | 0.086 | |
| boot pooled CI | [0.377, 0.856] | |
| boot within CI | [0.654, 0.851] | |
| needs_ppo_steps (گزارش‌شده) | 12000 | **توجه:** فیچرها شامل t25000 هم هستند؛ هزینهٔ واقعی early تا 25k است |

#### within به‌ازای خانواده

| خانواده | within | Top-3 (از OOF) |
|---|---:|---:|
| energy_aware | 1.000 | 1.0 |
| nonlinear_progress | 0.943 | 1.0 |
| progress_dense | 0.943 | 1.0 |
| efficiency | 0.905 | 1.0 |
| progress_heavy | 0.867 | 1.0 |
| crowded_progress | 0.790 | 1.0 |
| weak_contrast | 0.771 | 1.0 |
| risk_sensitive | 0.754 | 1.0 |
| safety_moderate | 0.714 | 1.0 |
| human_clearance | 0.657 | 1.0 |
| balanced | 0.486 | 1.0 |
| **time_pressure** | **0.124** | **0.0** |

منبع within: `within_by_family_150k.csv`؛ Top-3 خانواده از OOF محاسبه‌شده در ممیزی.

### ۵.۳ Selection simulation (موجود فقط برای 150k)

منبع: `results/crowdnav_v4_full_150k/selection_sim/regret_vs_budget_mean_over_repeats.csv`

| k | oracle best | surrogate best | random best | regret vs oracle | P(oracle#1 in sur top-k) |
|--:|---:|---:|---:|---:|---:|
| 1 | 0.808 | 0.606 | 0.429 | 0.202 | 0.00 |
| 5 | 0.808 | 0.641 | 0.663 | 0.167 | 0.00 |
| 10 | 0.808 | 0.765 | 0.721 | 0.044 | 0.00 |
| 20 | 0.808 | 0.794 | 0.758 | 0.015 | 0.67 |
| 30 | 0.808 | 0.808 | 0.776 | 0.000 | 1.00 |

نتیجهٔ عملی: shortlist حدود **k=10** از random بهتر است؛ برای نزدیک‌شدن به oracle حدود **k=20–30**.

---

## ۶. مقایسهٔ 50k در برابر 150k

### ۶.۱ آیا 50k پروکسی لیبل 150k است؟

| بررسی | نتیجه |
|---|---|
| تعداد `candidate_id` مشترک | 36 از 72 |
| هم‌نامی خانواده روی مشترک‌ها | 100٪ |
| وزن‌های یکسان روی مشترک‌ها | **0 / 36** (از ستون‌های `weight_*` دیتاست) |
| Spearman لیبل روی 36 ID مشترک | 0.547 (p≈0.00055) |

**نتیجه‌گیری:** نمی‌توان ادعا کرد «لیبل 50k همان کاندید را در 150k پیش‌بینی می‌کند». IDهای مشترک **پاداش‌های متفاوت**اند. Spearman 0.55 فقط شباهت درشت نام/خانواده را نشان می‌دهد، نه ثبات رتبهٔ یک تابع پاداش ثابت.

انتقال surrogate 50k→برچسب 150k روی پاداش یکسان: **قابل انجام از فایل‌های موجود نیست** (پاداش یکسان وجود ندارد).

### ۶.۲ بهترین‌ها

| | 50k | 150k |
|---|---|---|
| بهترین خانواده (mean SR) | progress_heavy (0.440) | progress_heavy (0.702) |
| بهترین کاندید | progress_heavy_01 (0.508) | progress_dense_03 (0.808) |

`progress_heavy` در هر دو از خانواده‌های قوی است، ولی عضو برتر سراسری یکی نیست و پول‌ها فرق دارند.

### ۶.۳ کیفیت surrogate

| متریک | 50k | 150k | جهت |
|---|---:|---:|---|
| pooled Spearman | 0.860 | 0.653 | ↓ (کمتر متورم؟) |
| within Spearman | 0.533 | 0.746 | ↑ مهم |
| within std | 0.578 | 0.234 | ↑ پایدارتر |
| Top-3 | 0.833 | 0.917 | ↑ |
| MAE | 0.037 | 0.124 | ↓ (مقیاس برچسب عوض شده) |
| boot within lo | 0.292 | 0.654 | ↑ قطعیت |

تفسیر صادقانه: در 50k pooled بالا و within متوسط/ناپایدار است — الگوی کلاسیک «جدا کردن خانواده‌های مرده از زنده». در 150k within بهتر و CI within خیلی سالم‌تر است؛ یعنی رتبه‌بندی داخل خانواده برای triage واقعی قابل‌اعتمادتر شده.

مدل/فیچر برنده در هر دو یکسان است — ولی چون **فقط همین یک تنظیم** اجرا شده، پایداری «کدام مدل می‌برد» بین جایگزین‌ها **آزموده نشده** (not found: grid چندمدلی CrowdNav).

### ۶.۴ هزینه در برابر فایده

| | 50k (دیتاست 72) | 150k (72) |
|---|---:|---:|
| CPU-h آموزش (جمع ردیف‌ها) | 6.59 | 22.39 |
| Wall ثبت‌شده | ناکامل (resume) | 6.19 h |
| mean SR | 0.100 | 0.435 |
| within Spearman | 0.533 | 0.746 |

تقریباً **۳.۴×** CPU train بیشتر برای 150k روی همان تعداد کاندید دیتاست، در ازای برچسب‌های غیرتبهگن و within قوی‌تر. برای ادعای علمی CrowdNav، 150k ارزشش را دارد؛ 50k بیشتر پایلوت منفی/محدود است.

---

## ۷. مقایسه با محیط ۲D

مرجع ۲D (تأیید نهایی): ExtraTrees / early+math+behavioral — pooled 0.935، within 0.648، Top-3 0.833، MAE 0.067  
منبع: `experiments/model_comparison_confirm/metrics_summary.csv` و `docs/report_2d_experiments.md`

| یافته | ۲D | CrowdNav 50k | CrowdNav 150k |
|---|---|---|---|
| pooled ≫ within؟ | بله (0.935 vs 0.648) | بله (0.860 vs 0.533) | **معکوس** (0.653 vs 0.746) |
| early+behavioral مفید برای within | بله | فقط همین ست اجرا شده | فقط همین ست |
| Top-3 حدود 0.83+ | 0.833 | 0.833 | 0.917 |
| خانوادهٔ شکست‌خوردهٔ تک‌نقطه‌ای | ناهمگن | progress_heavy within منفی | time_pressure |
| بودجهٔ PPO عملیاتی | 50k کافی برای ۲D | برای CrowdNav ناکافی | 150k عملیاتی |

**تکرار می‌شود:** ضرورت گزارش دوگانهٔ pooled/within؛ ExtraTrees+early به‌عنوان نقطهٔ عملیاتی؛ شکست ناهمگن بین خانواده‌ها.  
**تکرار نمی‌شود:** در CrowdNav@150k within از pooled بالاتر است — چون خانواده‌های مردهٔ legacy حذف/جایگزین شده‌اند و برچسب‌ها پهن‌ترند.

---

## ۸. تصمیم عملیاتی و توصیه

**برای CrowdNav، نقطهٔ عملیاتی پیشنهادی:**
- بودجهٔ PPO برچسب: **150k** (نه 50k)
- Surrogate: **ExtraTrees + early+math+behavioral + LOFO**
- هدف: `success_rate`
- استفاده: **shortlist** (حدود k=10 تا 30)، نه جایگزینی کامل PPO
- هشدار خانواده: `time_pressure` را جدا گزارش کنید

**توصیه‌ها**
1. درخت v1 را به‌عنوان آرشیو قفل کنید؛ به `candidates.json` آن اعتماد نکنید.  
2. مقایسهٔ 50k↔150k را به‌عنوان «همان کاندید» ادعا نکنید.  
3. اگر ادعا می‌کنید early تا 12k هزینه دارد، یا فیچر t25000 را از ست عملیاتی جدا کنید یا `needs_ppo_steps` را 25000 کنید.  
4. برای قدرت آماری: n_families=12 — CI هنوز پهن است؛ ادعای فرق مدل‌ها بدون تکرار بیشتر ضعیف است.

---

## ۹. محدودیت‌ها، ریسک‌ها، چیزهایی که باید قوی‌تر شوند

### محدودیت‌ها
- فقط یک مدل و یک feature set برای CrowdNav اجرا شده (برخلاف گرید ۲D).  
- n=12 خانواده → bootstrap CI وسیع روی pooled.  
- 50k و 150k پول پاداش یکسان ندارند → مقایسهٔ علیّ بودجه محدود است.  
- human_num / max_steps دقیق per-run در CSV ثبت نشده.  
- Selection sim فقط برای 150k.  
- Wall کامل 50k لاگ نشده.

### ریسک ادعا
- گفتن «surrogate CrowdNav Spearman 0.86» بدون گفتن within و بدون گفتن مرگ برچسب‌ها در 50k — گمراه‌کننده است.  
- گفتن «150k بدتر است چون pooled افت کرد» — غلط؛ within بهتر شده.

### برای تقویت ادعا (نیاز به کار جدید؛ خارج از این ممیزی)
- یک کمپین با **همان** `candidates.json` در دو بودجه (50k و 150k).  
- گرید مدل/فیچر روی v4.  
- ثبت صریح `human_num` و `max_steps` در dataset/status.

### سطح اطمینان ادعاها
| ادعا | سطح |
|---|---|
| 150k برچسب سالم‌تر از 50k (mean SR، کمتر tabeگن) | (a) چند منبع سازگار |
| ExtraTrees@150k within≈0.75 و Top-3≈0.92 | (a) ۳ تکرار + bootstrap |
| 50k candidates.json معتبر است | **رد شده** — ناسازگار |
| 50k پروکسی رتبهٔ 150k روی همان پاداش | **رد / غیرقابل‌تأیید** |
| کدام مدل جایگزین بهتر است در CrowdNav | (c) آزموده نشده |

---

## پیوست: شکل‌ها

تولیدشده در <code dir="ltr">docs/crowdnav_report/figures/</code>:
- `01_per_family_mean_sr.png`
- `02_label_histograms.png`
- `03_50k_vs_150k_scatter_shared_ids.png` (با هشدار پول متفاوت)
- `04_pooled_vs_within_50k_150k.png`
- `05_oof_pred_vs_true_50k.png` / `05_oof_pred_vs_true_150k.png`
- `06_selection_sim_150k.png`
- `07_cost_vs_within_spearman.png`
- `08_seed_noise_hist.png`

قابل‌استفادهٔ موجود در مخزن:
- <code dir="ltr">results/crowdnav_v4_full_150k/figures/*.png</code>
- <code dir="ltr">results/crowdnav_v1_50k/figures/*.html</code> (تعاملی؛ برای اسلاید PNGهای بالا مناسب‌ترند)

</div>
