<div dir="rtl" lang="fa">

# خلاصهٔ یک‌صفحه‌ای CrowdNav (برای ارائه)

## الف) ۱۰–۱۵ عدد کلیدی

| # | عدد | معنی | منبع |
|--:|---|---|---|
| 1 | **0.100** | میانگین success_rate در 50k | `experiments/crowdnav_v1_50k/dataset/...` seed-mean |
| 2 | **0.435** | میانگین success_rate در 150k | `experiments/crowdnav_v4_full_150k/dataset/...` |
| 3 | **0.898 → 0.562** | میانگین timeout_rate از 50k به 150k | همان دیتاست‌ها |
| 4 | **0.860** | pooled Spearman @50k | `results/crowdnav_v1_50k/leaderboard.csv` |
| 5 | **0.533** | within Spearman @50k | همان |
| 6 | **0.653** | pooled Spearman @150k | `results/crowdnav_v4_full_150k/leaderboard.csv` |
| 7 | **0.746** | within Spearman @150k | همان |
| 8 | **0.917** | Top-3 hit @150k | همان |
| 9 | **0.124** | within فقط برای `time_pressure` @150k | `within_by_family_150k.csv` |
| 10 | **6.19 h** | wall Phase-1 @150k | `phase1_summary.json:duration_sec` |
| 11 | **6.59 → 22.39 CPU-h** | جمع train دیتاست 72تایی 50k→150k | sum `train_duration_sec` |
| 12 | **0 / 36** | کاندید مشترک با وزن یکسان بین دو ران | مقایسهٔ `weight_*` دیتاست |
| 13 | **k=10 → regret 0.044** | selection sim 150k vs oracle | `selection_sim/regret_vs_budget_mean_over_repeats.csv` |
| 14 | **0.935 / 0.648 / 0.833** | ۲D confirm pooled / within / Top-3 | `experiments/model_comparison_confirm/metrics_summary.csv` |
| 15 | **candidates.json ≠ dataset در v1** | ناسازگاری آرشیو 50k | `discovery_integrity.json` |

## ب) ۵ پیام کلیدی

1. **50k برای CrowdNav برچسب تبهگن می‌سازد** (mean SR≈0.10، timeout≈0.90)؛ برای رتبه‌بندی پاداش کافی نیست.  
2. **150k برچسب را زنده می‌کند** (mean SR≈0.44) و **within Spearman را از 0.53 به 0.75** می‌رساند — این معیار تصمیم است، نه pooled.  
3. **دو ران پول پاداش یکسان ندارند**؛ 0 کاندید با وزن یکسان → نمی‌توان 50k را پروکسی 150k دانست.  
4. **نقطهٔ عملیاتی CrowdNav:** ExtraTrees + early+math+behavioral + LOFO روی برچسب 150k؛ استفاده به‌صورت shortlist (k≈10–30)، با پاورقی `time_pressure`.  
5. **یافتهٔ ۲D تکرار می‌شود که pooled گمراه‌کننده است**؛ در 150k حتی within از pooled بالاتر است.

## ج) Safe to claim / Not safe to claim

| Safe to claim | Not safe to claim |
|---|---|
| CrowdNav@150k within≈0.75 و Top-3≈0.92 با ۳ تکرار ExtraTrees | «Spearman کلی CrowdNav 0.86 پس مدل عالی است» (50k؛ within ضعیف‌تر و برچسب مرده) |
| 50k timeout-dominated است و برای این محیط بودجهٔ ناکافی بوده | «همان 72 پاداش را در 50k و 150k مقایسه کردیم» |
| Surrogate برای shortlist مفید است (selection k=10 بهتر از random) | «سورگیت بهترین کاندید را با k=1 پیدا می‌کند» |
| `time_pressure` تنها خانواده با Top-3=0 در 150k است | «همهٔ خانواده‌ها خوب رتبه می‌شوند» |
| درخت v1 از نظر candidates.json خراب/بازنویسی‌شده است | اتکا به `crowdnav_v1_50k/candidates.json` به‌عنوان پول واقعی |
| مقایسه با ۲D: ضرورت within vs pooled | «CrowdNav از ۲D بهتر/بدتر است» بدون جدا کردن pooled و within و بدون ذکر تفاوت پول |

## د) بهترین شکل‌ها برای اسلاید

| فایل | محتوا | عنوان پیشنهادی اسلاید |
|---|---|---|
| `docs/crowdnav_report/figures/02_label_histograms.png` | توزیع SR در 50k و 150k | «150k برچسب را از کف timeout بلند می‌کند» |
| `docs/crowdnav_report/figures/04_pooled_vs_within_50k_150k.png` | pooled در برابر within | «معیار درست: within، نه pooled» |
| `docs/crowdnav_report/figures/01_per_family_mean_sr.png` | میانگین خانواده | «خانواده‌های مرده در 50k؛ طیف زنده در 150k» |
| `docs/crowdnav_report/figures/05_oof_pred_vs_true_150k.png` | OOF 150k | «ExtraTrees روی CrowdNav@150k» |
| `docs/crowdnav_report/figures/06_selection_sim_150k.png` | oracle / sur / random | «Shortlist نه oracle یک‌شات» |
| `docs/crowdnav_report/figures/03_50k_vs_150k_scatter_shared_ids.png` | پراکنش IDهای مشترک | «هشدار: ID مشترک ≠ پاداش یکسان» |
| `docs/crowdnav_report/figures/07_cost_vs_within_spearman.png` | هزینه در برابر within | «۳× CPU بیشتر، within قابل‌اتکاتر» |
| `results/crowdnav_v4_full_150k/figures/within_family_spearman_ci.png` | CI موجود | «عدم‌قطعیت within در 150k» |

</div>
