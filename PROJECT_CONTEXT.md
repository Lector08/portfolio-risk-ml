# КОНТЕКСТ ПРОЕКТУ — Система управління ризиками портфеля на основі ML
# Курсова робота, 4 курс, спеціальність 113 Прикладна математика, 2026
# Відправ цей файл будь-якій LLM — вона отримає повний контекст проекту

---

## 1. СУТЬ І МЕТА ПРОЕКТУ

**Що це:** Комп'ютерна система управління ризиками інвестиційного портфеля на основі ML. Python.

**Мета:**
1. Прогнозувати волатильність акцій (XGBoost/LSTM)
2. Розраховувати VaR/CVaR кількома методами
3. Оптимізувати портфель за Марковіцем з EWMA-матрицею
4. Верифікувати VaR тестом Купєця
5. Порівнювати стратегії через walk-forward бектестинг
6. Надавати Streamlit веб-інтерфейс

**Мова звіту:** Українська. **Код:** Python 3.13.
**Шлях проекту:** /Users/user/Documents/Coursework_2026/

---

## 2. СТРУКТУРА ФАЙЛІВ

```
code/
├── config.py                    # END_DATE = date.today()
├── main.py                      # --step data|volatility|risk|portfolio|backtest|plots
├── app.py                       # Streamlit: python3 -m streamlit run app.py
├── train_sp500.py               # Навчання загальної моделі (~30-40 хв)
├── data/
│   ├── loader.py                # yfinance + кеш
│   ├── preprocessor.py          # лог-доходності, split 80/10/10
│   ├── feature_engineering.py  # 152 ознаки для main.py
│   ├── weighting.py             # EWMA-ваги, ewma_cov_matrix, BRW VaR
│   ├── sector_features.py       # 6 динамічних секторних ознак через ETF
│   └── ticker_info.py           # Автовизначення сектора (3-рівневий кеш)
├── models/
│   ├── volatility/xgboost_model.py, lstm_model.py, ensemble.py
│   ├── risk/var_cvar.py, monte_carlo.py
│   └── portfolio/markowitz.py, ml_optimizer.py
├── backtesting/backtest.py, metrics.py
└── models/saved/
    ├── xgb_AAPL_v2.json  ... xgb_SCHD_v2.json  # 8 per-ticker моделей, 19 ознак
    ├── xgb_sp500.json                             # загальна S&P500 модель, 26 ознак
    ├── sp500_meta.json                            # n_instruments, n_features, metrics
    └── ticker_sectors.json                        # кеш секторів тікерів

docs/report/
├── chapter2_theory.md       # VaR, EWMA, Марковіц, XGBoost, LSTM, всі метрики
├── chapter3_implementation.md  # 3.1-3.11, описи 13 функцій
└── chapter4_results.md      # результати, баги, висновки
```

---

## 3. КЛЮЧОВІ АЛГОРИТМИ

### Цільова змінна (однакова в main.py і app.py)
```
r = log(P_t / P_{t-1})                    # лог-доходності
RV = sqrt(sum(r_i^2, 20d)) * sqrt(252)    # реалізована волатильність
y  = log1p(RV)                             # цільова змінна (log1p для стабільності)
```

### 19 технічних ознак (per-ticker режим, app.py)
ma5_ratio, ma20_ratio, rsi14, pct_b, bandwidth,
std5, skew5, kurt5, std10, skew10, kurt10, std20, skew20, kurt20,
lag1, lag2, lag3, lag5, lag10

КЛЮЧОВА ВЛАСТИВІСТЬ: 19 ознак незалежно від складу портфеля.
xgb_AAPL_v2.json працює коректно при будь-якому складі.

### 6 секторних ознак (S&P500 режим, через ETF-proxy XLK/XLF/...)
sector_ret_5d, sector_ret_20d, sector_vol_20d,
rel_to_sector, sector_trend, sector_breadth + is_etf → разом 26 ознак

ЧОМУ НЕ ONE-HOT: sector_tech=1 — константа, важливість <1%.
sector_ret_5d=-0.023 — змінюється кожен день, реальна інформація.

### EWMA (λ=0.94, RiskMetrics JP Morgan 1994)
w_t = (1-λ) * λ^(T-1-t),  ESS = (1+λ)/(1-λ) ≈ 32 дні "пам'яті"

### BRW VaR (Boudoukh-Richardson-Whitelaw 1998)
Сортуємо доходності від найгіршої, накопичуємо EWMA-ваги,
VaR = рівень де накопичена вага >= (1-α)

### Оптимізація Марковіца
min  w^T Σ w
s.t. w^T μ >= r_min   (InfeasibleConstraintError якщо недосяжно)
     1^T w = 1
     lb_i <= w_i <= ub_i

Перевірка r_min: LP через cvxpy/scipy.linprog ДО оптимізації.
Якщо недосяжно → UI показує держоблігації (separation theorem).

### Тест Купєця (1995)
LR = -2[x*ln(p0/p_hat) + (T-x)*ln((1-p0)/(1-p_hat))] ~ χ²(1)
H0: p_hat = p0 = 1-α = 0.05
Відхиляється якщо LR > 3.841 (χ²_0.95(1))

---

## 4. РЕАЛЬНІ ЧИСЛОВІ РЕЗУЛЬТАТИ

### ML моделі (AAPL, тестова вибірка)
GARCH(1,1): ω=0.1858, α=0.1412, β=0.8196, α+β=0.961 (baseline)
LSTM:    RMSE=0.0465, MAE=0.0375, QLIKE=-2.249  (epoch 28/50)
XGBoost: RMSE=0.0150, MAE=0.0122, QLIKE=-2.447  (iter 325/500)
Ensemble: RMSE=0.0154, MAE=0.0128, QLIKE=-2.447
Ваги ансамблю: XGBoost=92.9%, LSTM=7.1% (val MSE 0.000258 vs 0.003363)

### VaR 95% (AAPL)
Historical (рівні ваги): VaR=0.0166, CVaR=0.0273
BRW EWMA (λ=0.94):       VaR=0.0108, CVaR=0.0153  (на 35% нижче)
Parametric (sample σ):   VaR=0.0186, CVaR=0.0235

### Оптимізація (тренувальні дані 2018-2022)
Equal-Weight:          12.02% / 23.78% / Sharpe=0.337
Min-Variance:          7.43%  / 19.48% / Sharpe=0.176
Max-Sharpe (Sample Σ): 26.44% / 31.21% / Sharpe=0.719
Max-Sharpe (EWMA Σ):   25.93% / 28.75% / Sharpe=0.763  ← краще

### Walk-forward бектестинг (1004 торгових дні)
ML-Markowitz:      CAGR=12.6%, Sharpe=0.446, MDD=-35.8%, Calmar=0.418, VaR%=4.4%, Купієць OK
Classic-Markowitz: CAGR=9.0%,  Sharpe=0.304, MDD=-31.0%, Calmar=0.436, VaR%=5.2%, Купієць OK
Equal-Weight:      CAGR=12.6%, Sharpe=0.446, MDD=-35.8%, Calmar=0.418, VaR%=4.4%, Купієць OK

ПРИМІТКА: ML = Equal-Weight через Баг #1 (виправлено, потрібен повторний запуск)

### S&P500 модель (xgb_sp500.json, 2.94 MB)
157 інструментів (73 акції + 84 ETF), 26 ознак, RMSE=0.019

### Збережені моделі на диску
xgb_AAPL_v2.json (756KB), xgb_AMZN_v2.json (792KB), xgb_GOOGL_v2.json (772KB),
xgb_JPM_v2.json (656KB), xgb_KO_v2.json (534KB), xgb_MSFT_v2.json (547KB),
xgb_PLTR_v2.json (912KB), xgb_SCHD_v2.json (540KB), xgb_sp500.json (2.94MB)

---

## 5. ВИПРАВЛЕНІ БАГИ

### БАГ #1 — КРИТИЧНИЙ (backtest._rebalance)
БУЛО:  ml_opt.optimize(sigma_ml, train_ret, ...)      # аргументи переставлені!
СТАЛО: ml_opt.optimize(returns=train_ret, sigma_forecasts=sigma_ml, ...)
ЕФЕКТ: ML-Markowitz завжди = Equal-Weight (AttributeError → fallback)
СТАТУС: Виправлено. Потрібен: python3 main.py --step backtest

### БАГ #2 — backtest.py
from scipy.stats import norm → перенесено на рівень модуля (було в циклі 3000 разів)

### БАГ #3 — ticker_info.py
Прибрано невикористаний import numpy as np

---

## 6. STREAMLIT ДОДАТОК (app.py)

Запуск: python3 -m streamlit run app.py

САЙДБАР:
- Кнопка Топ-10 (AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B AVGO JPM)
- Введення тікера → автовизначення сектора через ticker_info.py
- Перемикач Per-ticker (19 ознак) / S&P500 (26 ознак)
- Кнопка перенавчати per-ticker

ВКЛАДКА ОГЛЯД:
- Таблиця: тікер, сектор, ціна, Δ1д%, ML σ̂%, hist σ%, сигнал (⚠️ якщо ML > 1.2×hist)
- Зведена таблиця волатильності по секторах
- Нормовані ціни (база=100) лінійний графік

ВКЛАДКА ОПТИМІЗАЦІЯ:
1. Expander: введення поточних ваг (у %)
2. Слайдер мінімальної доходності 0-20%:
   - Недосяжно → СТОП! Блок "РИНОК НЕ ВІДПОВІДАЄ":
     таблиця T-Bills ~5%, T-Notes ~4.8%, T-Bonds ~4.5%
     таблиця EWMA-доходностей кожного активу
   - Досяжно → ✅ повідомлення, продовжуємо
3. Обмеження позицій:
   Пресети: Max25%, Max40%, Min5%, Без обмежень
   Індивідуально: checkbox min% + значення, checkbox max% + значення
   Превью: "🔒 5-25%", "🔽 min 5%", "🔼 max 25%", "✅ Вільна"
   Валідація: min > max → помилка, sum(min) > 100% → помилка
4. Кнопка Оптимізувати → таблиця порівняння + ваги Max-Sharpe + bar chart
   Статус ваг: 📌 на мін.межі, 🔒 на макс.межі, ✅ вільна

ВКЛАДКА VaR/CVaR: три методи паралельно (Hist, BRW EWMA, Parametric)

ВКЛАДКА МОДЕЛІ:
- Порівняння Per-ticker vs S&P500 (що є, RMSE)
- Таблиця per-ticker моделей (файл, дата, розмір, σ̂%)
- Джерело сектора (📚 вбудований / 💾 кеш/API)
- Реалізована волатильність rolling 20d графік

---

## 7. АВТОВИЗНАЧЕННЯ СЕКТОРА (ticker_info.py)

Рівень 1: KNOWN_SECTORS dict (~100 тікерів) — миттєво, без мережі
Рівень 2: ticker_sectors.json disk cache — миттєво
Рівень 3: yfinance.Ticker(t).info → поле sector, quoteType — 1-2 сек
Рівень 4: Fallback "Technology"

Нормалізація GICS: "Consumer Cyclical" → "Consumer Discretionary"
                    "Financial Services" → "Financials" тощо

KNOWN_SECTORS імпортується на рівні модуля в app.py:
from data.ticker_info import KNOWN_SECTORS  (НЕ через __import__ хак!)

---

## 8. СЕКТОРИ GICS + ETF-PROXY

Technology=XLK, Healthcare=XLV, Financials=XLF,
Consumer Discretionary=XLY, Consumer Staples=XLP,
Energy=XLE, Industrials=XLI, Materials=XLB,
Real Estate=XLRE, Utilities=XLU, Communication Services=XLC,
ETF → proxy SPY

---

## 9. ПАРАМЕТРИ СИСТЕМИ

λ = 0.94       (EWMA, RiskMetrics)
r_f = 0.04     (безризикова ставка, T-Bills)
α = 0.95       (рівень довіри VaR)
w = 20 днів    (вікно реалізованої волатильності)
252            (торгових днів у році)
train_window = 504 дні  (walk-forward вікно)
rebal_freq = 21 день    (частота ребалансування)
n_starts = 30           (мультистарт max_sharpe)
early_stopping = 30     (XGBoost)
optuna_trials = 50      (підбір гіперпараметрів)
batch_size = 20         (train_sp500.py batches)
delay = 3s              (між батчами, rate limiting Yahoo Finance)

---

## 10. ФОРМУЛИ (для написання коду або перевірки)

CAGR = (NAV_T/NAV_0)^(252/T) - 1
Sharpe = (mean(r)*252 - r_f) / (std(r)*sqrt(252))
Sortino = (mean(r)*252 - r_f) / (std(r[r<0])*sqrt(252))
MDD = min_t[(NAV_t - max_{s<=t}(NAV_s)) / max_{s<=t}(NAV_s)]
Calmar = CAGR / |MDD|

RSI = 100 - 100/(1 + AvgGain/AvgLoss),  нормалізація: RSI/50 - 1
%B = (P - Lower) / (Upper - Lower)
Bandwidth = (Upper - Lower) / MA20
MA_ratio = P/MA_w - 1

RMSE = sqrt(mean((sigma_hat - sigma)^2))
MAE = mean(|sigma_hat - sigma|)
QLIKE = mean(log(sigma_hat^2) + sigma^2/sigma_hat^2)

---

## 11. ДАНІ

main.py: 8 акцій S&P500 (AAPL MSFT GOOGL AMZN JPM GS XOM JNJ)
app.py:  будь-які тікери через введення або кнопку Топ-10
Діапазон: 2018 – сьогодні (END_DATE = date.today() динамічно)
Split: Train 80% (≈1208) / Val 10% (≈150) / Test 10% (≈150), хронологічний

---

## 12. КОМАНДИ

# Повний пайплайн (покроково)
python3 main.py --step data
python3 main.py --step volatility
python3 main.py --step risk
python3 main.py --step portfolio
python3 main.py --step backtest   # ← після виправлення Багу #1
python3 main.py --step plots

# Навчання загальної S&P500 моделі (один раз, ~30-40 хв)
python3 train_sp500.py

# Веб-інтерфейс
python3 -m streamlit run app.py

---

## 13. ПРАВИЛА ЗВІТУ

Мова: УКРАЇНСЬКА
Формат: Markdown (.md), LaTeX у $$ $$
Числа: реальні з запуску (не абстрактні)
Код: тільки ключові фрагменти
Формули: пояснювати кожен символ + посилання (автор, рік)
Таблиці: заповнювати реальними числами (не залишати "—")
Обмеження: описувати чесно

Структура Розділу 3:
3.1 Архітектура | 3.2 Дані (M1) | 3.3 Features (два підходи) |
3.4 Моделі (M2, per-ticker vs S&P500) | 3.5 VaR (M3) |
3.6 Оптимізація (M4, lb/ub/r_min) | 3.7 Бектестинг (M5) |
3.8 Секторні ознаки | 3.9 ticker_info | 3.10 app.py |
3.11 Описи 13 ключових функцій (сигнатура + алгоритм + реальний результат)

Теорія розділу 2 вже включає:
VaR/CVaR, EWMA/BRW, Марковіц+обмеження+separation theorem, GARCH,
XGBoost, LSTM, ансамбль, секторні ознаки, CAGR, Sharpe, Sortino,
MDD, Calmar, ануалізація, тест Купєця, RSI, Bollinger, MA, RMSE, MAE, QLIKE

---

## 14. ВАЖЛИВІ НЮАНСИ ДЛЯ LLM

1. ML-Markowitz = Equal-Weight у поточних результатах через Баг #1 (ВІДОМО, задокументовано)
2. Per-ticker КРАЩЕ за якістю (RMSE 0.015 vs 0.019) але не бачить сектор
3. S&P500 режим бачить сектор через 6 ЧИСЛОВИХ ознак, не one-hot
4. One-hot сектор марний — важливість <1%, XGBoost ігнорує
5. Scaler навчається ТІЛЬКИ на train — немає data leakage
6. user_w, total_uw ІНІЦІАЛІЗУЮТЬСЯ ДО st.tabs() — scope bug інакше
7. KNOWN_SECTORS — модульний імпорт (не __import__ хак)
8. train_sp500.py: batch=20, delay=3s, exponential backoff, max_retries=3
9. InfeasibleConstraintError → показуємо держоблігації (реалізація separation theorem)
10. Після виправлення Багу #1 → перезапустити python3 main.py --step backtest
