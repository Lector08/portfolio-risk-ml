# =============================================================================
# main.py — Точка входу: повний пайплайн системи управління ризиками
# =============================================================================

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TICKERS, START_DATE, END_DATE,
    CONFIDENCE_LEVEL, RISK_FREE_RATE, N_MONTE_CARLO,
    LSTM_SEQUENCE_LEN,
)

import numpy as np
import pandas as pd


def step_data(force_download: bool = False) -> dict:
    print("\n" + "="*60)
    print("КРОК 1: Завантаження та обробка даних")
    print("="*60)

    from data.loader import get_prices
    from data.preprocessor import (
        compute_log_returns, train_val_test_split,
        compute_realized_volatility,
    )
    from data.feature_engineering import build_feature_matrix, create_lstm_sequences
    from sklearn.preprocessing import StandardScaler

    prices  = get_prices(force_download=force_download)
    returns = compute_log_returns(prices)
    rv      = compute_realized_volatility(returns, window=20)
    train_val_test_split(returns)

    X_df = build_feature_matrix(prices, returns)
    common_idx = X_df.index.intersection(rv.index)
    X_df = X_df.loc[common_idx]
    y_df = rv.loc[common_idx]

    n       = len(X_df)
    t1, t2  = int(n * 0.8), int(n * 0.9)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float32))
    X_seq, _ = create_lstm_sequences(X_scaled, np.zeros(n), seq_len=LSTM_SEQUENCE_LEN)

    print(f"\n✅ M1 готово: {len(prices)} днів | {prices.shape[1]} активів | {X_df.shape[1]} ознак")
    return dict(
        prices=prices, returns=returns, rv=y_df,
        X_df=X_df, X_scaled=X_scaled, X_seq=X_seq,
        scaler=scaler, splits=(t1, t2),
    )


def step_volatility(data: dict) -> dict:
    print("\n" + "="*60)
    print("КРОК 2: Навчання моделей прогнозу волатильності")
    print("="*60)

    try:
        from models.volatility.xgboost_model import XGBoostVolatilityModel
    except ImportError as e:
        print(f"\n⚠️  XGBoost недоступний: {e}")
        print("   Щоб виправити на macOS: brew install libomp")
        return _volatility_fallback(data)

    from models.volatility.lstm_model import LSTMVolatilityModel
    from models.volatility.ensemble   import VolatilityEnsemble

    X_scaled = data["X_scaled"]
    X_seq    = data["X_seq"]
    rv       = data["rv"]
    t1, t2   = data["splits"]
    X_df     = data["X_df"]
    SEQ      = LSTM_SEQUENCE_LEN

    y = rv["AAPL"].reindex(X_df.index).ffill().values.astype(np.float32)

    X_tr, X_vl, X_te = X_scaled[:t1], X_scaled[t1:t2], X_scaled[t2:]
    y_tr, y_vl, y_te = y[:t1], y[t1:t2], y[t2:]

    print("\n--- XGBoost ---")
    xgb_model = XGBoostVolatilityModel()
    xgb_model.fit(X_tr, y_tr, X_vl, y_vl, feature_names=list(X_df.columns))
    xgb_metrics = xgb_model.evaluate(X_te, y_te)

    print("\n--- LSTM ---")
    X_tr_s = X_seq[:t1 - SEQ];  X_vl_s = X_seq[t1-SEQ:t2-SEQ];  X_te_s = X_seq[t2-SEQ:]
    y_tr_s = y[SEQ:t1];          y_vl_s = y[t1:t2];               y_te_s = y[t2:]

    lstm = LSTMVolatilityModel(seq_len=SEQ, n_features=X_scaled.shape[1])
    lstm.build_model()
    lstm.fit(X_tr_s, y_tr_s, X_vl_s, y_vl_s, epochs=50)
    lstm_metrics = lstm.evaluate(X_te_s, y_te_s)

    print("\n--- Ансамбль ---")
    ensemble = VolatilityEnsemble(lstm, xgb_model)
    ensemble.fit_weights(X_vl_s, X_vl, y_vl_s)
    ens_metrics = ensemble.compare_models(X_te_s, X_te, y_te_s)
    ensemble.fit_garch(data["returns"]["AAPL"].values[:t1])

    print("\n✅ M2 готово.")
    return dict(
        xgb_model=xgb_model, lstm_model=lstm, ensemble=ensemble,
        xgb_metrics=xgb_metrics, lstm_metrics=lstm_metrics, ens_metrics=ens_metrics,
        mode="ml",
    )


def _volatility_fallback(data: dict) -> dict:
    print("   Використовується rolling-std як proxy σ̂.")
    return dict(
        xgb_model=None, lstm_model=None, ensemble=None,
        sigma_proxy=data["returns"].rolling(20).std().ffill(),
        xgb_metrics={}, lstm_metrics={}, ens_metrics={},
        mode="fallback",
    )


def step_risk(data: dict, vol_data: dict) -> dict:
    print("\n" + "="*60)
    print("КРОК 3: Розрахунок ризик-метрик VaR та CVaR")
    print("="*60)

    from models.risk.var_cvar    import compute_all_var, portfolio_var_cvar
    from models.risk.monte_carlo import simulate_gbm, mc_summary

    returns = data["returns"]
    t1, t2  = data["splits"]
    test_r  = returns.iloc[t2:]

    # ML-прогноз або proxy
    if vol_data["mode"] == "ml" and vol_data["xgb_model"] is not None:
        sigma_ml = float(vol_data["xgb_model"].predict(data["X_scaled"][t2:])[-1])
    else:
        sigma_ml = float(returns["AAPL"].rolling(20).std().dropna().iloc[-1])

    print("\n--- VaR / CVaR для AAPL ---")
    var_results = compute_all_var(test_r["AAPL"].values, sigma_ml=sigma_ml,
                                  alpha=CONFIDENCE_LEVEL)

    print("\n--- VaR портфеля (рівні ваги) ---")
    n = returns.shape[1]
    w = np.ones(n) / n
    Sigma_sample = returns.iloc[:t2].cov().values

    # EWMA ковар. матриця для порівняння
    from data.weighting import ewma_cov_matrix
    Sigma_ewma = ewma_cov_matrix(returns.iloc[:t2], lambda_=0.94)

    var_p,  cvar_p  = portfolio_var_cvar(w, Sigma_sample, alpha=CONFIDENCE_LEVEL)
    var_ew, cvar_ew = portfolio_var_cvar(w, Sigma_ewma,   alpha=CONFIDENCE_LEVEL)

    print(f"Portfolio VaR  (Sample): {var_p:.4f}  |  CVaR: {cvar_p:.4f}")
    print(f"Portfolio VaR  (EWMA):   {var_ew:.4f}  |  CVaR: {cvar_ew:.4f}")

    print("\n--- Монте-Карло: 10 000 симуляцій ---")
    prices = data["prices"]
    S0     = prices.iloc[-1].values
    mu_v   = returns.iloc[:t2].mean().values
    sig_v  = returns.iloc[:t2].std().values
    corr_v = returns.iloc[:t2].corr().values
    paths  = simulate_gbm(S0, mu_v, sig_v, corr_v, T=252, n_sim=N_MONTE_CARLO)
    mc_res = mc_summary(paths, w, alpha=CONFIDENCE_LEVEL)

    print("\n✅ M3 готово.")
    return dict(
        var_results=var_results, var_p=var_p, cvar_p=cvar_p,
        mc_results=mc_res, mc_paths=paths, ew_weights=w,
    )


def step_portfolio(data: dict, vol_data: dict) -> dict:
    print("\n" + "="*60)
    print("КРОК 4: Оптимізація портфеля (Markowitz + ML + EWMA)")
    print("="*60)

    from models.portfolio.markowitz    import MarkowitzOptimizer
    from models.portfolio.ml_optimizer import MLPortfolioOptimizer

    returns = data["returns"]
    t1, t2  = data["splits"]
    train_r = returns.iloc[:t1]

    mu_a  = train_r.mean().values * 252
    Sig_a = train_r.cov().values  * 252

    # Класичний Markowitz
    opt = MarkowitzOptimizer(mu_a, Sig_a, tickers=TICKERS, risk_free_rate=RISK_FREE_RATE)
    w_mv = opt.min_variance()
    w_ms = opt.max_sharpe()
    opt.efficient_frontier(n_points=100)
    opt.compare_with_equal_weight()

    # ML-прогноз sigma (або proxy)
    if vol_data["mode"] == "ml" and vol_data["xgb_model"] is not None:
        X_vl      = data["X_scaled"][t1:t2]
        preds     = vol_data["xgb_model"].predict(X_vl)
        sigma_ml  = np.full(len(TICKERS), float(np.mean(preds)))
    else:
        sigma_ml = train_r.std().values

    # ML + EWMA оптимізатор — порядок аргументів: (returns, sigma, method, cov_method)
    ml_opt = MLPortfolioOptimizer(tickers=TICKERS, risk_free_rate=RISK_FREE_RATE,
                                   lambda_=0.94)
    print("\n--- Порівняння методів Σ ---")
    comparison = ml_opt.compare_all_methods(train_r, sigma_ml)

    # Основний результат: ML + EWMA
    ml_result = ml_opt.optimize(
        returns        = train_r,
        sigma_forecasts = sigma_ml,
        method         = "max_sharpe",
        cov_method     = "ml_ewma",
    )

    print("\n✅ M4 готово.")
    return dict(
        optimizer=opt, ml_optimizer=ml_opt,
        w_min_var=w_mv, w_max_sharpe=w_ms,
        w_ml_markowitz=ml_result["weights"],
        ml_result=ml_result,
    )


def step_backtest(data: dict) -> dict:
    print("\n" + "="*60)
    print("КРОК 5: Walk-Forward бектестинг")
    print("="*60)

    from backtesting.backtest import WalkForwardBacktest

    bt = WalkForwardBacktest(
        prices=data["prices"],
        train_window=504,
        rebal_freq=21,
        alpha=CONFIDENCE_LEVEL,
        risk_free_rate=RISK_FREE_RATE,
    )
    results = bt.run(verbose=True)

    print("\n✅ M5 готово.")
    return dict(backtest=bt, results=results)


def step_plots(data: dict, risk_data: dict, bt_data: dict, port_data: dict) -> list:
    print("\n" + "="*60)
    print("КРОК 6: Генерація графіків")
    print("="*60)

    from visualization.plots import (
        plot_price_history, plot_returns_distribution,
        plot_correlation_heatmap, plot_nav_comparison,
        plot_drawdown, plot_efficient_frontier, plot_mc_fan_chart,
    )

    returns = data["returns"]
    prices  = data["prices"]
    bt      = bt_data["backtest"]
    opt     = port_data["optimizer"]

    nav_df      = bt.get_nav_dataframe()
    nav_ml      = nav_df["ml_markowitz"]
    nav_classic = nav_df["classic_markowitz"]
    nav_ew      = nav_df["equal_weight"]

    saved = []
    saved.append(plot_price_history(prices))
    saved.append(plot_returns_distribution(returns, ticker="AAPL"))
    saved.append(plot_correlation_heatmap(returns))
    saved.append(plot_nav_comparison(nav_ml, nav_classic, nav_ew))
    saved.append(plot_drawdown(nav_ml, nav_classic, nav_ew))

    mc = risk_data["mc_results"]
    saved.append(plot_mc_fan_chart(
        mc["p05"], mc["p25"], mc["p50"],
        mc["p75"], mc["p95"], var_level=mc["var_1d"],
    ))

    if opt.frontier_risks is not None:
        S = opt.Sigma
        saved.append(plot_efficient_frontier(
            opt.frontier_risks, opt.frontier_returns,
            max_sharpe_pt=(
                float(np.sqrt(opt.max_sharpe_weights @ S @ opt.max_sharpe_weights)),
                float(opt.mu @ opt.max_sharpe_weights),
            ) if opt.max_sharpe_weights is not None else None,
            min_var_pt=(
                float(np.sqrt(opt.min_var_weights @ S @ opt.min_var_weights)),
                float(opt.mu @ opt.min_var_weights),
            ) if opt.min_var_weights is not None else None,
            asset_risks=np.sqrt(np.diag(S)),
            asset_returns=opt.mu,
            asset_names=opt.tickers,
        ))

    print(f"\n✅ M6 готово: збережено {len(saved)} графіків → docs/figures/")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Система управління ризиками портфеля")
    parser.add_argument("--step", default="all",
                        choices=["all","data","volatility","risk","portfolio","backtest","plots"])
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  СИСТЕМА УПРАВЛІННЯ РИЗИКАМИ ПОРТФЕЛЯ ІНВЕСТИЦІЙ")
    print("  на основі алгоритмів машинного навчання")
    print("="*60)
    print(f"  Тикери:  {', '.join(TICKERS)}")
    print(f"  Період:  {START_DATE} → {END_DATE}")
    print(f"  VaR α:   {CONFIDENCE_LEVEL*100:.0f}%")

    data     = step_data(force_download=args.force_download)
    if args.step == "data": return

    vol_data = step_volatility(data)
    if args.step == "volatility": return

    risk_data = step_risk(data, vol_data)
    if args.step == "risk": return

    port_data = step_portfolio(data, vol_data)
    if args.step == "portfolio": return

    bt_data  = step_backtest(data)
    if args.step == "backtest": return

    step_plots(data, risk_data, bt_data, port_data)

    print("\n" + "="*60)
    print("  ✅ ПАЙПЛАЙН ЗАВЕРШЕНО")
    print("  Графіки → docs/figures/")
    print("="*60)


if __name__ == "__main__":
    main()
