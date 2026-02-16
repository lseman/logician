## Tool: stat_forecast

**Description:** Statistical forecast using Nixtla's StatsForecast. Supports 13 methods from naive baselines to auto-tuned models.

**Parameters:**
- method (str, required): One of 'naive', 'snaive', 'moving_avg', 'ses', 'holt_winters', 'auto_arima', 'auto_ets', 'auto_theta', 'auto_ces', 'adida', 'croston', 'mstl'
- periods (int, required): Forecast horizon
- season_length (int, optional): Seasonal period (default 0 = auto-detect)
- level (list[int], optional): Prediction interval levels (default [80, 95])

**Returns:** JSON with forecast including prediction intervals at each level

**Implementation:**
```python
def stat_forecast(method, periods, season_length=0, level=None):
    """Statistical forecast via Nixtla StatsForecast."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    h = int(periods)
    if h <= 0:
        return _safe_json({"status": "error", "error": "periods must be positive"})

    if level is None:
        level = [80, 95]

    # --- Resolve value column and build Nixtla DataFrame ---
    col = "value" if "value" in ctx.data.columns else (ctx.value_columns[0] if ctx.value_columns else None)
    if col is None:
        return _safe_json({"status": "error", "error": "No numeric value column found"})

    df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.to_datetime(ctx.data["date"]),
        "y": pd.to_numeric(ctx.data[col], errors="coerce").astype(float),
    })
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    if len(df) < 8:
        return _safe_json({"status": "error", "error": "Need >= 8 observations"})

    # --- Infer frequency ---
    freq = pd.infer_freq(df["ds"])
    if freq is None:
        median_delta = df["ds"].diff().median()
        if median_delta <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_delta <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_delta <= pd.Timedelta(days=8):
            freq = "W"
        elif median_delta <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "QS"

    s = int(season_length) if season_length else _guess_season_length(df["y"].values)

    # --- Model registry ---
    from statsforecast import StatsForecast
    from statsforecast.models import (
        Naive, SeasonalNaive, WindowAverage, SimpleExponentialSmoothing,
        HoltWinters, AutoARIMA, AutoETS, AutoTheta, AutoCES,
        ADIDA, CrostonOptimized, MSTL
    )
    registry = {
        "naive":        Naive(),
        "snaive":       SeasonalNaive(season_length=s),
        "moving_avg":   WindowAverage(window_size=s or 7),
        "ses":          SimpleExponentialSmoothing(alpha=0.3),
        "holt_winters": HoltWinters(season_length=s or 1),
        "auto_arima":   AutoARIMA(season_length=s),
        "auto_ets":     AutoETS(season_length=s),
        "auto_theta":   AutoTheta(season_length=s),
        "auto_ces":     AutoCES(season_length=s),
        "adida":        ADIDA(),
        "croston":      CrostonOptimized(),
        "mstl":         MSTL(season_length=s),
    }

    if method not in registry:
        available = ", ".join(sorted(registry.keys()))
        return _safe_json({"status": "error", "error": f"Unknown method '{method}'. Available: {available}"})

    try:
        model = registry[method]
        sf = StatsForecast(models=[model], freq=freq, n_jobs=1)
        forecast_df = sf.forecast(df=df, h=h, level=level).reset_index()

        base_col = [c for c in forecast_df.columns if c not in ("unique_id", "ds") and "lo" not in c and "hi" not in c][0]

        forecast_list = []
        for _, row in forecast_df.iterrows():
            entry = {"date": str(row["ds"]), "value": float(row[base_col])}
            for lv in level:
                lo_col = [c for c in forecast_df.columns if f"-lo-{lv}" in c]
                hi_col = [c for c in forecast_df.columns if f"-hi-{lv}" in c]
                if lo_col:
                    entry[f"lower_{lv}"] = float(row[lo_col[0]])
                if hi_col:
                    entry[f"upper_{lv}"] = float(row[hi_col[0]])
            forecast_list.append(entry)

        return _safe_json({
            "status": "ok", "method": method, "engine": "statsforecast",
            "horizon": h, "column": col, "season_length": s,
            "levels": level, "forecast": forecast_list,
        })

    except Exception as e:
        return _safe_json({"status": "error", "error": f"{method} failed: {str(e)}"})
```

---

## Tool: neural_forecast

**Description:** Deep learning forecast using Nixtla's NeuralForecast. Supports SOTA architectures including N-BEATS, N-HiTS, PatchTST, TimesNet, TFT, iTransformer, TiDE, KAN, and more.

**Parameters:**
- method (str, required): One of 'nbeats', 'nhits', 'patchtst', 'timesnet', 'tft', 'itransformer', 'timellm', 'mlp', 'lstm', 'tide', 'kan'
- periods (int, required): Forecast horizon
- season_length (int, optional): Seasonal period (default 0 = auto-detect). Used to scale input_size.
- level (list[int], optional): Prediction interval levels (default [80, 95])
- max_steps (int, optional): Override training steps (default 0 = use model default)
- input_size_multiplier (int, optional): input_size = multiplier × season_length (default 0 = use 2×)
- extra_kwargs (dict, optional): Additional model kwargs to override defaults

**Returns:** JSON with forecast and bootstrap intervals if model doesn't produce native intervals

**Implementation:**
```python
def neural_forecast(method, periods, season_length=0, level=None,
                    max_steps=0, input_size_multiplier=0, extra_kwargs=None):
    """Neural forecast via Nixtla NeuralForecast."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    h = int(periods)
    if h <= 0:
        return _safe_json({"status": "error", "error": "periods must be positive"})

    if level is None:
        level = [80, 95]

    # --- Resolve value column and build Nixtla DataFrame ---
    col = "value" if "value" in ctx.data.columns else (ctx.value_columns[0] if ctx.value_columns else None)
    if col is None:
        return _safe_json({"status": "error", "error": "No numeric value column found"})

    df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.to_datetime(ctx.data["date"]),
        "y": pd.to_numeric(ctx.data[col], errors="coerce").astype(float),
    })
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    if len(df) < 30:
        return _safe_json({"status": "error", "error": "Neural models need >= 30 observations"})

    # --- Infer frequency ---
    freq = pd.infer_freq(df["ds"])
    if freq is None:
        median_delta = df["ds"].diff().median()
        if median_delta <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_delta <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_delta <= pd.Timedelta(days=8):
            freq = "W"
        elif median_delta <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "QS"

    s = int(season_length) if season_length else _guess_season_length(df["y"].values)

    # --- Model registry (all inline) ---
    NEURAL_MODEL_REGISTRY = {
        "nbeats": {
            "class": "NBEATS", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "early_stop_patience_steps": 10},
        },
        "nhits": {
            "class": "NHITS", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "early_stop_patience_steps": 10},
        },
        "patchtst": {
            "class": "PatchTST", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "patch_len": 16, "stride": 8,
                               "hidden_size": 64, "n_heads": 4, "encoder_layers": 2,
                               "early_stop_patience_steps": 10},
        },
        "timesnet": {
            "class": "TimesNet", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "hidden_size": 64,
                               "early_stop_patience_steps": 10},
        },
        "tft": {
            "class": "TFT", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "hidden_size": 64,
                               "early_stop_patience_steps": 10},
        },
        "itransformer": {
            "class": "iTransformer", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "n_heads": 4, "hidden_size": 64,
                               "early_stop_patience_steps": 10},
        },
        "timellm": {
            "class": "TimeLLM", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 300,
                               "scaler_type": "robust"},
        },
        "mlp": {
            "class": "MLP", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 300,
                               "scaler_type": "robust", "num_layers": 3, "hidden_size": 128},
        },
        "lstm": {
            "class": "LSTM", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "encoder_hidden_size": 64,
                               "encoder_n_layers": 2},
        },
        "tide": {
            "class": "TiDE", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust", "hidden_size": 128,
                               "decoder_output_dim": 16},
        },
        "kan": {
            "class": "KAN", "module": "neuralforecast.models",
            "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                               "scaler_type": "robust"},
        },
    }

    if method not in NEURAL_MODEL_REGISTRY:
        available = ", ".join(sorted(NEURAL_MODEL_REGISTRY.keys()))
        return _safe_json({"status": "error", "error": f"Unknown method '{method}'. Available: {available}"})

    try:
        import importlib
        from neuralforecast import NeuralForecast

        spec = NEURAL_MODEL_REGISTRY[method]
        mod = importlib.import_module(spec["module"])
        ModelClass = getattr(mod, spec["class"])

        kwargs = dict(spec["default_kwargs"])
        kwargs["h"] = h

        mult = int(input_size_multiplier) if input_size_multiplier else kwargs.get("input_size", 2)
        kwargs["input_size"] = max(mult * s, h, 24)

        if max_steps:
            kwargs["max_steps"] = int(max_steps)
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        fit_val_size = int(kwargs.pop("val_size", 0) or 0)
        if kwargs.get("early_stop_patience_steps", 0) and fit_val_size <= 0:
            fit_val_size = max(h, int(len(df) * 0.1))

        model = ModelClass(**kwargs)
        nf = NeuralForecast(models=[model], freq=freq)
        if fit_val_size > 0:
            try:
                nf.fit(df=df, val_size=fit_val_size)
            except TypeError:
                nf.fit(df=df)
        else:
            nf.fit(df=df)
        forecast_df = nf.predict().reset_index()

        fc_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
        base_col = fc_cols[0] if fc_cols else None
        if base_col is None:
            return _safe_json({"status": "error", "error": "No forecast column in output"})

        forecast_list = []
        for _, row in forecast_df.iterrows():
            entry = {"date": str(row["ds"]), "value": float(row[base_col])}
            for lv in level:
                lo_col = [c for c in fc_cols if f"-lo-{lv}" in c]
                hi_col = [c for c in fc_cols if f"-hi-{lv}" in c]
                if lo_col:
                    entry[f"lower_{lv}"] = float(row[lo_col[0]])
                if hi_col:
                    entry[f"upper_{lv}"] = float(row[hi_col[0]])
            forecast_list.append(entry)

        # Bootstrap intervals fallback if model doesn't produce native intervals
        has_intervals = any(f"lower_{lv}" in forecast_list[0] for lv in level)
        if not has_intervals:
            fc_vals = np.array([f["value"] for f in forecast_list])
            y = df["y"].values
            resids = y[-min(50, len(y)):] - np.mean(y[-min(50, len(y)):])
            boots = np.array([
                fc_vals + np.random.choice(resids, size=len(fc_vals))
                for _ in range(200)
            ])
            for lv in level:
                lo_pct = (100 - lv) / 2
                hi_pct = 100 - lo_pct
                lower = np.percentile(boots, lo_pct, axis=0)
                upper = np.percentile(boots, hi_pct, axis=0)
                for i in range(len(forecast_list)):
                    forecast_list[i][f"lower_{lv}"] = float(lower[i])
                    forecast_list[i][f"upper_{lv}"] = float(upper[i])

        return _safe_json({
            "status": "ok", "method": method, "engine": "neuralforecast",
            "horizon": h, "column": col, "season_length": s,
            "input_size": kwargs["input_size"],
            "max_steps": kwargs.get("max_steps"),
            "levels": level, "forecast": forecast_list,
        })

    except Exception as e:
        return _safe_json({"status": "error", "error": f"{method} failed: {str(e)}"})
```

---

## Tool: ensemble_forecast

**Description:** Cross-validation-weighted ensemble combining statistical and neural forecasters. Weights are computed via rolling-origin CV using inverse MAE (or RMSE/equal).

**Parameters:**
- stat_methods (list, optional): Statistical method names from stat_forecast registry
- neural_methods (list, optional): Neural method names from neural_forecast registry
- periods (int, required): Forecast horizon
- season_length (int, optional): Seasonal period (default 0 = auto)
- level (list[int], optional): Prediction interval levels (default [80, 95])
- cv_windows (int, optional): Number of cross-validation windows for weighting (default 3)
- weighting (str, optional): 'inverse_mae', 'inverse_rmse', or 'equal' (default 'inverse_mae')

**Returns:** JSON with weighted ensemble forecast, individual weights, and CV scores

**Implementation:**
```python
def ensemble_forecast(stat_methods=None, neural_methods=None, periods=0,
                      season_length=0, level=None, cv_windows=3,
                      weighting="inverse_mae"):
    """Ensemble with CV-based auto-weighting."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    h = int(periods)
    if h <= 0:
        return _safe_json({"status": "error", "error": "periods must be positive"})

    stat_methods = stat_methods or []
    neural_methods = neural_methods or []
    all_methods = stat_methods + neural_methods
    if not all_methods:
        return _safe_json({"status": "error", "error": "Provide at least one method"})

    if level is None:
        level = [80, 95]

    # --- Resolve value column and build Nixtla DataFrame ---
    col = "value" if "value" in ctx.data.columns else (ctx.value_columns[0] if ctx.value_columns else None)
    if col is None:
        return _safe_json({"status": "error", "error": "No numeric value column found"})

    df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.to_datetime(ctx.data["date"]),
        "y": pd.to_numeric(ctx.data[col], errors="coerce").astype(float),
    })
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    # --- Infer frequency ---
    freq = pd.infer_freq(df["ds"])
    if freq is None:
        median_delta = df["ds"].diff().median()
        if median_delta <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_delta <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_delta <= pd.Timedelta(days=8):
            freq = "W"
        elif median_delta <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "QS"

    s = int(season_length) if season_length else _guess_season_length(df["y"].values)

    # --- Stat model builder (inline) ---
    def _build_stat_models(s):
        from statsforecast.models import (
            Naive, SeasonalNaive, WindowAverage, SimpleExponentialSmoothing,
            HoltWinters, AutoARIMA, AutoETS, AutoTheta, AutoCES,
            ADIDA, CrostonOptimized, MSTL
        )
        return {
            "naive": Naive(), "snaive": SeasonalNaive(season_length=s),
            "moving_avg": WindowAverage(window_size=s or 7),
            "ses": SimpleExponentialSmoothing(alpha=0.3),
            "holt_winters": HoltWinters(season_length=s or 1),
            "auto_arima": AutoARIMA(season_length=s),
            "auto_ets": AutoETS(season_length=s),
            "auto_theta": AutoTheta(season_length=s),
            "auto_ces": AutoCES(season_length=s),
            "adida": ADIDA(), "croston": CrostonOptimized(),
            "mstl": MSTL(season_length=s),
        }

    # --- Neural model builder (inline) ---
    def _build_neural_model(m, h, s, n_obs):
        import importlib
        NEURAL_MODEL_REGISTRY = {
            "nbeats": {"class": "NBEATS", "module": "neuralforecast.models",
                       "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                          "scaler_type": "robust", "early_stop_patience_steps": 10}},
            "nhits": {"class": "NHITS", "module": "neuralforecast.models",
                      "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                         "scaler_type": "robust", "early_stop_patience_steps": 10}},
            "patchtst": {"class": "PatchTST", "module": "neuralforecast.models",
                         "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                            "scaler_type": "robust", "patch_len": 16, "stride": 8,
                                            "hidden_size": 64, "n_heads": 4, "encoder_layers": 2,
                                            "early_stop_patience_steps": 10}},
            "timesnet": {"class": "TimesNet", "module": "neuralforecast.models",
                         "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                            "scaler_type": "robust", "hidden_size": 64,
                                            "early_stop_patience_steps": 10}},
            "tft": {"class": "TFT", "module": "neuralforecast.models",
                    "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                       "scaler_type": "robust", "hidden_size": 64,
                                       "early_stop_patience_steps": 10}},
            "itransformer": {"class": "iTransformer", "module": "neuralforecast.models",
                             "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                                "scaler_type": "robust", "n_heads": 4, "hidden_size": 64,
                                                "early_stop_patience_steps": 10}},
            "mlp": {"class": "MLP", "module": "neuralforecast.models",
                    "default_kwargs": {"input_size": 2, "h": None, "max_steps": 300,
                                       "scaler_type": "robust", "num_layers": 3, "hidden_size": 128}},
            "lstm": {"class": "LSTM", "module": "neuralforecast.models",
                     "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                        "scaler_type": "robust", "encoder_hidden_size": 64,
                                        "encoder_n_layers": 2}},
            "tide": {"class": "TiDE", "module": "neuralforecast.models",
                     "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                        "scaler_type": "robust", "hidden_size": 128,
                                        "decoder_output_dim": 16}},
            "kan": {"class": "KAN", "module": "neuralforecast.models",
                    "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                       "scaler_type": "robust"}},
        }
        if m not in NEURAL_MODEL_REGISTRY:
            return None
        spec = NEURAL_MODEL_REGISTRY[m]
        mod = importlib.import_module(spec["module"])
        ModelClass = getattr(mod, spec["class"])
        kwargs = dict(spec["default_kwargs"])
        kwargs["h"] = h
        mult = kwargs.get("input_size", 2)
        kwargs["input_size"] = max(mult * s, h, 24)
        kwargs.pop("val_size", None)
        return ModelClass(**kwargs)

    # --- Helper to extract CV scores from a cv_df ---
    def _extract_scores(cv_df, models_list):
        scores = {}
        for m_obj in models_list:
            m_name = type(m_obj).__name__
            matching = [c for c in cv_df.columns if m_name in c and "lo" not in c and "hi" not in c]
            if matching:
                preds = cv_df[matching[0]].values
                actuals = cv_df["y"].values
                mask = np.isfinite(preds) & np.isfinite(actuals)
                if mask.sum() > 0:
                    err = actuals[mask] - preds[mask]
                    scores[m_name] = {
                        "mae": float(np.mean(np.abs(err))),
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                    }
        return scores

    # --- Step 1: Cross-validation to compute weights ---
    method_scores = {}

    if stat_methods:
        try:
            from statsforecast import StatsForecast
            stat_registry = _build_stat_models(s)
            models = [stat_registry[m] for m in stat_methods if m in stat_registry]
            if models:
                sf = StatsForecast(models=models, freq=freq, n_jobs=1)
                cv_df = sf.cross_validation(df=df, h=h, n_windows=cv_windows, level=level).reset_index()
                method_scores.update(_extract_scores(cv_df, models))
        except Exception:
            pass

    if neural_methods:
        try:
            from neuralforecast import NeuralForecast
            models = [_build_neural_model(m, h, s, len(df)) for m in neural_methods]
            models = [m for m in models if m is not None]
            if models:
                nf = NeuralForecast(models=models, freq=freq)
                neural_val_size = max(h, int(len(df) * 0.1))
                try:
                    cv_df = nf.cross_validation(
                        df=df,
                        n_windows=cv_windows,
                        val_size=neural_val_size,
                    ).reset_index()
                except TypeError:
                    cv_df = nf.cross_validation(df=df, n_windows=cv_windows).reset_index()
                method_scores.update(_extract_scores(cv_df, models))
        except Exception:
            pass

    # --- Step 2: Compute weights ---
    if weighting == "equal" or not method_scores:
        weights = {m: 1.0 / len(all_methods) for m in all_methods}
    else:
        metric_key = "mae" if "mae" in weighting else "rmse"
        raw = {m: 1.0 / (scores[metric_key] + 1e-8) for m, scores in method_scores.items()}
        total = sum(raw.values())
        weights = {m: v / total for m, v in raw.items()}
        unscored = [m for m in all_methods if m not in weights]
        if unscored:
            avg_w = np.mean(list(weights.values())) if weights else 1.0 / len(all_methods)
            for m in unscored:
                weights[m] = avg_w
            total = sum(weights.values())
            weights = {m: v / total for m, v in weights.items()}

    # --- Step 3: Generate final forecasts and combine ---
    forecasts = {}

    if stat_methods:
        try:
            from statsforecast import StatsForecast
            stat_registry = _build_stat_models(s)
            models = [stat_registry[m] for m in stat_methods if m in stat_registry]
            if models:
                sf = StatsForecast(models=models, freq=freq, n_jobs=1)
                fc_df = sf.forecast(df=df, h=h, level=level).reset_index()
                for m_obj in models:
                    m_name = type(m_obj).__name__
                    matching = [c for c in fc_df.columns if m_name in c and "lo" not in c and "hi" not in c]
                    if matching:
                        forecasts[m_name] = fc_df[matching[0]].values
        except Exception:
            pass

    if neural_methods:
        try:
            from neuralforecast import NeuralForecast
            models = [_build_neural_model(m, h, s, len(df)) for m in neural_methods]
            models = [m for m in models if m is not None]
            if models:
                nf = NeuralForecast(models=models, freq=freq)
                neural_val_size = max(h, int(len(df) * 0.1))
                try:
                    nf.fit(df=df, val_size=neural_val_size)
                except TypeError:
                    nf.fit(df=df)
                fc_df = nf.predict().reset_index()
                for m_obj in models:
                    m_name = type(m_obj).__name__
                    matching = [c for c in fc_df.columns if m_name in c and "lo" not in c and "hi" not in c]
                    if matching:
                        forecasts[m_name] = fc_df[matching[0]].values
        except Exception:
            pass

    if not forecasts:
        return _safe_json({"status": "error", "error": "No forecasts produced"})

    ensemble_fc = np.zeros(h)
    total_w = 0.0
    used_weights = {}
    for m_name, fc_vals in forecasts.items():
        w = weights.get(m_name, weights.get(m_name.lower(), 1.0 / len(forecasts)))
        ensemble_fc += w * fc_vals[:h]
        total_w += w
        used_weights[m_name] = w

    if total_w > 0:
        ensemble_fc /= total_w

    last_date = df["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=h + 1, freq=freq)[1:]

    y = df["y"].values
    resids = y[-min(50, len(y)):] - np.mean(y[-min(50, len(y)):])
    boots = np.array([ensemble_fc + np.random.choice(resids, size=h) for _ in range(200)])

    forecast_list = []
    for i, (d, v) in enumerate(zip(future_dates, ensemble_fc)):
        entry = {"date": str(d), "value": float(v)}
        for lv in level:
            lo_pct = (100 - lv) / 2
            hi_pct = 100 - lo_pct
            entry[f"lower_{lv}"] = float(np.percentile(boots[:, i], lo_pct))
            entry[f"upper_{lv}"] = float(np.percentile(boots[:, i], hi_pct))
        forecast_list.append(entry)

    return _safe_json({
        "status": "ok", "method": "cv_weighted_ensemble",
        "stat_methods": stat_methods, "neural_methods": neural_methods,
        "weights": {k: round(v, 4) for k, v in used_weights.items()},
        "cv_scores": method_scores, "weighting": weighting,
        "cv_windows": cv_windows, "horizon": h, "column": col,
        "season_length": s, "levels": level, "forecast": forecast_list,
    })
```

---

## Tool: cross_validate

**Description:** Rolling-origin cross-validation using Nixtla's built-in `.cross_validation()`. Returns per-method MAE, RMSE, sMAPE and a ranked leaderboard.

**Parameters:**
- methods (list, optional): Statistical method names
- neural_methods (list, optional): Neural method names
- horizon (int, required): Forecast horizon per window
- n_windows (int, optional): Number of CV windows (default 5)
- season_length (int, optional): Seasonal period (default 0 = auto)

**Returns:** JSON with ranked leaderboard of metrics per method

**Implementation:**
```python
def cross_validate(methods=None, neural_methods=None, horizon=0,
                   n_windows=5, season_length=0):
    """Rolling-origin cross-validation via Nixtla."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    h = int(horizon)
    if h <= 0:
        return _safe_json({"status": "error", "error": "horizon must be positive"})

    methods = methods or []
    neural_methods = neural_methods or []
    if not methods and not neural_methods:
        return _safe_json({"status": "error", "error": "Provide at least one method"})

    # --- Resolve value column and build Nixtla DataFrame ---
    col = "value" if "value" in ctx.data.columns else (ctx.value_columns[0] if ctx.value_columns else None)
    if col is None:
        return _safe_json({"status": "error", "error": "No numeric value column found"})

    df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.to_datetime(ctx.data["date"]),
        "y": pd.to_numeric(ctx.data[col], errors="coerce").astype(float),
    })
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    # --- Infer frequency ---
    freq = pd.infer_freq(df["ds"])
    if freq is None:
        median_delta = df["ds"].diff().median()
        if median_delta <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_delta <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_delta <= pd.Timedelta(days=8):
            freq = "W"
        elif median_delta <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "QS"

    s = int(season_length) if season_length else _guess_season_length(df["y"].values)

    # --- Helper to compute metrics from cv_df ---
    def _compute_metrics(cv_df, models_list):
        res = {}
        for m_obj in models_list:
            m_name = type(m_obj).__name__
            matching = [c for c in cv_df.columns if m_name in c and "lo" not in c and "hi" not in c]
            if matching:
                preds = cv_df[matching[0]].values
                actuals = cv_df["y"].values
                mask = np.isfinite(preds) & np.isfinite(actuals)
                if mask.sum() > 0:
                    err = actuals[mask] - preds[mask]
                    abs_err = np.abs(err)
                    res[m_name] = {
                        "mae": float(np.mean(abs_err)),
                        "rmse": float(np.sqrt(np.mean(err ** 2))),
                        "smape": float(np.mean(
                            2 * abs_err / (np.abs(actuals[mask]) + np.abs(preds[mask]) + 1e-8)
                        ) * 100),
                        "n_predictions": int(mask.sum()),
                    }
        return res

    results = {}

    if methods:
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import (
                Naive, SeasonalNaive, WindowAverage, SimpleExponentialSmoothing,
                HoltWinters, AutoARIMA, AutoETS, AutoTheta, AutoCES,
                ADIDA, CrostonOptimized, MSTL
            )
            stat_registry = {
                "naive": Naive(), "snaive": SeasonalNaive(season_length=s),
                "moving_avg": WindowAverage(window_size=s or 7),
                "ses": SimpleExponentialSmoothing(alpha=0.3),
                "holt_winters": HoltWinters(season_length=s or 1),
                "auto_arima": AutoARIMA(season_length=s),
                "auto_ets": AutoETS(season_length=s),
                "auto_theta": AutoTheta(season_length=s),
                "auto_ces": AutoCES(season_length=s),
                "adida": ADIDA(), "croston": CrostonOptimized(),
                "mstl": MSTL(season_length=s),
            }
            models = [stat_registry[m] for m in methods if m in stat_registry]
            if models:
                sf = StatsForecast(models=models, freq=freq, n_jobs=1)
                cv_df = sf.cross_validation(df=df, h=h, n_windows=n_windows).reset_index()
                results.update(_compute_metrics(cv_df, models))
        except Exception as e:
            results["_stat_error"] = str(e)

    if neural_methods:
        try:
            import importlib
            from neuralforecast import NeuralForecast

            NEURAL_MODEL_REGISTRY = {
                "nbeats": {"class": "NBEATS", "module": "neuralforecast.models",
                           "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                              "scaler_type": "robust", "early_stop_patience_steps": 10}},
                "nhits": {"class": "NHITS", "module": "neuralforecast.models",
                          "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                             "scaler_type": "robust", "early_stop_patience_steps": 10}},
                "patchtst": {"class": "PatchTST", "module": "neuralforecast.models",
                             "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                                "scaler_type": "robust", "patch_len": 16, "stride": 8,
                                                "hidden_size": 64, "n_heads": 4, "encoder_layers": 2,
                                                "early_stop_patience_steps": 10}},
                "timesnet": {"class": "TimesNet", "module": "neuralforecast.models",
                             "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                                "scaler_type": "robust", "hidden_size": 64,
                                                "early_stop_patience_steps": 10}},
                "tft": {"class": "TFT", "module": "neuralforecast.models",
                        "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                           "scaler_type": "robust", "hidden_size": 64,
                                           "early_stop_patience_steps": 10}},
                "itransformer": {"class": "iTransformer", "module": "neuralforecast.models",
                                 "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                                    "scaler_type": "robust", "n_heads": 4, "hidden_size": 64,
                                                    "early_stop_patience_steps": 10}},
                "mlp": {"class": "MLP", "module": "neuralforecast.models",
                        "default_kwargs": {"input_size": 2, "h": None, "max_steps": 300,
                                           "scaler_type": "robust", "num_layers": 3, "hidden_size": 128}},
                "lstm": {"class": "LSTM", "module": "neuralforecast.models",
                         "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                            "scaler_type": "robust", "encoder_hidden_size": 64,
                                            "encoder_n_layers": 2}},
                "tide": {"class": "TiDE", "module": "neuralforecast.models",
                         "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                            "scaler_type": "robust", "hidden_size": 128,
                                            "decoder_output_dim": 16}},
                "kan": {"class": "KAN", "module": "neuralforecast.models",
                        "default_kwargs": {"input_size": 2, "h": None, "max_steps": 100,
                                           "scaler_type": "robust"}},
            }

            models = []
            for m in neural_methods:
                if m in NEURAL_MODEL_REGISTRY:
                    spec = NEURAL_MODEL_REGISTRY[m]
                    mod = importlib.import_module(spec["module"])
                    ModelClass = getattr(mod, spec["class"])
                    kwargs = dict(spec["default_kwargs"])
                    kwargs["h"] = h
                    mult = kwargs.get("input_size", 2)
                    kwargs["input_size"] = max(mult * s, h, 24)
                    kwargs.pop("val_size", None)
                    models.append(ModelClass(**kwargs))
            if models:
                nf = NeuralForecast(models=models, freq=freq)
                neural_val_size = max(h, int(len(df) * 0.1))
                try:
                    cv_df = nf.cross_validation(
                        df=df,
                        n_windows=n_windows,
                        val_size=neural_val_size,
                    ).reset_index()
                except TypeError:
                    cv_df = nf.cross_validation(df=df, n_windows=n_windows).reset_index()
                results.update(_compute_metrics(cv_df, models))
        except Exception as e:
            results["_neural_error"] = str(e)

    ranked = sorted(
        [(m, sc) for m, sc in results.items() if not m.startswith("_")],
        key=lambda x: x[1].get("mae", float("inf"))
    )
    ranking = [{"rank": i + 1, "method": m, **sc} for i, (m, sc) in enumerate(ranked)]

    return _safe_json({
        "status": "ok", "horizon": h, "n_windows": n_windows,
        "season_length": s, "column": col,
        "ranking": ranking,
        "errors": {k: v for k, v in results.items() if k.startswith("_")},
    })
```

---

## Tool: suggest_models

**Description:** Analyze data characteristics and suggest appropriate forecasting models with reasoning.

**Parameters:** None

**Returns:** JSON with data characteristics, model suggestions, and reasoning

**Implementation:**
```python
def suggest_models():
    """Suggest models based on data characteristics."""
    if not ctx.loaded:
        return _safe_json({"status": "error", "error": "No data loaded"})

    col = "value" if "value" in ctx.data.columns else (ctx.value_columns[0] if ctx.value_columns else None)
    if col is None:
        return _safe_json({"status": "error", "error": "No numeric value column found"})

    y = pd.to_numeric(ctx.data[col], errors="coerce").values.astype(float)
    y = y[np.isfinite(y)]
    n = len(y)
    s = _guess_season_length(y)

    characteristics = {
        "n_observations": n,
        "season_length_guess": s,
        "has_trend": bool(abs(np.corrcoef(np.arange(n), y)[0, 1]) > 0.3),
        "has_seasonality": s > 1,
        "coefficient_of_variation": float(np.std(y) / (np.abs(np.mean(y)) + 1e-8)),
        "has_zeros": bool(np.any(y == 0)),
        "intermittent": bool(np.mean(y == 0) > 0.2) if np.any(y == 0) else False,
    }

    suggestions = {
        "always_include": ["auto_arima", "auto_ets"],
        "stat_baselines": [],
        "neural_models": [],
        "reasoning": [],
    }

    if characteristics["has_seasonality"]:
        suggestions["stat_baselines"].extend(["auto_theta", "mstl", "holt_winters"])
        suggestions["reasoning"].append(f"Detected seasonality (period ~{s}), added seasonal models")
    else:
        suggestions["stat_baselines"].extend(["auto_theta", "ses"])
        suggestions["reasoning"].append("No clear seasonality, sticking with non-seasonal models")

    if characteristics["intermittent"]:
        suggestions["stat_baselines"].extend(["croston", "adida"])
        suggestions["reasoning"].append("Intermittent demand detected, added Croston/ADIDA")

    if n >= 100:
        suggestions["neural_models"].append("nhits")
        suggestions["reasoning"].append("N-HiTS: good default neural model, handles multi-scale patterns")
        if n >= 200:
            suggestions["neural_models"].append("nbeats")
            suggestions["reasoning"].append("N-BEATS: strong with enough data, interpretable decomposition")
        if n >= 300 and characteristics["has_seasonality"]:
            suggestions["neural_models"].append("patchtst")
            suggestions["reasoning"].append("PatchTST: excellent for seasonal data with sufficient length")
            suggestions["neural_models"].append("timesnet")
            suggestions["reasoning"].append("TimesNet: captures multi-period temporal patterns via 2D convolutions")
        if n >= 500:
            suggestions["neural_models"].append("tft")
            suggestions["reasoning"].append("TFT: powerful with long history, attention-based interpretability")
            suggestions["neural_models"].append("itransformer")
            suggestions["reasoning"].append("iTransformer: inverted attention works well on longer series")
    elif n >= 50:
        suggestions["neural_models"].extend(["mlp", "nhits"])
        suggestions["reasoning"].append("Limited data: MLP and N-HiTS are most data-efficient neural models")
    else:
        suggestions["reasoning"].append("Too few observations for neural models, stick with statistical methods")

    suggestions["recommended_ensemble"] = {
        "stat": suggestions["always_include"] + suggestions["stat_baselines"][:2],
        "neural": suggestions["neural_models"][:3],
    }

    return _safe_json({
        "status": "ok",
        "characteristics": characteristics,
        "suggestions": suggestions,
    })
```

---
