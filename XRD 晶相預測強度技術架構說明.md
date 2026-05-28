# v6.4 MLOps 升級規格書

## 目標

在不重構檔案結構的前提下，為 v6.4 加入六項 MLOps 基礎設施，
使其從「每次跑完看 console」升級到「可追溯、可測試、可偵測異常」。

## 基準代碼

`cement_feature_analysis_v6_4.py`（2088 行）
`cement_predict_v6_4.py`（218 行）

---

## 修改一覽

| 編號 | 改善項 | 改動位置 | 新增行數 | 依賴 |
|------|--------|----------|----------|------|
| M1 | `if __name__` 保護 | L1462 前 | 2 行 | 無 |
| M2 | 數據指紋 + pkl 版本化 | L1466 後, L2044 | ~15 行 | 無 |
| M3 | config.yaml 分離 | L75-88, 新檔案 | ~40 行 | 無 |
| M4 | experiment_log.csv | 主迴圈結尾 | ~35 行 | M2 |
| M5 | feature_stats OOD 檢查 | pkl 存檔, 預測腳本 | ~30 行 | 無 |
| M6 | test notebook | 新檔案 | ~120 行 | M1 |

建議順序：M1 → M2 → M3 → M5 → M4 → M6

---

## M1：`if __name__ == '__main__':` 保護

### 問題

目前 import `cement_feature_analysis_v6_4` 會立即執行全部 2088 行，
包括 88 分鐘的主迴圈。這讓單獨測試函數變得不可能。

### 修改

在 L1459（`# 數據載入` 區塊的 `=====` 分隔線前）插入：

```python
# ============================================================
# 以上為函數定義和常量，以下為主執行流程
# ============================================================
if __name__ == '__main__':
```

然後將 L1462-2088 的所有代碼（從 `print("=" * 80)` 到檔案結尾）
**縮排一層**（4 spaces）。

### 注意

TASKS、V51 字典定義（L1399-1428）和 compute_trust_grade（L1431-1448）
以及 TRUST_LABELS（L1451-1456）要留在 `if __name__` **外面**，
因為它們是常量/函數，測試時需要 import。

精確的插入點：在 L1457（TRUST_LABELS 結束）之後、L1459 之前。

```python
TRUST_LABELS = {
    'A': '✓ 可部署',
    'B': '⚠ 可使用',
    'C': '⚠ 實驗性',
    'D': '✗ 不可靠',
}


# ============================================================
# 主執行流程（import 時不執行）
# ============================================================
if __name__ == '__main__':

    # ============================================================
    # 數據載入
    # ============================================================
    print("=" * 80)
    ...（原 L1462-2088 全部縮排 4 spaces）
```

### 驗證

修改後直接執行 `python cement_feature_analysis_v6_4.py` 的行為
和修改前完全一致。但在另一個 notebook 中：

```python
from cement_feature_analysis_v6_4 import compute_trust_grade
# 不會觸發主迴圈
```

---

## M2：數據指紋 + pkl 版本化

### 問題

pkl 檔名固定為 `cement_analysis_v6.4.pkl`，每次重訓覆蓋，
無法追溯「這個 pkl 是用哪個版本的 Excel 訓練的」。

### 修改

**步驟 1：計算數據指紋（L1474 之後，combined 建好後）**

```python
    # ★ MLOps M2: 數據指紋
    _data_hash = hashlib.md5(
        pd.util.hash_pandas_object(combined).values.tobytes()
    ).hexdigest()[:12]
    _data_mtime = os.path.getmtime(FILE_PATH)
    print(f"數據指紋: {_data_hash} (修改時間: "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(_data_mtime))})")
```

**步驟 2：pkl 檔名帶指紋和時間戳**

修改 L88 的 `MODEL_SAVE_PATH`（在配置區塊）。因為 `_data_hash` 要到
數據載入後才有值，所以改為在主迴圈中動態生成：

```python
    # 在數據指紋計算後（上面的代碼之後）
    _run_timestamp = time.strftime('%Y%m%d_%H%M')
    MODEL_SAVE_PATH = (
        f"/content/drive/MyDrive/台泥/AI/"
        f"cement_v6.4_{_data_hash}_{_run_timestamp}.pkl"
    )
    # 同時保留一個固定名稱的 latest 連結
    MODEL_LATEST_PATH = (
        "/content/drive/MyDrive/台泥/AI/cement_analysis_v6.4.pkl"
    )
```

**步驟 3：存入 pkl（修改 L2044 的 save_data）**

在 save_data 字典中新增：

```python
        'data_hash': _data_hash,
        'data_path': FILE_PATH,
        'data_mtime': _data_mtime,
        'run_timestamp': _run_timestamp,
        'version': 'v6.4',
```

存檔時同時寫兩個檔案：

```python
    joblib.dump(save_data, MODEL_SAVE_PATH)
    # 同時更新 latest
    import shutil
    shutil.copy2(MODEL_SAVE_PATH, MODEL_LATEST_PATH)
    print(f"\n✓ 模型存檔: {MODEL_SAVE_PATH}")
    print(f"  (latest → {MODEL_LATEST_PATH})")
```

### 驗證

跑完後 Drive 上會出現兩個檔案：
- `cement_v6.4_a3f2b1c9d4e5_20260519_1430.pkl`（帶指紋）
- `cement_analysis_v6.4.pkl`（latest，預測腳本用）

---

## M3：config.yaml 分離

### 問題

超參數、路徑、開關散落在腳本頂部（L78-88），改一個參數要打開
2088 行的腳本找位置。

### 新增檔案

建立 `/content/drive/MyDrive/台泥/AI/config.yaml`：

```yaml
# cement ML pipeline config v6.4
data:
  file_path: "/content/drive/MyDrive/台泥/AI/水泥XRD(整理後).xlsx"

training:
  use_optuna: true
  optuna_trials: 30
  bootstrap_n: 200
  shap_enabled: true

output:
  shap_dir: "/content/drive/MyDrive/台泥/AI/shap_outputs/"
  model_dir: "/content/drive/MyDrive/台泥/AI/"
  experiment_log: "/content/drive/MyDrive/台泥/AI/experiment_log.csv"
```

### 修改腳本

將 L75-88 的配置區塊替換為：

```python
# ============================================================
# 配置（從 config.yaml 載入，找不到則用預設值）
# ============================================================
import yaml as _yaml

_CONFIG_PATH = "/content/drive/MyDrive/台泥/AI/config.yaml"
try:
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _cfg = _yaml.safe_load(f)
    print(f"✓ 載入配置: {_CONFIG_PATH}")
except Exception:
    _cfg = {}
    print(f"⚠ 配置檔不存在，使用預設值")

FILE_PATH = _cfg.get('data', {}).get(
    'file_path', "/content/drive/MyDrive/台泥/AI/水泥XRD(整理後).xlsx")
USE_OPTUNA = _cfg.get('training', {}).get('use_optuna', True)
OPTUNA_TRIALS = _cfg.get('training', {}).get('optuna_trials', 30)
BOOTSTRAP_N = _cfg.get('training', {}).get('bootstrap_n', 200)
SHAP_ENABLED = _cfg.get('training', {}).get('shap_enabled', True)
SHAP_OUTPUT_DIR = _cfg.get('output', {}).get(
    'shap_dir', "/content/drive/MyDrive/台泥/AI/shap_outputs/")
EXPERIMENT_LOG_PATH = _cfg.get('output', {}).get(
    'experiment_log',
    "/content/drive/MyDrive/台泥/AI/experiment_log.csv")
```

需要在腳本頂部新增 `import yaml`，或用 `try/except` 包裹
（Colab 預裝了 PyYAML）。

### 驗證

- 有 config.yaml 時從中讀取
- 沒有 config.yaml 時 fallback 到預設值，行為和原來一致
- 修改 `optuna_trials: 50` 後重跑，Optuna 確實跑 50 trials

---

## M4：experiment_log.csv

### 問題

v6.1 到 v6.4 的歷次結果無法平行比較。

### 修改

在主迴圈結尾（pkl 存檔之前，約 L2038）新增：

```python
    # ★ MLOps M4: 實驗紀錄
    try:
        log_rows = []
        for task_key in TASKS:
            for side in ['hp', 'sa']:
                b = final_best[task_key][side]
                log_rows.append({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'version': 'v6.4',
                    'data_hash': _data_hash,
                    'task': task_key,
                    'factory': side,
                    'trust_grade': b.get('trust_grade', '?'),
                    'trust_mode': b.get('trust_mode', '?'),
                    'nested_r2': round(b.get('nested_r2', 0), 4),
                    'loo_r2': round(b.get('r2', 0), 4),
                    'official_r2': round(b.get('official_r2', 0), 4),
                    'rmse': round(b.get('rmse', 0), 4),
                    'n_features': len(b.get('features', [])),
                    'features': str(b.get('features', [])),
                    'model': b.get('model', '?'),
                    'bias': round(
                        b.get('r2', 0) - b.get('nested_r2', 0), 4),
                    'bootstrap_ci_low': round(
                        b.get('bootstrap_ci', {}).get('ci_low', 0), 4)
                        if b.get('bootstrap_ci') else None,
                })
        log_df = pd.DataFrame(log_rows)
        # Append 模式：如果檔案存在就追加，不存在就新建
        if os.path.exists(EXPERIMENT_LOG_PATH):
            log_df.to_csv(EXPERIMENT_LOG_PATH, mode='a',
                          header=False, index=False)
        else:
            log_df.to_csv(EXPERIMENT_LOG_PATH, index=False)
        print(f"✓ 實驗紀錄已追加至: {EXPERIMENT_LOG_PATH} "
              f"({len(log_rows)} rows)")
    except Exception as e:
        print(f"⚠ 實驗紀錄寫入失敗: {e}")
```

### 驗證

跑完後 Drive 上出現 `experiment_log.csv`，包含 12 行（6 策略 × 2 廠區）。
第二次跑完後追加 12 行（共 24 行），可用 pandas 讀取比較跨版本差異。

---

## M5：feature_stats OOD 檢查

### 問題

預測時如果新數據的 XRD 分布超出訓練範圍，模型會靜默給出錯誤預測。

### 修改——特徵分析腳本

在 pkl 的 save_data 中新增（L2044 附近）：

```python
        # ★ MLOps M5: 訓練數據特徵統計（供 OOD 檢測）
        'feature_stats': {
            side: {
                f: {
                    'mean': float(fd[f].mean()),
                    'std': float(fd[f].std()),
                    'min': float(fd[f].min()),
                    'max': float(fd[f].max()),
                }
                for f in fd.columns
                if f in fd.select_dtypes(include=[np.number]).columns
                and fd[f].notna().sum() > 5
            }
            for side, fd in [('hp', hp_data), ('sa', sa_data)]
        },
```

### 修改——預測腳本

在 `cement_predict_v6_4.py` 中新增 OOD 檢查函數：

```python
def check_ood(features_dict, data, factory, threshold=3.0):
    """
    檢查輸入特徵是否超出訓練分布。

    Returns
    -------
    list of str — 警告訊息（空 = 正常）
    """
    stats = data.get('feature_stats', {}).get(factory, {})
    if not stats:
        return []

    warnings = []
    for f, v in features_dict.items():
        if f in stats and stats[f]['std'] > 0:
            s = stats[f]
            z = abs(v - s['mean']) / s['std']
            if z > threshold:
                warnings.append(
                    f"  ⚠ {f}={v:.2f} 超出訓練範圍 "
                    f"(z={z:.1f}, 訓練: {s['mean']:.2f}±{s['std']:.2f})")
            elif v < s['min'] * 0.9 or v > s['max'] * 1.1:
                warnings.append(
                    f"  ⚠ {f}={v:.2f} 接近訓練邊界 "
                    f"(訓練範圍: [{s['min']:.2f}, {s['max']:.2f}])")
    return warnings
```

在 `predict_single` 的 L95（點預測之前）加入：

```python
    # ★ OOD 檢查
    ood_warnings = check_ood(features_dict, data, factory)
    if ood_warnings:
        result['ood_warnings'] = ood_warnings
```

在 `format_prediction` 中加入顯示：

```python
    if result.get('ood_warnings'):
        lines.append("分布異常警告:")
        for w in result['ood_warnings']:
            lines.append(w)
```

### 驗證

用一個極端值（例如 `C3S=95`，正常範圍約 55-75）呼叫 predict_single，
應該看到 OOD 警告。正常值不應觸發。

---

## M6：測試 Notebook

### 新增檔案

建立 `/content/drive/MyDrive/台泥/AI/test_v6.4.ipynb`
（或同等的 .py 腳本）。

以下為 notebook cell 內容：

```python
# ═══ Cell 1: Setup ═══
import sys
sys.path.insert(0, '/content/drive/MyDrive/台泥/AI/')

from cement_feature_analysis_v6_4 import (
    compute_trust_grade,
    calculate_features,
    build_quantile_deviation_model,
    _build_model,
    build_pool,
    TASKS,
    TRUST_LABELS,
    MODEL_ZOO_FULL,
)

import pandas as pd
import numpy as np
passed = 0
failed = 0


# ═══ Cell 2: Trust Grade 邊界測試 ═══
def test_trust_grade():
    global passed, failed
    tests = [
        # (bias, nested_r2, ci_low, expected_grade)
        (0.03, 0.80, 0.50, 'A'),   # 低偏差 + 高 R² + CI > 0
        (0.04, 0.25, 0.05, 'B'),   # 低偏差但低 R²
        (0.10, 0.50, 0.10, 'B'),   # 中偏差
        (0.14, 0.21, -0.15, 'B'),  # 邊界 B
        (0.20, 0.40, 0.05, 'C'),   # 高偏差
        (0.05, 0.35, -0.10, 'C'),  # CI 偏低但不到 D
        (0.02, -0.10, 0.10, 'D'),  # 負 R²
        (0.05, 0.30, -0.50, 'D'),  # CI 太低
        (0.01, 0.80, -0.45, 'D'),  # 其餘好但 CI 崩潰
    ]
    for bias, r2, ci_low, expected in tests:
        ci = {'ci_low': ci_low}
        result = compute_trust_grade(bias, r2, ci)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ trust_grade({bias}, {r2}, {ci_low}) "
                  f"= {result}, expected {expected}")
    print(f"Trust Grade: {len(tests)} tests, "
          f"{passed} passed, {failed} failed")

test_trust_grade()


# ═══ Cell 3: 特徵工程回歸測試 ═══
def test_features():
    global passed, failed
    row = pd.DataFrame([{
        'C3S': 60, 'C2S': 15, 'C4AF': 10,
        'C3A_cubic': 5, 'C3A_ortho': 3,
        'FreeLime': 1.0, 'Calcite': 2.0,
        'Gypsum': 3.0, 'Bassanite': 1.0, 'Anhydrite': 0.5,
        '比表面積': 3500, '濕式f-CaO': 1.2,
        '3天': 25.0, '7天': 32.0, '28天': 42.0,
    }])
    r = calculate_features(row)

    checks = [
        ('C3A_total', 8.0),
        ('Sulfate_total', 4.5),
        ('Silicate_total', 75.0),
        ('Hydraulic_potential',
         60*1.0 + 15*0.2 + 5*0.8 + 3*0.4 + 10*0.15),
        ('Heat_est',
         136*60/100 + 62*15/100 + 200*8/100 + 30*10/100),
        ('Burning_degree', 60 / (60 + 15 + 0.001)),
        ('Strength_gain_3to7', 7.0),
    ]
    for name, expected in checks:
        actual = r[name].iloc[0]
        if abs(actual - expected) < 0.01:
            passed += 1
        else:
            failed += 1
            print(f"  ✗ {name} = {actual:.4f}, expected {expected:.4f}")

    # C-1: 沒有 K2O → 不應生成 Na2Oeq
    if 'Na2Oeq' not in r.columns:
        passed += 1
    else:
        failed += 1
        print("  ✗ Na2Oeq should not exist without K2O")

    # C-2: 沒有 C3S_M1 → 不應生成 C3S_M1_ratio
    if 'C3S_M1_ratio' not in r.columns:
        passed += 1
    else:
        failed += 1
        print("  ✗ C3S_M1_ratio should not exist without C3S_M1")

    print(f"特徵工程: {len(checks)+2} checks, "
          f"{passed} passed, {failed} failed")

test_features()


# ═══ Cell 4: _build_model 工廠函數測試 ═══
def test_build_model():
    global passed, failed
    for name in ['Ridge_0.5', 'Ridge_1.0', 'Ridge_5.0',
                 'ENet_0.5', 'XGB_Small', 'GB_Stable']:
        if name in MODEL_ZOO_FULL:
            try:
                m = _build_model(name,
                                 MODEL_ZOO_FULL[name]['params'])
                # 確認能 fit 和 predict
                X = np.random.randn(10, 3)
                y = np.random.randn(10)
                m.fit(X, y)
                pred = m.predict(X[:1])
                assert len(pred) == 1
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  ✗ _build_model('{name}'): {e}")
    print(f"_build_model: {passed} passed, {failed} failed")

test_build_model()


# ═══ Cell 5: Conformal 覆蓋率保證測試 ═══
def test_conformal():
    global passed, failed
    np.random.seed(42)
    for trial in range(3):
        n = 40
        X = np.random.randn(n, 2)
        y = X[:, 0] * 3 + X[:, 1] * 1.5 + np.random.randn(n) * 2
        fd = pd.DataFrame(X, columns=['f1', 'f2'])
        fd['target'] = y
        result = build_quantile_deviation_model(
            fd, ['f1', 'f2'], 'target', label=f'test_{trial}')
        if result and result['calibrated_coverage'] >= 0.75:
            passed += 1
        else:
            cov = result['calibrated_coverage'] if result else 0
            failed += 1
            print(f"  ✗ trial {trial}: coverage={cov:.1%}")
    print(f"Conformal: 3 trials, "
          f"{passed} passed, {failed} failed")

test_conformal()


# ═══ Cell 6: build_pool 過濾測試 ═══
def test_build_pool():
    global passed, failed
    # 建一個 mock DataFrame
    fd = pd.DataFrame({
        'C3S': [60]*10, 'C2S': [15]*10, 'C4AF': [10]*10,
        'C3A_cubic': [5]*10, 'C3A_ortho': [3]*10,
        'FreeLime': [1]*10, 'Calcite': [2]*10,
        'Gypsum': [3]*10, 'Bassanite': [1]*10, 'Anhydrite': [0.5]*10,
        '比表面積': [3500]*10, '濕式f-CaO': [1.2]*10,
        '3天': [25]*10, '7天': [32]*10, '28天': [42]*10,
        # 以下是 calculate_features 會生成的欄位
        'C3A_total': [8]*10, 'SSA_C3S': [210]*10,
        'Na2Oeq': [None]*10,  # 全 NaN → 應被過濾
    })

    pool = build_pool(fd, '28天', include_strength=False, side='hp')
    # Na2Oeq 全 NaN → 不應出現在 pool 中
    if 'Na2Oeq' not in pool:
        passed += 1
    else:
        failed += 1
        print("  ✗ Na2Oeq should be filtered (all NaN)")

    # 28天_含3天: allowed_strength 應排除 7天
    pool_3d = build_pool(fd, '28天', include_strength=True,
                          side='hp',
                          allowed_strength=['3天', 'Str3_EffBurn'])
    if '7天' not in pool_3d:
        passed += 1
    else:
        failed += 1
        print("  ✗ 7天 should be excluded by allowed_strength")

    if '3天' in pool_3d:
        passed += 1
    else:
        failed += 1
        print("  ✗ 3天 should be in pool with allowed_strength")

    print(f"build_pool: 3 checks, "
          f"{passed} passed, {failed} failed")

test_build_pool()


# ═══ Cell 7: 總結 ═══
print(f"\n{'=' * 50}")
print(f"總計: {passed} passed, {failed} failed")
if failed == 0:
    print("✓ 全部通過")
else:
    print(f"✗ {failed} 項失敗，請檢查")
print(f"{'=' * 50}")
```

### 使用方式

在 Colab 中新建一個 notebook，貼入上述 cells，執行即可。
全部測試應在 15 秒內完成。每次改版前跑一次確認沒有回歸。

---

## 不修改的部分

- 所有 v6.4 的計算邏輯（Nested LOO, Trust Grade, SHAP, Optuna, MTL, MAML）
- calculate_features 的公式
- build_pool 的過濾規則
- 主迴圈的執行順序
- 預測腳本的 Grade 分層邏輯

---

## 依賴關係

```
M1 (if __name__) ── 最先做，M6 依賴它
M2 (數據指紋) ───── 獨立，M4 需要 _data_hash
M3 (config.yaml) ── 獨立
M5 (OOD 檢查) ──── 獨立
M4 (實驗紀錄) ──── 依賴 M2 的 _data_hash
M6 (測試 notebook) ─ 依賴 M1 的 if __name__ 保護
```

M1/M2/M3/M5 可平行開發。M4 等 M2 完成。M6 等 M1 完成。
