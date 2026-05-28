# 水泥 XRD 晶相強度預測系統 — 技術架構說明

## 1. 系統總覽

本系統從 XRD Rietveld 定量分析結果出發，經過特徵工程、自動特徵選擇、模型訓練與驗證、可解釋性分析、不確定性量化五個階段，最終輸出帶有信任等級的強度預測。系統架構分為兩個主要元件：特徵分析腳本（離線訓練）和預測腳本（線上推論）。

數據規模：和平廠 40 筆、蘇澳廠 31 筆。預測目標：3 天、7 天、28 天抗壓強度（MPa）。

---

## 2. 數據流與管線架構

```
Excel (XRD + 強度) 
  → calculate_features（特徵工程，42+ 候選特徵）
    → build_pool（特徵池過濾）
      → greedy_forward（Ridge LOO 逐步選擇）
        → Full Nested LOO（特徵 + 模型都在內層選擇）
          → trust_mode 判定（A/B/C）
            → 防過擬合特徵裁剪
              → Optuna 超參數優化
                → SHAP + Permutation + Interaction 三重驗證
                  → 偏差量化 + Conformal 校準
                    → Trust Grade 分級
                      → STRATEGIES + pkl 存檔
```

---

## 3. 特徵工程

### 3.1 原始 XRD 特徵（10 個）

C3S, C2S, C4AF, C3A_cubic, C3A_ortho, FreeLime, Calcite, Gypsum, Bassanite, Anhydrite。這些來自 Rietveld 精修的晶相含量（wt%）。

### 3.2 工程特徵設計原理

特徵工程基於 Portland 水泥化學的 Bogue 計算體系和水化反應動力學。

氧化物估算：從晶相含量反推 CaO, SiO₂, Al₂O₃, Fe₂O₃ 估計值。例如 `CaO_est = C3S×0.737 + C2S×0.651 + C3A_total×0.623 + C4AF×0.462 + FreeLime + Calcite×0.56`，基於各晶相的化學計量比。

水泥模數：LSF（石灰飽和因子）、IM（鐵率）、SM（矽率）是水泥工業標準的品質控制指標，直接影響熟料中各晶相的比例 [1]。

比表面積交互項：`SSA_C3S = 比表面積 × C3S / 1000`。比表面積放大了 C3S 的水化反應速率，兩者的乘積比單獨使用任一指標更能預測早期強度 [2]。v6.4 的 SHAP 分析驗證了 SSA_C3S 在 3 天和平預測中佔 76.5% 的重要性。

C3A 品質指標：`C3A_quality = C3A_cubic / (C3A_cubic + 1.5×C3A_ortho)`。Cubic C3A 的水化反應速率高於 orthorhombic C3A，且受硫酸鹽/鹼比控制 [2]。權重 1.5 反映了 ortho 對早期強度的負面效應。

水化熱估算：`Heat_est = 136×C3S/100 + 62×C2S/100 + 200×C3A_total/100 + 30×C4AF/100`，基於 Bogue 計算的各晶相水化熱係數（cal/g）[1]。

硫酸鹽平衡：`SO3_deficit = max(0, 0.8×C3A_total − Sulfate_total)`。當硫酸鹽不足時，C3A 會發生不受控水化（flash set），影響強度發展 [3]。

### 3.3 前向兼容特徵

SO₃/Alkali 比（C-1）：`SO3_Alkali_ratio = Sulfate_total / Na2Oeq`。研究表明硫酸鹽-鹼比顯著影響 C3A 含量和多晶型，以及 Alite 晶體大小 [2]。僅在 Excel 中存在 K₂O 和 Na₂O 欄位時計算。

C3S 多晶型比例（C-2）：`C3S_M1_ratio = C3S_M1 / (C3S_M1 + C3S_M3)`。C3S 的 M1 多晶型在較低溫度下形成，影響早期水化行為和 portlandite 生成 [4]。C3S 的形成受硫酸鹽/鎂比和 Na₂Oeq 含量控制（而非 LSF），多晶型受硫酸鹽/(鎂+鹼) 比控制 [2]。僅在未來 Rietveld 精修區分 M1/M3 後生效。

---

## 4. 驗證框架

### 4.1 Full Nested LOO-CV

傳統的 K-fold 交叉驗證在小樣本下會產生嚴重的樂觀偏差。研究證實，K-fold CV 在小樣本下效能估計偏差顯著，而 Nested CV 和 train/test split 不論樣本大小都能產生穩健且無偏的效能估計 [5]。2026 年的 bias-resistant framework 進一步指出，傳統 CV 同時用於模型選擇和效能估計時，可能膨脹效能指標超過 20% [6]。

本系統的 Full Nested LOO 在每個外層迭代中完整執行三步：

Step 1：相關性預過濾（保留 top 15 特徵，降低內層搜索空間）
Step 2：在 train set 上執行 inner greedy forward（Ridge，最多 4 步，增量 R² < 0.005 則停止）
Step 3：在選出的特徵上遍歷 6 個模型候選（Ridge, XGB, GB, CatBoost_Ordered, RF, ElasticNet），選最佳

輸出 Nested R² 作為真正無偏的泛化估計，同時統計特徵組合穩定性（各 fold 選出的特徵是否一致）和內層模型分布。

### 4.2 Bootstrap 置信區間

200 次 bootstrap 重抽樣計算 R² 的 95% 置信區間。OOB（Out-Of-Bag）最小樣本數設為 max(5, n×0.15)，避免 OOB 過少導致 R² 無統計意義。CI_low（下界）是否 > 0 是判定模型是否穩定的關鍵閾值。

### 4.3 Trust Grade 分級

綜合三個指標判定四級信任等級：

Grade A（可部署）：bias < 0.05 且 Nested R² > 0.3 且 CI_low > 0
Grade B（可使用）：bias < 0.15 且 Nested R² > 0.2 且 CI_low > -0.2
Grade D（不可靠）：Nested R² < 0 或 CI_low < -0.4
Grade C（實驗性）：其餘

v6.4 結果中僅 7 天含 3 天蘇澳達到 Grade A（Nested R² = 0.934, bias = 0.022, CI = [0.882, 0.981]）。

### 4.4 防過擬合自動選模

基於 bias 值觸發三種模式：

Mode A（bias < 0.05）：直接採用 greedy 最佳配置。
Mode B（0.05-0.15）：從候選中篩選特徵數 ≤ 3 的最佳。
Mode C（≥ 0.15）：從 Nested LOO 特徵穩定性取 top-1 組合並裁剪到 ≤ 3。

此設計確保了最終報告的 R² 是基於 Nested LOO 的無偏估計，而非虛高的標準 LOO R²。

---

## 5. 模型選擇

### 5.1 MODEL_ZOO 構成

Ridge（alpha=0.5/1.0/5.0）、ElasticNet（l1_ratio=0.5/0.8）、RandomForest、XGBoost、GradientBoosting、CatBoost（含 Ordered boosting）、LightGBM。

所有模型通過 `_build_model(name, params)` 工廠函數按需創建，避免 CatBoost 的 deepcopy 效能損失。

### 5.2 Ridge 主導的原因

v6.4 結果中 Ridge 被選為最佳模型 10/12 次。在本系統的小樣本（30-40 筆）、高維度（42 候選特徵）場景下，Ridge 的 L2 正則化能穩定共線性嚴重的特徵矩陣。水泥 ML 研究的系統性回顧指出超過 55% 的相關研究樣本量不到 200 筆，小樣本被認為不足以支撐複雜模型 [7]。CatBoost 的 ordered boosting 雖然理論上抗過擬合，但 30 筆數據的排列組合太少，其優勢無法體現。

### 5.3 Optuna 貝氏超參數優化

在最終選定的特徵上，用 Optuna 在 LOO 內做貝氏搜索。搜索空間涵蓋 model_type（ridge/xgb/gb/catboost/lgbm/elasticnet）和各模型特定的超參數。如果 Optuna R² 比固定 Zoo 高 0.005 以上，自動回寫到 STRATEGIES。v6.4 中 28 天 XRD 和平的 Optuna_ridge（alpha=0.103）比固定 Ridge_0.5 提升了 +0.023。

---

## 6. 可解釋性分析

### 6.1 SHAP（SHapley Additive exPlanations）

SHAP 基於合作博弈論的 Shapley 值，量化每個特徵對單一預測的邊際貢獻 [8]。本系統使用 cross-validated SHAP：在 n ≤ 60 的場景下，每個 LOO fold 訓練模型後只計算留出那一筆的 SHAP 值，最後拼接成完整矩陣。這避免了在全量資料上 fit 後計算 SHAP 可能反映過擬合模式的問題。

輸出包括全域特徵重要性排序（平均絕對 SHAP 值）和 Beeswarm/Bar plot。

### 6.2 SHAP Interaction Values（代理模型）

由於最佳模型通常是 Ridge（不支援 TreeExplainer），系統用 GradientBoostingRegressor 作為「解釋代理」計算 interaction values。只在代理 R² ≥ 基礎模型 R² × 85% 時才信任。v6.4 中揭示的 top 互動效應包括：SSA_C3A × Calcite（蘇澳 3 天，0.2254）和 7 天 × Str7_C2S（和平 28 天主策略，0.1603）。

### 6.3 Permutation Importance

使用 5-fold CV 的 permutation importance 作為 SHAP 的交叉驗證。與 SHAP 的 top-N 比較（N = min(5, n_features)）判定排序一致性。v6.4 結果中 11/11 個組合達到排序一致（修正 E 後）。

### 6.4 三重驗證的意義

SHAP、Permutation、Interaction 三個完全不同的計算路徑得出一致的特徵重要性排序，極大增強了結果的可信度。如果三者不一致，通常意味著模型對特徵關係的學習不穩定，是過擬合的信號。

---

## 7. 不確定性量化

### 7.1 分位數回歸 + Conformal 校準

對 28 天策略，使用 GradientBoostingRegressor 的 `loss='quantile'`（alpha=0.10/0.50/0.90）在 LOO 下計算三個分位數預測，構成 80% 預測區間。

原始分位數回歸在小樣本下區間過窄（覆蓋率 64-74%）。系統引入 Conformal Prediction 校準：計算 conformity scores `s_i = max(q10_i − y_i, y_i − q90_i)`，取其 `ceil((1−α)(n+1)/n)` 分位數作為校準幅度 Q，將區間膨脹為 `[q10 − Q, q90 + Q]`。

這基於 Conformal Prediction 的有限樣本有效性理論。NeurIPS 2024 的 Self-Calibrating Conformal Prediction 將 Venn-Abers 校準從分類擴展到回歸，能在有限樣本下同時提供校準的點預測和具有有限樣本有效性的預測區間 [9]。AAAI 2025 的 Conformal Thresholded Intervals (CTI) 進一步提出用多輸出分位數回歸直接建構最小預測集 [10]。

v6.4 結果：校準後覆蓋率全部達到 82-84%，conformal margin 在 0.42-0.79 MPa 之間。

### 7.2 偏差量化指標

五個核心指標：

覆蓋率：80% 區間包含多少比例的實際值（目標 ≥ 80%）。
平均區間寬度：區間的平均 MPa 寬度（越窄越好，但不能犧牲覆蓋率）。
偏差方向正確率：模型預測的偏差方向（高/低於廠區平均）和實際一致的比例（> 50% 才有價值）。
偏差相關性：偏差分數和實際偏差的 Pearson r。
極端值辨識力：模型標記為「偏低 20%」的批次，實際平均強度比其他批次低多少 MPa（> 1 MPa 為可辨識）。

蘇澳 28 天 XRD 雖然 Grade D（Nested R² = 0.336），但方向正確率 81%、偏差相關性 0.694、極端值辨識力 +3.64 MPa，對混合決策有顯著實務價值。

---

## 8. Multi-Task Learning

使用 `sklearn.multioutput.MultiOutputRegressor(Ridge)` 同時預測 3 天/7 天/28 天強度，共享 XRD 特徵空間。

MTL 的理論基礎是任務間的共享結構：三個齡期的強度來自同一套水泥晶相的水化過程，物理上存在因果鏈（C3S 水化 → 3 天 → 繼續水化 → 7 天 → C2S 參與 → 28 天）。研究表明 MTL 能利用不同目標之間的共享資訊提升各個目標的預測精度 [11]。

v6.4 結果顯示弱任務獲益最大：和平 7 天從 0.342 提升到 0.496（+0.154），和平 28 天從 0.152 到 0.221（+0.068）。強任務（蘇澳 3 天 0.793）幾乎不受影響（+0.007）。

---

## 9. MAML 跨廠區遷移（實驗性）

Ridge-MAML 將每個（廠區, 齡期）組合視為一個 task，學習跨 task 的共享權重初始化，然後每個 task 用 3 步梯度下降適應。基於 MAML（Model-Agnostic Meta-Learning）框架 [12]，2025 年研究已證明 few-shot meta-learning 在資料受限的混凝土強度預測中的有效性 [13]。

v6.4 結果全面退步（-0.10 到 -0.38），根本原因是兩廠 XRD 分布差異太大（不同原料、窯型、混合比例），共享初始化造成負遷移。標記為「不適用於當前兩廠場景」，保留代碼待未來廠區數量增加時重新評估。

---

## 10. 部署架構

### 10.1 分層預測

預測腳本根據 Trust Grade 自動決定輸出格式：

Grade A：80% 預測區間，無警告。
Grade B：90% 預測區間，無警告。
Grade C：95% 預測區間 + 「僅供參考」警告。
Grade D：不輸出點預測，只給偏差方向和 conformal 校準後的區間。

### 10.2 OOD 檢查

預測時檢查每個輸入特徵的 z-score 是否 > 3（超出訓練分布）以及是否低於 min×0.9 或高於 max×1.1（接近訓練邊界）。警告自動附加到預測結果中。

### 10.3 MLOps 基礎設施

數據指紋：每次載入 Excel 後計算 MD5 hash，存入 pkl，確保模型和數據版本綁定。
pkl 版本化：檔名帶指紋和時間戳，同時維護一個 latest 副本。
實驗紀錄：每次跑完追加到 experiment_log.csv，支持跨版本追蹤比較。
配置分離：config.yaml 管理路徑和超參數，不存在時 fallback 到預設值。
測試腳本：46 項自動化測試覆蓋核心函數的正確性。

---

## 11. 已知限制

28 天 XRD-only 預測能力不足（和平 Nested R² = 0.152，蘇澳 0.336），根本原因是 XRD 晶相和 28 天強度之間的因果鏈太長。水泥成品由不同製程批次的熟料混合磨細而成，batch-level 的製程參數在混合後被平均掉，XRD 測到的是「混合後的平均結晶相組成」，已是距離 28 天強度最近的可測量信號。

樣本量限制（和平 40 筆、蘇澳 31 筆）導致 tree-based 模型無法發揮優勢，Ridge 壓倒性勝出。當樣本累積到 80-100 筆時應重新評估模型選擇。

ElasticNet 僅在 28 天含 3 天蘇澳的 Nested LOO 內層被大量選中（13/31 次），其餘場景 Ridge 仍然最優。

---

## 參考文獻

[1] Taylor, H.F.W. (1997). Cement Chemistry, 2nd ed. Thomas Telford Publishing. — Portland 水泥化學基礎，Bogue 計算，水化熱係數。

[2] Andrade Neto, J.S., Carvalho, I., Monteiro, P.J.M., & Kirchheim, A.P. (2025). "Unveiling the key factors for clinker reactivity and cement performance: A physic-chemical and performance investigation of 40 industrial clinkers." Cement and Concrete Research, 187, 107726. — 40 個工業熟料的系統性研究，硫酸鹽-鹼比對 C3A 多晶型和 Alite 晶體大小的影響，Alite 晶體大小對早期強度的負面效應。

[3] Kamakshi, T.A., et al. (2024). "Impact of particle size, clinker mineralogy and sulfate availability on early cement hydration." Cement and Concrete Research, 184, 107611. — 硫酸鹽供應不足導致 C3A 不受控水化和 C3S 水化延遲。

[4] Andrade Neto, J.S., et al. (2024). C3S 多晶型研究：M1 多晶型與 portlandite 生成的關聯，C3S 形成受硫酸鹽/鎂比和 Na₂Oeq 控制。（見 [2] 同一研究組）

[5] Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A.J. (2019). "Machine learning algorithm validation with a limited sample size." PLOS ONE, 14(11), e0224365. — 證實 K-fold CV 在小樣本下產生嚴重樂觀偏差，Nested CV 和 train/test split 產生穩健無偏估計。

[6] Nawabi, J., et al. (2026). "A Reproducible Framework for Bias-Resistant Machine Learning on Small-Sample Data." arXiv:2602.02920. — 嚴格 nested CV 框架，傳統 CV 可能膨脹效能指標超過 20%，提出 leakage-controlled 驗證流程。

[7] 多篇水泥 ML 系統性回顧研究，包括 Concrete ML 領域超過 55% 的研究樣本量不到 200 筆。

[8] Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." Advances in Neural Information Processing Systems 30 (NeurIPS 2017). — SHAP 框架原始論文。

[9] van der Laan, L., et al. (2024). "Self-Calibrating Conformal Prediction." Advances in Neural Information Processing Systems 37 (NeurIPS 2024). arXiv:2402.07307. — 將 Venn-Abers 校準從分類擴展到回歸，提供校準點預測和具有限樣本有效性的條件預測區間。

[10] Luo, R., et al. (2025). "Conformal Thresholded Intervals for Efficient Regression." Proceedings of AAAI 2025. arXiv:2407.14495. — 多輸出分位數回歸建構最小預測集，透過 conformity scores 自動調整區間寬度達到目標覆蓋率。

[11] 多篇 MTL 應用於混凝土強度預測的研究，包括：Optimization and predictive performance of fly ash-based sustainable concrete using integrated multitask deep learning framework. Scientific Reports 15, 30820 (2025). — MTL 框架結合 XGBoost、DNN 和 AutoGluon，同時預測抗壓和抗拉強度，R² = 0.91。

[12] Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." Proceedings of the 34th International Conference on Machine Learning (ICML 2017). — MAML 原始論文。

[13] Few-shot meta-learning for concrete strength prediction, AI in Civil Engineering (2025). — MAML 在混凝土標準數據集的小子集（低至 50 筆）上驗證了 meta-learning 在資料受限場景的有效性。
