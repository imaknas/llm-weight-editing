# 設計:LLM 讀/改小型網路權重的複現實驗

## 目標
回答兩個問題(MIT Weight-Space Programming 的小規模複現,外加對照組):
1. **讀**:LLM 只看權重文字,能否判斷網路學了什麼?還是只是靠 prompt 脈絡猜?
2. **改**:給 LLM 工具(forward pass 次數上限、輸出權重 diff),能否修掉網路的局部缺陷?

## 目前進度
- 「讀」實驗已就緒:`wse/read_exp.py`,對照組如下表。純標準庫,訓練約 6 秒。
- 「改」實驗已寫好(`wse/edit_exp.py`),尚未用 LLM 跑過。

| kind | 內容 | 期望答案 | 意義 |
|---|---|---|---|
| trained | 訓練好的網路 | 真正的任務 | 基本能力 |
| permuted | 隱藏單元重排(函數完全相同) | 真正的任務 | 排除「記住特定數字排列」,測是否真讀懂結構 |
| shuffled | 權重值整批打亂 | none | 函數被破壞,數值分佈相同;若仍答任務 = 靠脈絡猜 |
| random | 沒訓練,偏置也隨機 | none | 權重幅度仍明顯偏小,只能當寬鬆對照 |
| randlabel | 同樣流程訓練在隨機標籤上 | none | 權重像訓練過但不對應任何規則;none 類中最嚴格 |

若只有 trained/permuted 高、shuffled/random/randlabel 也答對 none,才算有讀到權重內容的證據。
none 類對照不能留表面破綻:最初的 random 偏置全為 0,LLM 靠這點就能認出來,那一輪的 random 結果不算數。
另外要記得有 4+1 個選項,隨機猜的基準約 20%。

## 結構
- `wse/mlp.py` 純 Python MLP(訓練、排列、打亂、序列化、轉文字)
- `wse/tasks.py` 四個 2D 分類任務
- `wse/llm.py` `LLM` 協定;`BrunnaCompleteLLM`(正式)、`ScriptedLLM`(測試)
- `wse/read_exp.py` 案例產生、prompt、評分;CLI 入口 `main()` 是唯一組裝具體 LLM 的地方
- `tests/test_wse.py` 單元測試(`python3 -m unittest discover -s tests -t .`)

## 與 MIT 原實驗的差異(要誠實講)
- 網路約 205 個權重(2-12-12-1),比他們的 1.7 萬小很多;純 Python 訓練與 prompt 長度的取捨。
  之後有 torch 可放大。
- 任務是 2D 分類,不是遊戲策略;好處是函數可精確驗證。
- 每種任務只有 2 個種子,樣本很小,第一輪只能當 smoke test,不能下統計結論。

## 未驗證 / 風險
- `BrunnaCompleteLLM` 對 `brunna complete` 輸出格式(`structured_output`、`llm_error`)的解析是依平台說明寫的,
  還沒真正呼叫過。
- 需要先宣告 `models`(見 `registry-models-draft.yaml`)才能跑。
- 小網路上 tanh 單元的權重可能有多種解法,即使正確答案存在,LLM 也不一定能從 205 個數字算出來;
  這是實驗要量的東西,不是 bug。

## 下一步
1. 宣告 models → 跑 `run`,看四種 kind 的正確率。
2. 寫「改」實驗:訓練後在某區域(例如某象限)注入缺陷,LLM 輸出新權重或 diff,量修復率與改動比例。
3. PyPI 開通後用 torch 放大網路。
