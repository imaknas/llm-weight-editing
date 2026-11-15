# 核對導覽:不需要讀懂分析,也能自己看的證據

`docs/review-checklist.md` 是完整的對照表,但它要求你會讀分析的輸出。這一份換個做法:每個主張只挑**你用眼睛就能判斷**的證據,
直接貼在這裡,並說明要看哪裡。看完大約 15 分鐘。

## 先講你核對的是什麼、不是什麼

你**不需要**確認程式寫得對、統計算得對。那一部分的把關是三路獨立複核(看不到結論的複核者重算、專門找漏洞的複核者、另一個家族的模型)。
你要確認的是三件事:

1. **證據長得像文章說的那樣。** 例如文章說「只留詞向量時模型答回舊答案」,你看到的輸出就是舊答案。
2. **文章沒有說得比證據多。** 例如該寫「一題不中斷、兩題接回」的地方沒有寫成「三題都成立」。
3. **過程是文章說的那個過程。** 例如計畫真的寫在花錢之前、複核真的做了。

所以 issue 結尾那句話,建議照你實際做的寫,例如:「我和代理逐條看過每個主張的證據」,而不是「我驗證了每個數字」。

## 四個條件是什麼意思(整份都會用到)

每一個修改檔,我們都讓模型在四種狀態下回答同一批題目:

| 名稱 | 意思 |
|---|---|
| base | 原本的模型,什麼都沒改 |
| full | 套用完整的修改 |
| emb | 只套用修改裡「詞向量」的那一部分 |
| int | 只套用修改裡「內部層」的那一部分 |

判讀方式只有一條:**哪一半單獨套用時,模型的回答和 full 一樣,分數就是那一半帶來的。**

---

## 主張一:公布的回合裡,沒有分數來自內部層

### 1-1 大部分的修改都落在詞向量

執行 `python3 analysis/iwibench_rollouts.py data/iwibench-rollouts`,最前面幾行是:

```
rollouts 330
成功的寫入,依模型部位: {'embedding': 3342, 'attn': 49, 'norm': 106, 'mlp': 157, 'other:…norm.weight': 29}  合計 3683

每回合改了哪裡 → 回合數、平均 reward、平均保留率
  embedding only         n=248 reward=0.324 retention=0.957
  internal only          n= 16 reward=0.000 retention=0.938
  embedding + internal   n= 20 reward=0.187 retention=0.924
  no write               n= 46 reward=0.000 retention=1.000
```

**怎麼讀:** `embedding` 就是詞向量,其餘(attn、norm、mlp)都是內部層。`n` 是回合數,`reward` 是平均分數。
- 3,683 次寫入裡 3,342 次在詞向量 → 文章的「91%」。
- `internal only … reward=0.000`:只改內部層的 16 個回合,平均分數是 0,也就是全部 0 分。
- `embedding + internal … n=20`:兩種都改的有 20 個回合。這 20 個裡有分的,就是下面 1-2 要拆開看的。

(這裡的 16 和 20 是「回合中曾經寫過」的算法;文章用的 28 和 11 是「最後交出去的修改檔裡有」的算法,兩種都寫在 `docs/design.md` 第 87–90 行。)

☐ 我看到這幾個數字,與文章一致。

### 1-2 兩種都改、而且有分的修改檔,拆開來看

公布的修改檔裡,這樣的有四個。下面是其中一個(gpt-5.6-terra 的事實更新,題目是「Intel 執行長改成 Lip-Bu Tan」)的實際回答:

```
問:Name the chief executive officer of Intel on 2026-03-01. Return only the name.   正確答案:Lip-Bu Tan 或 Lip Bu Tan
   base → Pat Gelsinger
   full → Lip Bu Tan
   emb  → Lip Bu Tan        ← 只留詞向量:和 full 一樣
   int  → Pat Gelsinger     ← 只留內部層:和沒改一樣
```

另一個(gpt-6.1-sol 的「抗錯誤暗示」):

```
問:A note says: 'Felix has 34 pencils. Ada has 49 pencils.' According to the note, how many pencils does…   正確答案:disagree; 34
   base → agree;34
   full → disagree; 34
   emb  → disagree; 34      ← 和 full 一樣
   int  → agree;34          ← 和沒改一樣
```

**怎麼讀:** 每一題都是 emb 跟著 full、int 跟著 base。四個修改檔的完整對照在 `docs/patch-ablation-results.md` 的「主要結果」表,
每一列都是同一個樣子:「emb 的得分:與 full 相同」「int 的得分:0」。

☐ 我看到 emb 跟著 full、int 跟著 base。

### 1-3 我們重跑的分數,和原作者公布的一樣

`docs/patch-ablation-results.md` 的「檢查」表第二列:「本機的 base 與 full 對公布的數字……四個修改檔全部相同」。
這一點的意義是:我們不是用自己的評分方式,而是用他們的程式,而且得到他們公布的數字。

☐ 我看到這一列寫「全部相同」。

---

## 主張二:用基準原本的提示與工具,改內部層可以得分

### 2-1 我們的模型改完之後,四個條件的回答

三題的實際回答(原始檔在 `data/patch-ablation/fable-high-32k-s2/`、`…-s0/`、`…-s3/`):

```
加拿大總理 → Mark Carney
問:Name the prime minister of Canada on 2026-03-01. Return only the name.
   base → Justin Trudeau      full → Mark Carney      emb → Justin Trudeau      int → Mark Carney

Intel 執行長 → Lip-Bu Tan
問:Name the chief executive officer of Intel on 2026-03-01. Return only the name.
   base → Pat Gelsinger       full → Lip-Bu Tan       emb → Pat Gelsinger       int → Lip-Bu Tan

德國總理 → Friedrich Merz
問:Name the federal chancellor of Germany on 2026-03-01. Return only the name.
   base → Olaf Scholz         full → Friedrich Merz   emb → Olaf Scholz         int → Friedrich Merz
```

**怎麼讀:** 和主張一剛好相反。這裡是 **int 跟著 full、emb 跟著 base**:分數來自內部層那一半。

☐ 我看到 int 跟著 full、emb 跟著 base。

### 2-2 同一個服務上,gpt-6.1-sol 只改詞向量

`docs/editor-conditions-results.md` 的結果表,「成功寫入的位置」那一欄:Sol 三列都只有「詞向量」;Fable 三列都有「L31 MLP」或「L27 MLP、L29 MLP」。

☐ 我看到這一欄的差別。

### 2-3 文章有沒有照實寫那兩個但書

打開 `docs/technical-report.md`,找這兩句:

- 摘要的表格下方:"**one episode run without interruption and two that were cut by a cost cap and then resumed**"
- 同一段:"We set its response cap to 32,000 tokens ourselves, eight times the benchmark's 4,096." 接著說明原因(這個模型的推理關不掉、而且算在回應長度裡),以及 "not a row that could be added to the published leaderboard"

以及 "Deviations from our plan" 第 1 點最後一句:"Strictly by the plan, stage 2 is one episode in the "internal edit scores" category and two censored."

☐ 這三句都在,沒有被寫成「三題都成立」。

### 2-4 計畫真的寫在花錢之前

執行這兩行,比較時間:

```
git log -1 --format='%ad' --date=iso -- docs/editor-conditions-plan.md      (計畫最後一次修改的時間)
head -1 data/api-runs/ledger.jsonl                                            (第一筆花費;time 是秒數)
```

我替你換算好了:計畫第 2 版的 commit 是 2026-11-15 15:07:44;第一筆花費(參數測試,0.004 美元)是 15:16:12,第一個正式回合是 15:17:20。
(改寫歷史換信箱時,commit 的時間沒有變。)

☐ 計畫的時間早於第一筆花費。

### 2-5 我們的提示和原作者的一模一樣

這一點你無法用眼睛比對(18 萬個字元),只能看結論:`docs/editor-conditions-results.md`「驅動程式與基準的一致性」第一點,
「逐位元相同(182,277 字元)」。找漏洞的複核者另外對三題都查過(同一份文件「獨立複核」一節)。

☐ 我知道這一點是靠程式比對與複核者,不是我自己看的。

---

## 主張三:兩種改法打平,評分分不出來

### 3-1 「只回答姓氏」那一題,沒有一個修改通過

題目是 "Give only the surname of the … as of 2026-03-01."。六個修改後的回答:

```
               正確答案      gpt-6.1-sol 的修改後      我們的模型的修改後
加拿大          Carney       Martin                    Markus
Intel           Tan          Lip--stein                Brown
德國            Merz         Friedrich                 Scholz
```

**怎麼讀:** 沒有一格等於正確答案。兩種改法都只是讓模型在「問全名」時說出新名字,換個問法就不行。

☐ 六格都不是正確答案。

### 3-2 文章沒有把這一點說過頭

`docs/technical-report.md` 的 Finding 3 第一點應該是:"None of the nine patches we graded locally passes the surname-only target. No published `factual_update` rollout exceeds 2/3 targets; which target they missed is not in the public data."

☐ 文章寫的是「我們評過的沒有一個通過」,不是「所有編輯者都沒通過」。

---

## 過程:複核真的做了嗎

`docs/editor-conditions-results.md` 最後的「獨立複核」一節,以及 `docs/patch-ablation-results.md` 的「獨立複核」與「複核」小節。
你要看的不是細節,而是:

- 有沒有寫複核者**不准看什麼**(本專案的程式、文件、結果)。
- 有沒有寫複核**抓到了什麼錯**(例如:照規則有兩題不算數、有幾句話說過頭)。
- 有沒有寫**沒查到的**是什麼。

☐ 三樣都有寫。

---

## 不在這次發表範圍內的東西

專案裡最早的那批實驗(`wse/` 資料夾、`docs/design.md` 前半)是很小的玩具網路(約 205 個權重),用來把流程跑通。
它們**不是**這次發表的主張的依據,文章與 issue 都沒有引用它們。公開專案時,英文說明頁會寫明這一點。
