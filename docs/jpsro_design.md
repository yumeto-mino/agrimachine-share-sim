# JPSRO(CCE) 設計方針

任意の POSG に対して、JPSRO(CCE)（Marris et al., *Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium Meta-Solvers*, ICML 2021; `references/jpsro.pdf`）で NFCCE を求めるプログラムの設計方針をまとめる。

---

## 1. 目的と全体像

### 用語

| 用語 | 意味 |
|---|---|
| POSG | 部分観測確率ゲーム。n 人が同時に行動し、各自は自分の観測だけを見る |
| Π_p | プレイヤー p がこれまでに得た方策の集合（方策集合） |
| G | 利得表（メタゲーム）。G_p(π) は方策の組 π = (π_0, …, π_{n-1}) で遊んだときのプレイヤー p の期待利得 |
| σ | 方策の組の上の同時分布。CCE の相関装置にあたる |
| ER | Expected Return。方策の組で実際に遊ばせて G を埋める処理（`ERs/simulate.py`） |
| MS | Meta-Solver。G から方策の組の上の分布 σ を決める処理（`MSs/`） |
| BR | Best Response。σ から引いた相手に対して最もよい方策を学習する処理（`BRs/ppo.py`） |
| CCE gap Δ_p | σ に従う代わりに BR に乗り換えたとき、p が得をする量 |

### JPSRO(CCE) の 1 反復

1. **ER**：これまでの方策の全組み合わせで遊ばせ、G を埋める
2. **MS**：G から σ^t を求める
3. **BR**：各プレイヤー p について、σ^t から p 自身を周辺化した相手の分布

   σ^t(π_{-p}) = Σ_{π_p ∈ Π_p} σ^t(π_p, π_{-p})

   に対する最適応答を PPO で学習する

   BR_p^{t+1} ∈ argmax_{π_p} Σ_{π_{-p}} σ^t(π_{-p}) G_p(π_p, π_{-p})

4. 全員の BR を方策集合に加える

### 本体ループ（`algorithms/jpsro.py`）

```python
Π = [game.initial_policies(p) for p in players]   # 農機シェアでは「正直申告」
G = ER(Π)
σ = MS(G)
while True:
    encoders.fit(simulate_under(σ))               # (ii) の Encoder のみ実際に学習する
    new = [BR(p, σ) for p in players]             # 全員が同じ σ に対して学習する
    for p in players:
        Π[p].append(new[p])
    G = ER(Π)                                     # 新しい組み合わせだけ評価する
    Δ = cce_gap(G, σ)                             # 新しい BR の行から計算する
    save(t, Π, G, σ, Δ)
    if should_stop(Δ):
        break
    σ = MS(G)
```

---

## 2. 全体原則

- **ppo.py は CleanRL との差分を極力少なくする。** Encoder や相手の方策などは ppo.py の外（ラッパ）に置き、ppo.py は普通の 1 人用 gym を学習するだけにする。
- **ゲームに依らない部品とゲーム固有のものを分ける。** `algorithms/` はどの POSG でも使える。`envs/` と `examples/` はゲームごとに書く。
- **実験内容はコードの大枠が完成してから決める。** K、反復数、閾値、PPO のハイパーパラメータなどは、設定項目として持たせるだけにする（問題の切り分けと頭の整理のため）。

---

## 3. ファイル構成

`src/` を廃止し、パッケージをリポジトリ直下に置く。

```
agrimachine_share_sim/
  examples/
    agrimachine_main.py     設定を読み、env と jpsro を組み立てて実行する（ゲーム・実験ごと）
  envs/
    base.py                 POSG の共通の型（PettingZoo ParallelEnv + state() + action_mask + 初期方策）
    agrimachine.py          農機シェアのゲーム、正直申告の方策、(i)(ii) の Encoder
  algorithms/
    jpsro.py                JPSRO の本体ループ
    BRs/
      ppo.py                CleanRL PPO + 最小限の変更
    ERs/
      simulate.py           ゲームに依らない rollout
    MSs/
      mgcce.py              共通 IF: solve(G) -> σ
      mwcce.py
      uniform.py
    tools/
      wrappers.py           BR 用の 1 人用ラッパ
      policies.py           方策・Encoder の共通の型、NN 方策、一様ランダム方策、GRU 用の素通し Encoder
configs/
  envs/default.yaml         既存
  jpsro/default.yaml        新規（MS、Encoder の種類、K、乱数の種、PPO の設定など）
docs/
results/<run>/              出力（.gitignore 済み）
```

### 依存の向き

- 依存は **`examples` → `algorithms` → `envs`** の一方向にする。
- 方策と Encoder の型は `typing.Protocol` で書く（「`reset()` と `act()` を持っていれば方策」というダックタイピング）。こうすると `envs/agrimachine.py` の正直申告方策や Encoder は、`algorithms` を import せずに書ける。
- `examples/*_main.py` だけが、ゲームの部品とアルゴリズムの部品をつなぐ。

### pyproject.toml の変更

- `[tool.uv.build-backend]` に `module-root = ""` を追加する（`uv_build` は標準では `src/` を前提にしているため）。
- `[project.scripts]` の `agrimachine-share-sim = "agrimachine_share_sim:main"` は削除する。起動は `uv run python -m agrimachine_share_sim.examples.agrimachine_main` のように行う。
- 依存に `pettingzoo`、`gymnasium`、`tyro`、`tensorboard` を追加する（後の 3 つは現在の ppo.py が import しているのに未登録）。Python 3.13 に対応するバージョンを確認する。
- 現在の ppo.py は gymnasium 1.0 より前の `infos["final_info"]` の書き方を前提にしている。gymnasium 1.x では自動リセットの挙動が違うため、バージョンを固定するか、ppo.py 側を合わせる。

---

## 4. env（`envs/`）

### 形

- ゲーム本体は **n 人同時手番の PettingZoo `ParallelEnv`** として 1 か所に書く。ER（simulate.py）も BR（ppo.py、ラッパ経由）も同じ本体を使う。
- 新しいゲームを書いたら、`pettingzoo.test.parallel_api_test` で API を検証する。

### env が返すもの

| もの | 取り出し方 | 使う人 |
|---|---|---|
| 生の観測 o_i^t | `reset()` / `step()` の obs[i] | 各プレイヤーの方策（Encoder を通す） |
| 真の状態 | `state()` | Critic のみ |
| 合法な行動のマスク | `info[i]["action_mask"]` | 方策（行動の分布を合法手に限る） |
| 初期方策 | `initial_policies(p)` | JPSRO の Π^0 |

- **マスクは、そのプレイヤー自身の情報だけで決まること。** 他人の私的情報でマスクが変わると、マスクから情報が漏れる（例：農機シェアの「θ_{e,f} = 0 の畑には 0 しか報告できない」は自分の θ_i で決まるので問題ない）。
- 初期方策をゲームが与えなければ、合法手の一様ランダム方策を使う。
- 農機シェアの観測規約：各メンバーは自分の履歴 h_i^t = (φ^0, θ_i^0, o_i^1, …, o_i^t) だけを見る。t = 0 でも見えるのは自分の θ_i^0 だけで、θ^0 全体は見えない。

### 農機シェアのゲーム（`envs/agrimachine.py`）

- 現在の `simulate.py` の骨組みにあるゲームのルール（`Member`、申告の集約、割当、`_transit_feasibility`、`_transit_grade`、`_generate_rewards` など）はここへ移す。
- パラメータは既存の `configs/envs/default.yaml` を使う。
- 正直申告の方策と、(i)(ii) の Encoder もここに置く。

---

## 5. 方策と Encoder（`algorithms/tools/policies.py`）

### 方策の共通の型

```python
policy.reset()                          # エピソードの最初（GRU の記憶や Encoder の推定を初期化）
action = policy.act(obs, action_mask)   # 毎期：自分の生の観測 o_i^t とマスク → 行動
```

方策集合には次の種類が混ざるが、simulate.py とラッパはこの 2 つの操作だけで、中身を区別せずに動かす。

| 種類 | 置き場所 | 中身 |
|---|---|---|
| ルール方策（正直申告） | `envs/agrimachine.py` | 自分の θ_i をそのまま報告する。重みなし |
| 一様ランダム方策 | `tools/policies.py` | マスクで許された行動から一様に選ぶ |
| NN 方策 | `tools/policies.py` | Encoder のスナップショット + 学習済みネット |

### Encoder の共通の型

```python
encoder.fit(rollouts)       # σ の下でのシミュレーション結果から内部の表などを作る（不要なら何もしない）
encoder.reset()             # エピソードの最初
x = encoder.step(obs)       # 毎期：生の観測 o_i^t → Actor への入力ベクトル
```

### Actor への入力（3 種類を実験で比較）

1 回の実験の中では、全プレイヤーの全方策が同じ種類を使う（設定 1 項目）。

| 種類 | 中身 | fit | 置き場所 |
|---|---|---|---|
| (i) 素朴な推定状態 | 他者の 1 期前の申告を真とみなし、遷移モデルで今期の状態を予測する | 何もしない | `envs/agrimachine.py` |
| (ii) 洗練された推定状態 | 申告 θ̃_j から申告者の私的情報 θ_j を対応表 Pr(θ_j \| θ̃_j) で推定する | 各反復で σ^t の下で simulate.py を回し、集めた（申告, 真の私的情報）から表を作る | `envs/agrimachine.py` |
| (iii) 観測履歴 + GRU | o_i^t をそのまま渡し、記憶はネット側の GRU が持つ | 何もしない | `tools/policies.py`（素通し） |

- (ii) の対応表は、その反復で学習する方策の Encoder に**固定して**持たせる。後の反復で表が作り直されても、過去の方策は自分が学習したときの表を使い続ける。
- NN 方策は「Encoder + ネット」をひと組で保存する。ER で過去の方策を動かすときも、学習時と同じ Encoder を使うため。
- (ii) の表が変わっても Actor の入力の長さは変わらないので、`previous` 初期化（§6）で重みを引き継げる。

---

## 6. BR（`algorithms/BRs/ppo.py` + `algorithms/tools/wrappers.py`）

### 1 人用ラッパ（`tools/wrappers.py`）

n 人用の env を包み、ppo.py からは普通の 1 人用 `gymnasium.Env` に見せる。

- `reset()`：σ^t(π_{-p}) から相手の方策の組を 1 つ引き、全員の方策と Encoder を `reset()` する
- `step(a_p)`：
  1. 相手の方策にそれぞれの観測とマスクを渡して行動を決めさせる
  2. 学習者の行動と合わせて本体の `step()` に渡す
  3. 学習者の観測を Encoder に通して返す。`info` に `state`（真の状態）と `action_mask` を入れる
- 学習者 p は、相手がどの方策を引いたかを知らない（CCE の前提）。

### Actor と Critic

Critic には **真の状態 + Actor の入力** を見せる。真の状態だけを見る Critic は部分観測下で学習を偏らせることがあり、Actor が持つ情報（履歴の要約）も合わせて見せると偏りが消える（Baisero & Amato, *Unbiased Asymmetric Reinforcement Learning under Partial Observability*, AAMAS 2022）。

| 種類 | Actor | Critic | 手本 |
|---|---|---|---|
| (i) (ii) | MLP(x) | MLP(cat(x, state))。Actor とは別のネット | 現在の CleanRL `ppo.py` |
| (iii) | GRU の出力 → Actor ヘッド | cat(GRU の出力, state) → Critic ヘッド。GRU は共有 | CleanRL `ppo_atari_lstm.py`（LSTM → GRU） |

### 合法手

- 方策は合法な行動の上の分布にする。logits の非合法手を -∞ 相当にしてから `Categorical` を作る。
- ラッパ内の相手の方策と ER（simulate.py）でも同じマスクを使う。

### ppo.py への変更（これ以外は CleanRL のまま）

1. `if __name__ == "__main__":` の本体を `train(args, make_env, init_state_dict=None)` という関数にまとめる。学習済みの Agent を返す。tyro による単体実行も残す。
2. `make_env` が CartPole ではなく 1 人用ラッパを作るようにする。
3. `info["state"]` と `info["action_mask"]` を保持するバッファを追加し、Critic の入力に state を足し、logits にマスクをかける。
4. (iii) のみ GRU 化する。CleanRL `ppo_atari_lstm.py` の変更パターン（初期隠れ状態、done での記憶のリセット、環境単位のミニバッチ）に倣い、変更箇所を局所化する。

### 学習の進め方

- **初期化**：設定で切り替える。
  - `scratch`：毎回ネットを初期化してゼロから学習する（CleanRL のまま）
  - `previous`：そのプレイヤーの直前の BR の重み（Actor と Critic の両方）から始める。直前がルール方策で重みがない場合は `scratch`
- **1 反復の中**：全プレイヤーが同じ σ^t に対して BR を学習し、反復の最後にまとめて方策集合に加える。各プレイヤーの学習は独立だが、まずは順番に実行する。

---

## 7. ER（`algorithms/ERs/simulate.py`）

simulate.py はゲームに依らない rollout 処理だけを持つ。

```python
for i in players:
    policies[i].reset()
obs, info = env.reset(seed=seed)
while env.agents:
    actions = {i: policies[i].act(obs[i], info[i]["action_mask"]) for i in env.agents}
    obs, rewards, terminations, truncations, info = env.step(actions)
```

### 役割

1. **利得表 G を埋める**：各方策の組み合わせを K エピソード回し、各自の利得の平均と標準誤差を記録する。前の反復で評価済みの組み合わせは使い回し、新しい組み合わせだけ評価する。G の形は (n, |Π_0|, …, |Π_{n-1}|)。
2. **CCE gap を出す**：

   Δ_p^t = Σ_{π_{-p}} σ^t(π_{-p}) G_p(BR_p^{t+1}, π_{-p}) − Σ_π σ^t(π) G_p(π)

   第 1 項は、新しい BR の行（次の 1. で必ず評価する）から計算できるので、追加のシミュレーションは不要。
3. **σ の下でのデータ集め**：毎エピソード σ から方策の組を引いて遊ばせ、（申告, 真の私的情報）などを記録する。(ii) の Encoder の `fit` に使う。

### 設定項目（値は後で決める）

- K（1 つの組み合わせあたりのエピソード数）
- 乱数の種。共通乱数法を使う：k 番目のエピソードは、どの組み合わせでも同じ種（同じ初期状態・同じ自然の遷移）にし、方策の違いによる差が乱数のぶれに埋もれにくくする
- 並列化（multiprocessing）の有無

### 注意：利得表の大きさ

組み合わせの数は ∏_p |Π_p| で増える。3 人ゲームで 20 反復後には 21³ = 9,261 通り、1 反復あたりの新しい組み合わせは 21³ − 20³ = 1,261 通り。CCE を解くには表全体が必要なので、この増え方は避けられない。

---

## 8. MS（`algorithms/MSs/`）

どの MS も同じ形 `solve(G) -> σ` で呼べるようにし、設定で切り替える。

CCE 制約（全プレイヤー p、全 π_p' ∈ Π_p について）：

Σ_π σ(π) [G_p(π_p', π_{-p}) − G_p(π)] ≤ ε,  σ ≥ 0,  Σ_π σ(π) = 1

| MS | 目的 | 解き方 | 備考 |
|---|---|---|---|
| MGCCE（学習用の既定） | Gini 不純度 1 − σᵀσ を最大化 | cvxpy の二次計画 | 論文の推奨。解が一意に決まる |
| MWCCE | 社会厚生 Σ_π σ(π) Σ_p G_p(π) を最大化 | cvxpy の線形計画 | 解が一意に決まらないことがある |
| uniform | 全組み合わせを等確率に | 計算不要 | CCE にはならず、収束保証なし |

- 学習に使う σ は MGCCE を既定にする。
- 評価（CCE gap、各自の利得）は、学習に使った σ とは別の分布（MWCCE など）でも計算できるようにする。MS の解が均衡とは限らない場合や、利得が最大の均衡を見たい場合に使う（論文 §5.3）。

---

## 9. 終了条件

- まず論文どおり厳密にする：`while True` で回し、**全プレイヤーの CCE gap Δ_p ≤ 0** になったら止める。
- 止めるかどうかの判定は 1 つの関数 `should_stop(Δ, history)` に分け、後から差し替えられるようにする（最大反復数、閾値、連続回数など）。
- PPO による BR は近似なので gap はぶれ、この条件では実際には止まらない可能性が高い。そのため毎反復の結果を保存し、中断と再開ができるようにする。

---

## 10. 保存と設定

### 方策の保存

方策 1 つにつき 1 ファイルを `torch.save` で保存する：`results/<run>/policies/p{p}/{index:03d}.pt`

| キー | NN 方策 | ルール方策 |
|---|---|---|
| `kind` | `"nn"` | `"rule"` |
| `name` | — | `"truthful"` など |
| `player`, `iteration` | ○ | ○ |
| `agent_config` | 入力の長さ、state の長さ、行動の数、MLP か GRU か | — |
| `state_dict` | Actor と Critic の重み（`previous` 初期化に Critic も使う） | — |
| `encoder` | `{type, state}`（(ii) なら対応表） | — |

読み込むときは、`agent_config` から Agent を作って `load_state_dict` し、Encoder と組んで NN 方策にする。ルール方策は `name` からゲーム側の方策を作り直す。

### 毎反復の保存

`results/<run>/` に次のものを書き出す。

- 利得表 G とその標準誤差
- σ（学習用、評価用）
- 各プレイヤーの CCE gap と利得の推移
- PPO の学習曲線（TensorBoard。現在の ppo.py と同じ）

再開するときは、最後に保存した反復から続ける。

### 設定

- env と JPSRO の設定は yaml（`configs/envs/*.yaml`、`configs/jpsro/*.yaml`）。
- PPO の設定は ppo.py の `Args`（dataclass）をそのまま使い、`configs/jpsro/*.yaml` の `ppo:` の項目で上書きする。ppo.py 単体で動かすときのコマンドライン（tyro）も残す。
- 主な JPSRO の設定項目：MS（学習用・評価用）、Encoder の種類、K、乱数の種、BR の初期化（`scratch` / `previous`）。

---

## 11. 後で決めること

コードの大枠が完成してから検討する。

- K、反復数、gap の閾値など終了条件の具体形
- PPO のハイパーパラメータ（total_timesteps など）
- ER と BR の並列化
- 実験の組み方（Encoder 3 種類の比較の設計、ゲームのパラメータ）
- `examples/` 以外の起動方法が必要か
