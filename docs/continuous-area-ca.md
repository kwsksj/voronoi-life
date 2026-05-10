# ボロノイセルの面積・境界長・セル間距離を使う連続量CA 追加仕様

## 目的

現在の `density` / `absolute` 系ルールは、セル状態を二値として扱う。
この追加仕様では、各 Voronoi cell に連続量の `life amount` を持たせ、セル面積・共有辺長・中心間距離を使って隣接セル間の移動・拡散を計算する。

目的は、Conway Life の移植ではなく、ボロノイメッシュ上の反応拡散的な連続量CAを観察することである。

初期実験では、見た目の面白さと挙動の比較を優先する。
物理的厳密性は二次的に扱う。

## 現在の実験条件との関係

添付された初期実験では、以下の条件で `density` ルールを実行している。

- boundary condition: `periodic`
- cell count: `50000`
- point generation method: `random`
- initial alive ratio: `0.28`
- rule type: `density`
- seed: `3`
- steps: `100`

連続量CAは、この既存実装に追加する別ルールとして実装する。
既存の二値ルールを壊さず、`rule_type = "continuous"` などで切り替えられるようにする。

## 基本変数

各セル `i` について以下を保持する。

```text
A_i      : セル面積
L_i      : セル内の life amount
rho_i    : セル密度。rho_i = L_i / A_i
N(i)     : 隣接セル集合
ell_ij   : セル i と j の共有辺長
d_ij     : セル中心 i と j の距離
```

`L_i` は保存量として扱う。
表示や反応判定には `rho_i` を使う。

## 幾何量の前計算

実装時には、シミュレーション開始前に以下を前計算する。

- 各セル面積 `A_i`
- 各隣接ペア `(i, j)` の共有辺長 `ell_ij`
- 各隣接ペア `(i, j)` の中心間距離 `d_ij`
- 各隣接ペアの coupling weight `w_ij`

周期境界条件ありの場合、距離 `d_ij` は周期境界を考慮した最短距離にする。

## 拡散項

基本形は以下とする。

```text
rho_i = L_i / A_i
flux_i = sum_j w_ij * (rho_j - rho_i)
L_i_next = L_i + diffusion_rate * flux_i
```

ここで `j` は `N(i)` の全隣接セル。

`w_ij` は以下から切り替え可能にする。

### 1. graph

```text
w_ij = 1
```

隣接グラフだけを見る。
面積・辺長・距離を使わない比較用。

### 2. edge

```text
w_ij = ell_ij
```

共有辺が長いほど強く混ざる。
中心間距離では割らない。
見た目の実験として最初に試す候補。

### 3. edge_distance

```text
w_ij = ell_ij / d_ij
```

共有辺長を通路の広さ、中心間距離を粗視化された勾配距離として扱う。
有限体積法の拡散に近い比較用。

## 保存性

上の拡散式は、無向エッジで `w_ij = w_ji` として扱えば、全体の `sum(L_i)` が保存される。

実装では、可能ならセルごとに独立に `flux_i` を計算するだけでなく、エッジ単位で流量を計算する方法も検討する。

エッジ単位の実装例:

```text
for each undirected edge (i, j):
    flow = diffusion_rate * w_ij * (rho_j - rho_i)
    delta_L[i] += flow
    delta_L[j] -= flow
L_next = L + delta_L
```

この方式のほうが保存性を確認しやすい。

## 反応項

拡散だけだと最終的に均質化しやすい。
観察対象としては、拡散に加えて局所反応を入れる。

初期実装では、以下のいずれかを選べるようにする。

### A. 拡散のみ

```text
reaction = 0
```

幾何量による混ざり方を見るための基準。

### B. logistic growth

```text
reaction_i = growth_rate * rho_i * (1 - rho_i / carrying_capacity) * A_i
L_i_next = L_i_after_diffusion + reaction_i
```

密度が低いと増え、高すぎると増えにくい。
単純な生態系モデルとして扱う。

### C. bell-shaped growth

適正密度 `optimal_density` の近くで増え、離れると減る。

```text
growth = growth_rate * exp(-((rho_i - optimal_density)^2) / (2 * sigma^2))
death = death_rate * rho_i
reaction_i = (growth - death) * A_i
```

視覚的には、斑点・膜・波のようなパターンが出る可能性がある。

## 値の制約

各ステップ後に以下を行う。

```text
L_i = max(L_i, 0)
rho_i = L_i / A_i
```

必要に応じて上限も入れる。

```text
rho_i = min(rho_i, rho_max)
L_i = rho_i * A_i
```

ただし、上限を入れると保存性は崩れる。
保存性を観察したい実験では上限なし、見た目を安定させたい実験では上限ありとする。

## 時間刻みと安定性

明示オイラー更新なので、`diffusion_rate` が大きいと発散・振動・負値化しやすい。
初期値は小さくする。

推奨初期値:

```text
diffusion_rate = 0.01
```

`edge_distance` を使う場合は `w_ij` が大きくなる可能性があるため、さらに小さくする。

```text
diffusion_rate = 0.001
```

必要なら、1描画ステップの中で複数の小さな内部ステップを回す。

## 初期状態

二値ではなく連続量として初期化する。

候補:

### random_density

```text
rho_i ~ Uniform(0, initial_density_max)
L_i = rho_i * A_i
```

### binary_density

既存の alive / dead に近い初期化。

```text
rho_i = alive_density if random() < initial_alive_ratio else 0
L_i = rho_i * A_i
```

### gaussian_blob

領域内に1つまたは複数の密度塊を置く。

```text
rho_i = sum_k amplitude_k * exp(-dist(center_i, blob_center_k)^2 / sigma_k^2)
L_i = rho_i * A_i
```

周期境界ありの場合、距離は周期距離にする。

## 表示

連続量CAでは、alive/dead の黒白表示ではなく、密度 `rho_i` を濃淡で表示する。

最低限:

- `rho_i` を grayscale で描画
- `rho_min`, `rho_max` を指定可能
- 自動スケーリングと固定スケーリングを切り替え可能

追加表示:

- `A_i` 面積ヒートマップ
- `degree` ヒートマップ
- `sum(edge length)` ヒートマップ
- `flux_i` ヒートマップ

## 追加するCLI引数案

既存CLIの構造に合わせて調整する。

```text
--rule continuous
--continuous-init random_density | binary_density | gaussian_blob
--coupling graph | edge | edge_distance
--diffusion-rate 0.01
--reaction none | logistic | bell
--growth-rate 0.02
--death-rate 0.01
--carrying-capacity 1.0
--optimal-density 0.35
--sigma 0.08
--rho-max 1.0
--density-scale auto | fixed
```

## 実験メニュー

最初に以下を比較する。

### Experiment 1: 拡散のみ

```text
rule=continuous
reaction=none
coupling=graph / edge / edge_distance
```

目的:

- グラフだけ、共有辺長、共有辺長/距離で混ざり方がどう違うかを見る。
- 最終的に均質化するまでの過程を見る。

### Experiment 2: logistic reaction

```text
reaction=logistic
coupling=edge
```

目的:

- 拡散と局所増殖だけで、安定した濃淡分布が出るかを見る。

### Experiment 3: bell-shaped reaction

```text
reaction=bell
coupling=edge
```

目的:

- 適正密度付近で増え、離れると減るルールで、波・斑点・境界が出るかを見る。

### Experiment 4: coupling comparison

同じ seed、同じ点群、同じ初期状態で以下を比較する。

```text
coupling=graph
coupling=edge
coupling=edge_distance
```

目的:

- 「隣接グラフだけ」から「ボロノイ幾何を使う」方向へ変えたとき、視覚的・力学的差が出るかを見る。

## 実験条件JSONへの追加項目

連続量CAでは、実験条件に以下を保存する。

```json
{
  "rule_type": "continuous",
  "continuous_parameters": {
    "continuous_init": "random_density",
    "coupling": "edge",
    "diffusion_rate": 0.01,
    "reaction": "bell",
    "growth_rate": 0.02,
    "death_rate": 0.01,
    "carrying_capacity": 1.0,
    "optimal_density": 0.35,
    "sigma": 0.08,
    "rho_max": 1.0,
    "density_scale": "fixed"
  }
}
```

## 実装上の注意

- 既存の二値 `alive` 配列とは別に、連続量 `life_amount` または `density` 配列を持つ。
- 内部保存量は `life_amount`、表示量は `density` とする。
- 面積が極端に小さいセルは密度が大きく振れやすい。必要なら最小面積しきい値や Lloyd relaxation を検討する。
- 周期境界条件で Voronoi polygon を描画する場合、中央タイルのセル面積・共有辺長を正しく使う。
- 最初の実装では、完全な物理的厳密性より、同じ seed で比較できることを優先する。

## 非目標

この追加実装では、以下は目標にしない。

- Lenia の完全再現
- GNN による更新則学習
- 厳密な有限体積法ソルバ
- 連続空間の高精度PDE解法

ここでの目的は、ボロノイ幾何を使った連続量CAを、既存の二値CAと比較可能な形で追加することである。
