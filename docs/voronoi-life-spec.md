# voronoi-life 仕様書（初期実験版）

## 目的

ボロノイ分割上でセルオートマトンを実行し、正方格子上のライフゲームとは異なる挙動を観察する。

主な関心は、ボロノイ空間の不均質性が以下にどう影響するかを見ることである。

- 生存領域の偏り
- 活動しやすい場所・死滅しやすい場所の発生
- 密度正規化と絶対数ルールの違い
- 周期境界条件の有無による端効果の違い

この段階では、完成度の高いアプリではなく、実験と観察を優先する。

## 実装方針

実装環境は固定しない。

候補:

- Python + SciPy + matplotlib
- Python + SciPy + NetworkX + matplotlib
- ブラウザ + D3.js / canvas / SVG
- p5.js

初期実験では Python 実装を優先する。
理由は、Voronoi / Delaunay の生成、数値実験、GIF出力が容易なため。

## 空間モデル

### 基本構造

- 2次元平面上に点群を生成する。
- 点群から Voronoi 分割を作る。
- 各 Voronoi cell を1つのセルとみなす。
- セルの状態は `alive` / `dead` の二値とする。

### 隣接関係

計算上は、Voronoi cell の共有辺に基づく隣接関係を使う。
実装では Delaunay triangulation を使って隣接グラフを得てもよい。

つまり、以下を対応させる。

- Voronoi cell: CAのセル
- Delaunay edge: セル間の隣接関係

## 点群生成

最低限、以下の3種類を切り替えられるようにする。

### 1. 完全ランダム

単位正方形 `[0, 1] x [0, 1]` に一様ランダムに点を置く。

目的:

- 不均質性の強い空間での挙動を見る。

### 2. 揺らした格子

正方格子または六角格子を作り、各点を小さくランダムにずらす。

目的:

- 格子的秩序を少し残した状態で、Life的な挙動が残るかを見る。

### 3. 密度勾配あり

領域の一部に点が多く集まるような点群を作る。

目的:

- 「豊かな土地」「疎な土地」のような空間差がCA挙動に現れるかを見る。

## 周期境界条件

可能なら初期段階から実装する。

### 理由

通常のVoronoi分割では端のセルが大きく歪み、端効果が強くなる。
周期境界条件を入れることで、端による死滅・固定化を減らせる。

### 実装案

点群を 3 x 3 にタイル複製する。
その上で Voronoi / Delaunay を生成する。
中央領域のセルだけを観察対象とする。

最低限、以下を切り替えられるようにする。

- 周期境界あり
- 周期境界なし

## 更新ルール

同期更新とする。
全セルの次状態を計算してから、一斉に状態を置き換える。

### 1. 絶対数版 Life-like rule

Conway Life に近いルール。

- dead cell は alive neighbor 数が `birth_min_count <= n <= birth_max_count` のとき alive になる。
- alive cell は alive neighbor 数が `survive_min_count <= n <= survive_max_count` のとき alive のまま。
- それ以外は dead になる。

初期値:

- `birth_min_count = 3`
- `birth_max_count = 3`
- `survive_min_count = 2`
- `survive_max_count = 3`

観察したい点:

- 近傍数が多い場所ほど活動しやすくなるか。
- 近傍数が少ない場所が死滅しやすくなるか。
- 空間構造がそのまま「地形」として現れるか。

### 2. 密度版 rule

alive neighbor の割合で判定する。

定義:

```text
rho = alive_neighbors / total_neighbors
```

初期案:

- dead cell: `rho` が `birth_min <= rho <= birth_max` のとき alive
- alive cell: `rho` が `survive_min <= rho <= survive_max` のとき alive
- それ以外は dead

初期値例:

```text
birth_min = 0.30
birth_max = 0.45
survive_min = 0.20
survive_max = 0.45
```

観察したい点:

- 少ないセル数でも滑らかな場として振る舞うか。
- 空間の不均質性が薄まるか。
- 波・境界・斑状パターンが出るか。

### 3. 確率版 rule

密度または絶対数から出生・死亡確率を決める。

初期案:

- dead cell は `rho` が適正範囲に近いほど出生確率が上がる。
- alive cell は過疎または過密で死亡確率が上がる。

単純実装例:

```text
birth_probability = clamp(a * (rho - birth_threshold), 0, 1)
death_probability = clamp(b * abs(rho - optimal_density), 0, 1)
```

最初は厳密なモデル化より、固定化・全滅を避けるための実験ルールとして扱う。

## 表示

最低限の表示:

- dead cell: 白または薄色
- alive cell: 黒または濃色
- Voronoi polygon を描画する

追加表示:

- Delaunay edge の表示切り替え
- cell degree（近傍数）のヒートマップ
- alive neighbor 数のヒートマップ
- alive neighbor 密度のヒートマップ

初期段階では、美観より観察しやすさを優先する。

## 操作・パラメータ

最低限あるとよい操作:

- 再生 / 停止
- 1ステップ進める
- リセット
- ランダム初期状態生成
- 点群再生成
- ルール切り替え
- 周期境界条件の切り替え

最低限あるとよいパラメータ:

- セル数
- 初期 alive 比率
- 点群生成方式
- 更新ルール
- birth / survive 閾値
- 確率版の強度

## 出力

初期実験では以下を保存できるとよい。

- 静止画 PNG
- アニメーション GIF または MP4
- 実験条件 JSON

実験条件には以下を含める。

- cell count
- seed
- point generation method
- boundary condition
- rule type
- rule parameters
- initial alive ratio
- number of steps

## 最初のマイルストーン

### Milestone 1: 最小実装

- ランダム点群を生成する。
- Voronoi cell を描画する。
- Delaunay graph から隣接関係を作る。
- alive / dead を描画する。
- 絶対数版 B3/S23 を同期更新する。

### Milestone 2: 比較実験

- 密度版 rule を追加する。
- 絶対数版と密度版を切り替える。
- 同じ点群・同じ初期状態で比較できるようにする。

### Milestone 3: 空間生成の比較

- 完全ランダム点群
- 揺らした格子
- 密度勾配あり

を切り替える。

### Milestone 4: 周期境界条件

- 3 x 3 タイル複製による周期境界を実装する。
- 周期境界あり / なしで挙動を比較する。

### Milestone 5: 記録出力

- PNG / GIF / MP4 のいずれかを出力する。
- 実験条件を JSON で保存する。

## 観察メモ

このプロジェクトで見たい差は、単に「面白い模様が出るか」ではない。

主な観察対象は以下である。

- 密度正規化すると、空間差がどれくらい消えるか。
- 絶対数ルールでは、高次数領域が活動帯になるか。
- 疎な領域は死滅領域になるか。
- 境界や密度勾配に沿ってパターンが流れるか。
- 完全ランダム空間ではノイズ化し、揺らした格子ではLife的構造が残るか。
- 周期境界条件を入れると、移動パターンや波が維持されやすくなるか。

## 非目標

初期段階では以下を目標にしない。

- 完成度の高いUI
- Web公開
- GNNによる学習
- 論文再現
- Leniaのような連続値モデル
- 厳密な生物モデル化

まずは、ボロノイセル空間上の二値CAを動かし、見た目と挙動を観察する。
