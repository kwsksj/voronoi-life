# voronoi-life

ボロノイ分割の上でセルオートマトンを動かすための初期実験ツールです。

正方形のマス目ではなく、不均一なボロノイセルを「1つのセル」として扱います。これにより、場所ごとの隣接数やセルの大きさの違いが、生存・死滅・活動しやすさにどう影響するかを観察できます。

## セットアップ

初回だけ、以下を実行して必要なライブラリを入れます。

```bash
uv sync
```

`uv` は Python の実行環境とライブラリをそろえるための道具です。手元の環境差で動いたり動かなかったりする問題を減らすために使っています。

## 実行方法

基本の実行コマンドです。

```bash
uv run voronoi-life run --cells 300 --rule absolute --steps 200 --seed 1
```

実行結果は、標準では `runs/` の下に保存されます。

- `final.png`: 最終状態の画像
- `animation.gif`: 変化のアニメーション
- `experiment.json`: セル数、乱数 seed、ルール、ステップ数などの実験条件

## 実行例

### 絶対数ルールと密度ルールを比べる

同じ `seed` を使うと、同じ点群・同じ初期状態から実験できます。`seed` は乱数の出発点を固定する番号です。

```bash
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule absolute
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule density
```

### 点群の作り方を変える

点群は、ボロノイセルの元になる点の並び方です。点群を変えると、空間の不均一さが変わります。

```bash
uv run voronoi-life run --points random
uv run voronoi-life run --points jittered-grid
uv run voronoi-life run --points density-gradient
```

指定できる点群は以下です。

- `random`: 完全ランダム
- `jittered-grid`: 格子を少し崩した配置
- `density-gradient`: 点が多い場所と少ない場所を作る配置

### 周期境界を使う

周期境界は、右端と左端、上端と下端がつながっているように扱う設定です。端だけが特殊な場所になってしまう影響を減らせます。

```bash
uv run voronoi-life run --periodic
```

### 表示を切り替える

`--overlay` で、通常表示以外の観察用表示に切り替えられます。

```bash
uv run voronoi-life run --overlay degree
uv run voronoi-life run --overlay alive-count
uv run voronoi-life run --overlay alive-density
```

- `degree`: 各セルの近傍数
- `alive-count`: 生きている近傍セルの数
- `alive-density`: 近傍セルのうち、生きているセルの割合

## テスト

```bash
uv run pytest
```

テストは、点群生成、隣接関係、更新ルール、周期境界、100ステップ以上の実行が壊れていないかを確認します。
