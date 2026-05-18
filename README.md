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
- `experiment.json`: セル数、乱数 seed、ルール、実際に進んだステップ数、停止状態などの実験条件

同じ状態が再び出た場合は、指定ステップ数の途中でも計算を止めます。1つ前と同じ状態なら `steady`、2ステップ以上前の状態に戻った場合は `oscillating` として記録します。

## GUI で調整する

PC でパラメーターを変えながら確認したい場合は、ブラウザ GUI を使えます。

```bash
uv run voronoi-life gui
```

起動すると、ローカルブラウザで `http://127.0.0.1:8765/` が開きます。画面左で変化を確認し、画面右でセル数、seed、点群、ルール、表示方法、初期状態の作り方を調整できます。

- `設定を適用`: 入力した条件で最初から作り直します。
- `再生` / `停止`: 自動でステップを進めます。
- `1ステップ` / `10ステップ`: 手動で変化を確認します。
- `初期状態を作り直す`: 点群はそのまま、alive の初期配置や連続量の初期密度を作り直します。
- `点群を作り直す`: ボロノイセルの形も含めて作り直します。
- `PNG保存`: 現在の表示を `runs/` の下に保存します。
- `数式`: 選択中のルールで、各パラメーターがどう使われるかを表示します。
- `ヒントを表示`: 各パラメーターの意味を画面内に表示します。慣れたらオフにできます。

計算が定常状態に入った場合は、自動再生が止まり、画面下部の `state` とメッセージ欄に停止理由が表示されます。振動状態に入った場合は、新しい計算は進めず、検出済みの周期パターンを画面上で繰り返し表示します。

## 実行例

### 絶対数ルールと密度ルールを比べる

同じ `seed` を使うと、同じ点群・同じ初期状態から実験できます。`seed` は乱数の出発点を固定する番号です。

```bash
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule absolute
uv run voronoi-life run --cells 200 --steps 50 --seed 1 --rule density
```

`absolute` では、alive 隣接セル数の下限・上限を誕生と生存で別々に指定できます。

```bash
uv run voronoi-life run --rule absolute --birth-min-count 3 --birth-max-count 4 --survive-min-count 2 --survive-max-count 4
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

### 連続量CAを使う

`continuous` ルールでは、生死の二値ではなく、各セルに `life amount` という量を持たせます。表示では、セル面積で割った密度を白黒の濃淡として描きます。

```bash
uv run voronoi-life run --rule continuous --periodic --cells 500 --steps 100 --coupling edge --reaction none
uv run voronoi-life run --rule continuous --periodic --cells 500 --steps 100 --coupling edge_distance --diffusion-rate 0.001
uv run voronoi-life run --rule continuous --periodic --cells 500 --steps 100 --reaction bell --density-scale fixed --rho-max 1.0
```

主な指定項目です。

- `--continuous-init`: 初期密度の作り方。`random_density`、`binary_density`、`gaussian_blob`
- `--coupling`: 隣接セル間の混ざり方。`graph`、`edge`、`edge_distance`
- `--reaction`: 局所反応。`none`、`logistic`、`bell`
- `--rho-max`: 密度の上限。保存量を見たい場合は `--rho-max none`
- `--density-scale`: 表示の濃淡範囲。`auto` または `fixed`

### 表示を切り替える

`--overlay` で、通常表示以外の観察用表示に切り替えられます。

```bash
uv run voronoi-life run --overlay degree
uv run voronoi-life run --overlay alive-count
uv run voronoi-life run --overlay alive-density
uv run voronoi-life run --overlay area
uv run voronoi-life run --overlay edge-length
```

- `degree`: 各セルの近傍数
- `alive-count`: 生きている近傍セルの数
- `alive-density`: 近傍セルのうち、生きているセルの割合
- `area`: 各セルの面積
- `edge-length`: 共有境界長の合計

## テスト

```bash
uv run pytest
```

テストは、点群生成、隣接関係、更新ルール、周期境界、100ステップ以上の実行が壊れていないかを確認します。
