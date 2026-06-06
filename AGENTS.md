# AGENTS.md - voronoi-life リポジトリ用

## 0) 位置づけとタスク管理

- このリポジトリは遊び・実験用プロジェクトとして扱う。
- 実行状態の正本は GitHub Issues / PR / GitHub Projects とする。
- GitHub Project は `Kawasakiseiji Life & Play`（https://github.com/users/kwsksj/projects/2）を使う。
- Linear は使わない。過去の外部タスクが見つかった場合も、必要なら GitHub Issue に要約してから進める。
- 小さな実験、文言修正、docs、テスト補助は GitHub Issue なしで進めてよい。継続作業、判断待ち、後で再開したい作業、開発メモとして後で見返したい内容は GitHub Issue に残す。

## 2) 実行ルール

- ユーザーの判断なしに進められる部分は、判断を仰がずに進めてよい。
- 軽微な改修は `main` へ直接コミットしてよい。ただし、文言修正、docs、テスト補助、非挙動変更の設定整理など、影響範囲が限定的で戻しやすいものに限る。
- 挙動変更や実験結果を後で追いたい変更は、GitHub Issue または PR に背景を残す。
- 削除、不可逆なデータ変更、課金影響、公開導線の大きな切替、秘密情報の新規発行・変更は、対象・影響・戻し方を示してユーザー確認を取る。

## 3) 検証

- 依存関係を更新した場合は `uv sync` で環境をそろえる。
- 通常変更では `uv run pytest` を実行する。
- CLI や出力形式を変えた場合は、必要に応じて小さな smoke test を実行する。例: `uv run voronoi-life run --cells 50 --steps 5 --seed 1`。
- UI や可視化に関わる変更では、可能な範囲で `uv run voronoi-life gui` または生成画像の表示確認も行う。
- `runs/` 配下の生成物は Git 管理しない。必要な場合は確認用のローカル出力として扱う。
