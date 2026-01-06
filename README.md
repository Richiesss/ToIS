# ToIS: 多層パーセプトロン (MLP) スクラッチ実装

NumPy (および GPU 高速化のための CuPy) のみを用いて多層パーセプトロン (MLP) をスクラッチで実装し、Streamlit を用いた GUI アプリケーションとして構築したプロジェクトです。MNIST 手書き数字認識タスクを通して、ディープラーニングの基礎的な動作原理をインタラクティブに学習・実験できます。

## 主な機能

- **MLP スクラッチ実装**: PyTorch や TensorFlow 等のフレームワークを使わず、`Forward`, `Backward`, `Optimizer` (SGD, Adam), `Layer` (Dense, ReLU, Sigmoid) をすべて自作。
- **GPU 高速化 (CuPy)**: `CuPy` を導入しており、GPU (CUDA) を利用した高速な学習が可能。サイドバーで CPU/GPU の切り替えができます。
- **インタラクティブな GUI**: Streamlit を採用し、ブラウザ上で学習の進捗 (Loss/Accuracy) をリアルタイムに可視化。
- **手書き文字認識**: Canvas に描いた数字をその場で推論し、確率分布を表示します。
- **日本語対応**: UI およびコード内のコメントはすべて日本語で記述されています。

## 必要要件

- Python 3.10 以上
- CUDA 対応 GPU (GPU モード利用時)
- 依存ライブラリ: `requirements.txt` 参照

## インストール

1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/Richiesss/ToIS.git
   cd ToIS
   ```

2. 仮想環境を作成し、依存ライブラリをインストールします。
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## 使い方

### アプリケーションの起動

起動用スクリプト `./start.sh` を使用することを推奨します。
これは、GPU 利用時に必要なライブラリパス (`LD_LIBRARY_PATH`) を自動的に設定します。

```bash
chmod +x start.sh
./start.sh
```

または、手動で以下のように実行します。

```bash
# GPU用ライブラリパスの設定 (環境に合わせて調整)
export LD_LIBRARY_PATH=$(find .venv/lib/python3.10/site-packages/nvidia -name lib -type d | paste -sd ":" -):$LD_LIBRARY_PATH
streamlit run src/app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

### 操作方法

1. **学習 (Training)**
   - サイドバーでハイパーパラメータ (隠れ層の数、学習率、Optimizer 等) を設定します。
   - 「学習開始」ボタンを押すと、MNIST データセットの学習が始まります。
   - Loss と Accuracy のグラフがリアルタイムで更新されます。

2. **推論 (Inference)**
   - 「モード選択」で「推論」を選びます。
   - 黒いエリアにマウスで数字 (0-9) を書きます。
   - 「推論実行」ボタンを押すと、予測結果と確率分布が表示されます。
   - **注意**: モデルが学習されていない（リセット直後など）場合は、先に学習を行ってください。

3. **評価 (Evaluation)**
   - テストデータを用いた混同行列 (Confusion Matrix) を表示し、モデルの性能を詳細に評価できます。

## ディレクトリ構成

- `src/`: ソースコード
  - `app.py`: Streamlit アプリケーションのエントリーポイント
  - `mlp/`: MLP ライブラリ (レイヤー, 活性化関数, オプティマイザ, バックエンドなど)
  - `utils/`: データローダーなど
- `docs/`: ドキュメント関連
- `start.sh`: 起動用スクリプト (GPUライブラリパス設定済み)

## ライセンス

MIT License
