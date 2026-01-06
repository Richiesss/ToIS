import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import load_mnist
from src.mlp.network import MLP
from src.mlp.optimizers import SGD, Adam
from src.mlp.backend import to_gpu, to_cpu # backend からインポート
from streamlit_drawable_canvas import st_canvas
from sklearn.metrics import confusion_matrix
import seaborn as sns
import japanize_matplotlib

st.set_page_config(page_title="MLP スクラッチ実装", layout="wide")

st.title("多層パーセプトロン (MLP) のスクラッチ実装")
st.markdown("NumPy のみで実装しています。タスク: MNIST 手書き数字認識")

# サイドバー - ハイパーパラメータ
# サイドバー - モード選択を最上部へ
st.sidebar.markdown("### モード選択")
page = st.sidebar.radio("モード (Mode)", ["学習 (Training)", "推論 (Inference)", "評価 (Evaluation)"])

# サイドバー - ハイパーパラメータ
st.sidebar.markdown("---")
st.sidebar.header("ハイパーパラメータ")

hidden_units = st.sidebar.text_input("隠れ層のユニット数 (カンマ区切り)", "128,64")
activation = st.sidebar.selectbox("活性化関数", ["relu", "sigmoid"])
optimizer_name = st.sidebar.selectbox("最適化手法 (Optimizer)", ["Adam", "SGD"])
learning_rate = st.sidebar.number_input("学習率 (Learning Rate)", value=0.001 if optimizer_name == "Adam" else 0.01, format="%.4f")
epochs = st.sidebar.number_input("エポック数 (Epochs)", min_value=1, max_value=100, value=5)
batch_size = st.sidebar.number_input("バッチサイズ (Batch Size)", min_value=1, value=64)
use_gpu_flag = st.sidebar.checkbox("GPU を使用する (CuPy)", value=True)

# データの準備
@st.cache_data
def get_data():
    return load_mnist()

if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = {'loss': [], 'acc': []}

(X_train, y_train), (X_test, y_test) = get_data()

# データの確認表示
st.sidebar.markdown("---")
st.sidebar.markdown(f"**データセット情報**")
st.sidebar.text(f"学習データ: {X_train.shape}")
st.sidebar.text(f"テストデータ: {X_test.shape}")
st.sidebar.markdown(f"クラス数: 10")


# ページ切り替えロジック
# page変数は上で定義済み


if page == "学習 (Training)":

    st.header("モデルの学習")
    
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("学習開始")
    with col2:
        reset_btn = st.button("モデルのリセット")
        
    if reset_btn:
        st.session_state.model = None
        st.session_state.history = {'loss': [], 'acc': []}
        st.success("モデルをリセットしました。")

    # リアルタイム更新用のプレースホルダー
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # チャート用のカラム (横並び) または 縦並び
    col_loss, col_acc = st.columns(2)
    with col_loss:
        st.markdown("### 損失 (Loss)")
        chart_loss = st.empty()
    with col_acc:
        st.markdown("### 精度 (Accuracy)")
        chart_acc = st.empty()
    
    if start_btn:
        # 隠れ層ユニット数のパース
        try:
            hidden_list = [int(u) for u in hidden_units.split(',')]
        except:
            st.error("隠れ層ユニット数の形式が無効です")
            st.stop()
            
        # モデルの初期化
        model = MLP(input_size=784, hidden_size_list=hidden_list, output_size=10, use_activation=activation)
        
        # Optimizer の初期化
        if optimizer_name == "SGD":
            optimizer = SGD(lr=learning_rate)
        else:
            optimizer = Adam(lr=learning_rate)
            
        st.session_state.model = model
        st.session_state.history = {'loss': [], 'acc': []}
        
        # 学習ループ
        # GPU利用フラグに応じてデータを移動
        if use_gpu_flag:
            try:
                import cupy
                X_train_exec = to_gpu(X_train)
                y_train_exec = to_gpu(y_train)
                X_test_exec = to_gpu(X_test)
                y_test_exec = to_gpu(y_test)
                st.sidebar.success("GPU モードで実行中")
            except ImportError:
                st.sidebar.error("CuPy がインストールされていないため CPU で実行します")
                X_train_exec = X_train
                y_train_exec = y_train
                X_test_exec = X_test
                y_test_exec = y_test
        else:
             X_train_exec = X_train
             y_train_exec = y_train
             X_test_exec = X_test
             y_test_exec = y_test
        
        train_size = X_train_exec.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        
        # 初期状態の精度を確認 (Epoch 0)
        initial_loss = model.loss(X_test_exec[:100], y_test_exec[:100]) # 高速化のため一部で計算
        initial_acc = model.accuracy(X_test_exec, y_test_exec)
        # to_cpu をして記録
        st.session_state.history['loss'].append(float(to_cpu(initial_loss)))
        st.session_state.history['acc'].append(float(to_cpu(initial_acc)))
        chart_loss.line_chart(st.session_state.history['loss'])
        chart_acc.line_chart(st.session_state.history['acc'])
        
        for i in range(epochs):
            status_text.text(f"エポック {i+1}/{epochs} 実行中...")
            
            # 各エポックでデータをシャッフル
            indices = np.random.permutation(train_size)
            # indices は numpy のままでよいが、GPU配列のインデックス参照に使う場合は
            # cupy の配列に対して numpy の indices を使っても機能する (cupy 互換性)
            X_train_shuffled = X_train_exec[indices]
            y_train_shuffled = y_train_exec[indices]
            
            epoch_loss = 0
            
            for j in range(iter_per_epoch):
                batch_mask = slice(j * batch_size, (j + 1) * batch_size)
                x_batch = X_train_shuffled[batch_mask]
                t_batch = y_train_shuffled[batch_mask]
                
                # 勾配計算
                params, grads = model.gradient(x_batch, t_batch)
                
                # パラメータ更新
                optimizer.update(params, grads)
                
                # 簡易的な損失追跡 (最後のバッチ)
                loss = model.loss(x_batch, t_batch)
                epoch_loss += loss
            
            # エポック終了時にテストセットでの精度を計算
            # (速度重視のためサブセットを使う場合もあるが、ここでは全件)
            test_acc = model.accuracy(X_test_exec, y_test_exec)
            avg_loss = epoch_loss / iter_per_epoch
            
            st.session_state.history['loss'].append(float(to_cpu(avg_loss)))
            st.session_state.history['acc'].append(float(to_cpu(test_acc)))
            
            progress_bar.progress((i + 1) / epochs)
            
            # グラフ更新
            # チャートごとにデータを整形して渡す必要があります
            # Streamlit の line_chart はデータフレームや辞書を受け取れます
            chart_loss.line_chart(st.session_state.history['loss'])
            chart_acc.line_chart(st.session_state.history['acc'])
            
        st.success("学習完了！")
        
    elif st.session_state.model is not None:
         # モデルが存在する場合は履歴を表示
         chart_loss.line_chart(st.session_state.history['loss'])
         chart_acc.line_chart(st.session_state.history['acc'])

elif page == "推論 (Inference)":
    st.header("手書き数字認識")
    st.markdown("下の黒いエリアにマウスで数字 (0-9) を大きく書いてください。")
    
    # 2カラムレイアウトを廃止し、縦に並べて表示安定性を向上
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 塗りつぶし (今回は線画なのであまり影響なし)
        stroke_width=20,     # 線を少し太く
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_inference", # キーを変更して強制リセット
        # update_streamlit=True は削除 (debug_canvas.py で動作確認できた設定に合わせる)
    )

    if st.button("推論実行"):
        if canvas_result.image_data is not None:
            if st.session_state.model is None:
                st.error("先にモデルを学習させてください！")
            else:
                # 前処理
                from PIL import Image
                # RGBA -> グレースケール変換時にアルファチャンネルを考慮
                img_data = canvas_result.image_data.astype('uint8')
                
                # 背景が透明(0)の場合黒(0)として扱いたいが、background_color='black' なので
                # キャンバス画像データは背景色を含んでいるはず。
                # ただし、stroke_color='white' (255)
                
                img = Image.fromarray(img_data)
                img = img.resize((28, 28))
                img = img.convert('L') # グレースケール (0-255)
                
                img_array = np.array(img)
                
                # デバッグ用に画像を表示 (オプション)
                # st.image(img_array, caption="リサイズ後の入力画像", width=100)
                
                # 正規化
                x = img_array.reshape(1, 784).astype(np.float32)
                x /= 255.0
                
                # 推論
                # モデルの重みがGPUにある場合、入力もGPUにする必要がある
                # backend.py の to_gpu は global な gpu_available フラグも見るが、まだ連動していない可能性がある
                # Dense.forward で自動転送するようにしたが、predict 呼び出し前にモデルの状態を知る術がない
                # 簡易的に、学習時の use_gpu_flag を session_state に保存しておくのが良いが、
                # 今回は Dense.forward の自動転送機能に任せる。
                # ただし、Dense.forward で `W` が GPU にあるなら入力 `x` も GPU にあるべき。
                # 現状の実装: `xp = get_array_module(x)` -> `xp` に合わせて `W` を移動。
                # つまり、`x` が CPU (numpy) なら `W` は CPU に戻ってくる。
                # 推論は CPU で十分高速かつ、Canvas 画像は CPU にあるので、そのままで OK。
                # もし GPU で推論したいなら明示的に `to_gpu(x)` する必要があるが、
                # ここでは「入力に合わせてモデルが動く」実装にしたので、そのまま CPU で推論されるはず。
                
                probs = st.session_state.model.predict(x)
                
                # 結果が GPU array の可能性があるため to_cpu
                probs = to_cpu(probs)
                
                # ソフトマックス
                probs = np.exp(probs - np.max(probs)) / np.sum(np.exp(probs - np.max(probs)))
                
                probs = probs[0]
                prediction = np.argmax(probs)
                
                st.metric("予測された数字", prediction, f"確信度: {probs[prediction]*100:.2f}%")
                
                # 棒グラフ
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(range(10), probs)
                ax.set_xticks(range(10))
                ax.set_title("確率分布")
                st.pyplot(fig)
        else:
            st.warning("キャンバスに何か書いてください。")

elif page == "評価 (Evaluation)":
    st.header("評価")
    if st.button("混同行列を計算"):
        if st.session_state.model is None:
            st.error("モデルが学習されていません。")
        else:
            # 評価用データも、モデルの状態（GPUにあるかCPUにあるか）に合わせて...
            # しかしモデルの重みは最後の学習状態（GPU）にある可能性が高い。
            # Dense.forward は「入力 x の場所」に合わせて重みを移動させる実装にした。
            # X_test (CPU) を渡せば、重みが CPU に戻ってきて計算される。
            # よって特別な処理は不要。
            
            y_pred_probs = st.session_state.model.predict(X_test)
            y_pred_probs = to_cpu(y_pred_probs)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # y_test が one-hot の場合インデックスに変換するが、ローダーは通常インデックスを返す
            # data_loader を確認。sklearn の data.target は文字列か整数。int32 に変換済み。
            # そのためラベルの 1次元配列。
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('予測ラベル (Predicted)')
            ax.set_ylabel('正解ラベル (True)')
            st.pyplot(fig)
            
            st.write(f"テスト精度: {np.mean(y_test == y_pred):.4f}")
