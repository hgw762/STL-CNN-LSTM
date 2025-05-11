import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten, GRU
from tensorflow.keras.regularizers import l2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN提示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 隐藏INFO日志
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
tf.random.set_seed(42)


class ModelComparator:
    def __init__(self, seq_length, data_path):
        self.seq_length = seq_length
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._prepare_data()

    def _stl_decomposition(self, data, period=10):
        """统一STL分解处理"""
        result = STL(data, period=period).fit()
        return np.column_stack([result.trend, result.seasonal, result.resid])

    def _prepare_data(self):
        """统一数据预处理流程"""
        # 加载数据
        df = pd.read_excel(self.data_path)
        raw_data = df['收盘'].values.astype(float)
        data = pd.Series(raw_data).ffill().values

        # STL分解
        decomposed = self._stl_decomposition(data)
        scaled_data = self.scaler.fit_transform(decomposed)

        # 创建序列
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length - 1):
            X.append(scaled_data[i:i + self.seq_length])
            y.append(scaled_data[i + self.seq_length])

        # 数据集划分
        X, y = np.array(X), np.array(y)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        self.X_train = X[:int(0.8 * len(X))]
        self.X_val = X[int(0.8 * len(X)):int(0.9 * len(X))]
        self.X_test = X[int(0.9 * len(X)):]
        self.y_train = y[:int(0.8 * len(y))]
        self.y_val = y[int(0.8 * len(y)):int(0.9 * len(y))]
        self.y_test = y[int(0.9 * len(y)):]

    def build_stl_cnn_lstm(self):
        """原始STL-CNN-LSTM模型"""
        inputs = keras.Input(shape=(self.seq_length, 3))
        x = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        trend = LSTM(128, kernel_regularizer=l2(0.001))(x)
        trend = Dense(64, activation='relu')(trend)
        trend = Dense(1, name='trend')(trend)

        seasonal = LSTM(128, kernel_regularizer=l2(0.001))(x)
        seasonal = Dense(64, activation='relu')(seasonal)
        seasonal = Dense(1, name='seasonal')(seasonal)

        resid = LSTM(128, kernel_regularizer=l2(0.001))(x)
        resid = Dense(64, activation='relu')(resid)
        resid = Dense(1, name='resid')(resid)

        outputs = keras.layers.concatenate([trend, seasonal, resid])
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


    def build_pure_cnn(self):
        """纯CNN模型"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, 3)),  # 添加Input层
            Conv1D(128, 5, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_pure_rnn(self):
        """纯RNN模型（使用GRU）"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, 3)),  # 添加Input层
            GRU(128, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64),
            Dense(64, activation='relu'),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_cnn_rnn(self):
        """CNN-RNN混合模型"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.seq_length, 3)),  # 添加Input层
            Conv1D(128, 5, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(64, activation='relu'),
            Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _evaluate_model(self, model, history):
        """统一评估和逆变换"""
        # 验证集损失曲线数据
        val_loss = history.history['val_loss']

        # 测试集评估
        y_pred = model.predict(self.X_test)
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        predicted_price = y_pred_inv.sum(axis=1)

        y_test_inv = self.scaler.inverse_transform(self.y_test)
        true_price = y_test_inv.sum(axis=1)

        rmse = np.sqrt(np.mean((true_price - predicted_price) ** 2))
        mae = np.mean(np.abs(true_price - predicted_price))
        return val_loss, rmse, mae

    def run_comparison(self):
        """执行对比实验"""
        models = {
            "STL-CNN-LSTM": self.build_stl_cnn_lstm(),
            "Pure-CNN": self.build_pure_cnn(),
            "Pure-RNN": self.build_pure_rnn(),
            "CNN-RNN": self.build_cnn_rnn()
        }

        results = {}
        plt.figure(figsize=(12, 6))

        for name, model in models.items():
            print(f"\nTraining {name}...")
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=500,
                batch_size=64,
                callbacks=[
                    EarlyStopping(patience=50),
                    ReduceLROnPlateau(factor=0.5, patience=20,verbose=1)
                ]

            )

            # 记录结果
            val_loss, rmse, mae = self._evaluate_model(model, history)
            results[name] = (rmse, mae, val_loss)

            # 绘制验证损失曲线
            plt.plot(history.history['val_loss'], label=f'{name}', linestyle='--' if "Pure" in name else '-')

        # 可视化设置
        plt.title('（贵州茅台）不同模型验证损失对比')
        plt.xlabel('训练轮数')
        plt.ylabel('验证损失值')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 输出性能指标
        print("\n表现对比:")
        print("{:<15} {:<10} {:<10}".format('Model', 'RMSE', 'MAE'))
        for name, (rmse, mae, _) in results.items():
            print("{:<15} {:.4f}    {:.4f}".format(name, rmse, mae))

        return results


# 运行对比实验
if __name__ == "__main__":
    comparator = ModelComparator(
        seq_length=15,
        data_path=r'D:\600519历史数据2.xlsx'
    )
    comparison_results = comparator.run_comparison()