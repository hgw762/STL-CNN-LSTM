import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
np.random.seed(42)
tf.random.set_seed(42)

class CNNLSTM:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def stl_decomposition(self, data, period=10):
        """STL分解获取多维特征"""
        result = STL(data, period=period).fit()
        return np.column_stack([result.trend, result.seasonal, result.resid])

    def load_data(self, file_path):
        """加载股票数据"""
        df = pd.read_excel(file_path)
        return df['收盘'].values.astype(float)

    def preprocess_data(self, data):
        data = pd.Series(data).ffill().values  # 修复fillna警告
        decomposed = self.stl_decomposition(data)
        # 验证分解重构误差（调试用）
        reconstruct = decomposed.sum(axis=1)
        print(f"STL Reconstruction MAE: {np.mean(np.abs(data - reconstruct)):.4f}")
        scaled_data = self.scaler.fit_transform(decomposed)
        return scaled_data

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length - 1):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])  # 预测三个分量
        X, y = np.array(X), np.array(y)
        # 添加随机打乱（需保持时序关系）
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def train_test_split(self, X, y):
        train_idx = int(0.8 * len(X))  # 调整划分比例
        val_idx = int(0.9 * len(X))
        return (X[:train_idx], X[train_idx:val_idx], X[val_idx:],
                y[:train_idx], y[train_idx:val_idx], y[val_idx:])

    def build_model(self):
        inputs = keras.Input(shape=(self.seq_length, 3))

        # 共享特征提取层
        x = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)  # 降低Dropout比例
        x = Conv1D(64, 3, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # 分量化预测分支
        trend = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))(x)
        trend = Dense(64, activation='relu')(trend)
        trend = Dense(1, name='trend')(trend)

        seasonal = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))(x)
        seasonal = Dense(64, activation='relu')(seasonal)
        seasonal = Dense(1, name='seasonal')(seasonal)

        resid = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))(x)
        resid = Dense(64, activation='relu')(resid)
        resid = Dense(1, name='resid')(resid)

        outputs = keras.layers.concatenate([trend, seasonal, resid])
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),  # 增大学习率
            loss='mse',
            metrics=['mae']
        )
        return model


# 超参数配置
SEQ_LENGTH = 15
DATA_PATH = r'D:\601318历史数据2.xlsx'

# 初始化模型
model_wrapper = CNNLSTM(seq_length=SEQ_LENGTH)

# 数据管道
raw_data = model_wrapper.load_data(DATA_PATH)
processed_data = model_wrapper.preprocess_data(raw_data)
X, y = model_wrapper.create_sequences(processed_data)

# 数据集划分
X_train, X_val, X_test, y_train, y_val, y_test = model_wrapper.train_test_split(X, y)

# 构建模型
model = model_wrapper.build_model()

# 训练配置
callbacks = [
    EarlyStopping(patience=50, restore_best_weights=True),  # 减少耐心值
    ReduceLROnPlateau(factor=0.5, patience=20, verbose=1)   # 添加学习率衰减提示
]

# 模型训练
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# 保存模型
model.save('cnn_lstm_stock.keras')

# 测试集预测
y_pred = model.predict(X_test)
y_pred_inv = model_wrapper.scaler.inverse_transform(y_pred)  # 逆变换三个分量

# 重构预测值（趋势+季节+残差）
predicted_price = y_pred_inv.sum(axis=1) # 将三个分量相加得到最终预测值

# 真实值处理
y_test_inv = model_wrapper.scaler.inverse_transform(y_test)
true_price = y_test_inv.sum(axis=1)

# 评估指标
rmse = np.sqrt(np.mean((true_price - predicted_price) ** 2))
mae = np.mean(np.abs(true_price - predicted_price))
print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')

# 可视化预测结果
plt.figure(figsize=(14, 6))
plt.plot(true_price, label='真实股票', marker='o', markersize=4)
plt.plot(predicted_price, label='预测股票', linestyle='--', marker='x', markersize=4)
plt.title('中国平安CNN-LSTM股价预测（STL分解集成）')
plt.xlabel('时间步')
plt.ylabel('股票 (人民币)')
plt.grid(True)
plt.legend()
plt.show()