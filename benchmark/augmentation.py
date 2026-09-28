import tensorflow as tf


def augment_video(frames, labels):
    """
    輕量資料增強，只用於訓練集
    frames shape: (batch=1, 75, 46, 140, 1)

    原始資料為 z-score 標準化（平均 0、標準差 1），
    增強幅度必須非常小，否則訓練分布偏離驗證分布。
    移除水平翻轉：嘴唇說話有微小不對稱性，翻轉可能使模型混亂。
    """
    frames = tf.cast(frames, tf.float32)

    # 亮度微調（±3%）
    brightness_delta = tf.random.uniform((), minval=-0.03, maxval=0.03)
    frames = frames + brightness_delta

    # 極少量高斯雜訊（stddev=0.01）
    noise = tf.random.normal(tf.shape(frames), mean=0.0, stddev=0.01)
    frames = frames + noise

    return frames, labels
