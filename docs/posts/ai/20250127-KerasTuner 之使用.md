---
date: 2025-01-27
---

# KerasTuner 的使用

!!! warning "本文正在编写中"

!!! info "声明"

    由于该文章为本人大一时为[“科学与社会”研讨课](https://www.teach.ustc.edu.cn/?attachment_id=17309)学习需要而创作。由于作者才疏学浅，在严谨性上可能存在缺陷。

本文章为我为“科学与社会”研讨课之需要，对 [KerasTuner](https://keras.io/keras_tuner/) 的学习笔记。部分内容摘录、翻译自其[官方网站](https://keras.io/keras_tuner/)。

<!-- more -->

KerasTuner 是一个通用目的的超参数调节库，它和 Keras 的工作流有着紧密的集成。

## 对识别 Fashion MNIST 的神经网络调节超参数

以一个简单的单层神经网络为例，我们可能这样用 Keras 描述它的模型：

```py
def build_model():
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model
```

使用 KerasTuner 最简单的方法就是，用 `hp.Int()`、`hp.Float()`、`hp.Choice()` 等方法代替具体的数字，如：

```py
def build_model(hp : keras_tuner.HyperParameters):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(
        hp.Int('units', min_value=32, max_value=512),
        activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']),
    ))
    model.add(layers.Dense(10, activation='softmax'))
    return model
```

之后我们就可以用 `RandomSearch`、`BayesianOptimization`、`Hyperband` 等方法进行搜索，具体的使用方法在[这里](https://keras.io/keras_tuner/getting_started/#start-the-search)有很好的讲解，这里就不翻译了。

