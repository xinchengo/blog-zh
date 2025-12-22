---
date: 2025-12-22
---

# Neural ODE

在 Physics-informed Neural Networks (PINNs) 中，神经常微分方程（Neural Ordinary Differential Equations, Neural ODEs）是一个绕不开的话题。在阅读[NewtonGen](https://arxiv.org/abs/2509.21309) 的过程中，我觉得我有必要把 Neural ODE 是什么搞清楚，所以我以这篇文章作为笔记。

<!-- more -->

## 可能有用的资料

- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)：2018 年的原论文
- [Deep Implicit Layers](https://implicit-layers-tutorial.org/)：这是 2020 年 NeurIPS 上的一篇 Tutorial

## Neural ODE 的思想

> Neural ODE 是一种特殊 ResNet 的连续情况；Neural ODE 的梯度计算使用了伴随方程的性质。

### Neural ODE 与 ResNet

ResNet 的缘起虽然最初来自于解决“梯度消失”这个问题，但是也可以用一种“差分”的思想去理解它：

一个 $K$ 层残差神经网络定义了一个**离散动态系统**（discrete dynamical system）

$$
\mathbf{h}_{k+1}=
\mathbf{h}_k
+
\Delta t_k , f_{\theta_k}(\mathbf{h}_k),
\quad k = 0, \dots, K-1
$$

- $\mathbf{h}_k \in \mathbb{R}^d$: 第 $k$ 层的隐藏状态
- $f_{\theta_k}$: 每层都不同的神经网络
- $\Delta t_k$: 步长，一般为隐式，$=1$

而相比之下，一个 Neural ODE 定义了一个**连续动态系统**：

$$
\begin{gather}
\frac{d\mathbf{h}(t)}{dt}
=
f_\theta(\mathbf{h}(t), t),
\quad t \in [t_0, t_1] \\
\mathbf{h}(t_1)
=
\text{ODESolve}(f_\theta, \mathbf{h}(t_0)).
\end{gather}
$$

- $f_\theta$ 是一个向量场
- “深度”是连续的

#### 都是某种迭代计算

$$
\begin{aligned}
\text{ResNet:} &\quad\mathbf{h}_K = \mathbf{h}_0 + \sum_{k=0}^{K-1}\Delta t_k f_{\theta_k}(\mathbf{h}_k) \\
\text{NODE:} &\quad\mathbf{h}(t_1) = \mathbf{t_0} + \int_{t_0}^{t_1}f_{\theta}(\mathbf{h}(t),t) \mathrm dt
\end{aligned}
$$

不同的是，ResNet 的计算图是固定的，ResNet 的每一层隐状态可能是不同的空间，每一层也可能捕捉完全不同的信息；而 NODE 的步长由数值求解器（如 `odeint`）决定，所有隐状态都共享一空间。

### 伴随方程与 Neural ODE 的求导

#### 伴随方程

对于下面一个 Neural ODE：

$$
\begin{gather}
\begin{cases}
\displaystyle \frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t), \quad t \in [t_0, t_1], \\
\mathbf{h}(t_0) = \mathbf{h}_0(\mathbf{x}),
\end{cases} \\
\mathcal{L} = \ell(\mathbf{h}(t_1))\\
\end{gather}
$$

我们要求：

$$
\frac{d\mathcal L}{d\mathcal \theta},\quad\frac{d\mathcal L}{d\mathbf{h}_0}
$$

定义**伴随变量**（adjoint variable）：

$$
\mathbf{a}(t) \triangleq \frac{\partial \mathcal L}{\partial \mathbf{h}(t)} \in \mathbb R^d
$$

则该伴随变量满足以下的微分方程：

$$
\begin{cases}
\displaystyle \frac{d \mathbf{a}(t)}{dt} =-\mathbf{a}(t)^T\frac{\partial f_{\theta}(\mathbf{h}(t), t)}{\partial \mathbf{h}},\quad t\in[t_0, t_1], \\
\displaystyle\mathbf{a}(t_1) = \frac{\partial \ell}{\partial \mathbf{h}(t_1)}
\end{cases}
$$

这个方程称为**伴随方程**（adjoint equation），具体由 Neural ODE 推出伴随方程的过程比较复杂，见[数学原理](#伴随方程的推导)部分。

#### Neural ODE 的求导

由 $\mathbf{a}(t)$ 的定义，该过程就很显然了：

$$
\frac{d\mathcal L}{d\theta}=-\int_{t_1}^{t_0}\mathbf{a}(t)^T\frac{\partial f_{\theta}(\mathbf{h}(t),t)}{\partial \theta}\mathrm dt
$$

这个积分又可以使用 `odeint` 这样的微分方程库求解，这样我们就可以避免在求解器内进行梯度传播；可以认为，它是 ResNet 在时间连续时的极限：

$$
\mathbf{a}_k = \mathbf{a_{k+1}}\left(I+\Delta t\frac{\partial f}{\partial \mathbf{h}_k}\right)
$$

## Play Neural ODE

目前待完成，可以先参考 Deep Implicit Layers 上的[Chapter 3 – Neural ordinary differential equations](https://implicit-layers-tutorial.org/neural_odes) ([colab](https://colab.research.google.com/drive/1o2jWvmrQYjX99hZpTF415uT1rc35Mvft?usp=sharing))

可以看到，Neural ODE 拟合的曲线相比 ResNet 更为平滑；貌似可以更适合 Physics-informed AI 的使用。

## Neural ODE 的数学原理

### 伴随方程的推导

### Neural ODE 与基于流的生成模型

参考：[维基百科中的 Flow-based generative model](https://en.wikipedia.org/wiki/Flow-based_generative_model)

#### 流式生成模型

#### 标准流

$$
\begin{gather}
\mathbf{z}_0 \sim p_0(\mathbf{z}) \\
\mathbf{z}_1 = T(\mathbf{z_0}) \\
p_1(\mathbf{z}_1) = p_0(\mathbf{z}_0) \left|\det\frac{\partial T^{-1}}{\partial \mathbf{z}_1}\right|
\end{gather}
$$

- $p_0(\mathbf{z})$ 是一个初始分布
- $T$ 是一个**标准流**
- 第三行的公式就是概率密度变换公式

#### 连续标准流

$$
\frac{d\mathbf{z}(t)}{dt}= f_{\theta}(\mathbf{z}(t),t)
$$

设 $p(\mathbf{z},t)$ 是 $\mathbf{z}(t)$ 的概率密度函数，则 $p$ 满足连续性方程（Liouville 方程）：

$$
\frac{\partial p}{\partial t}+\nabla\cdot(pf_{\theta})=0
$$

这也是为什么这种概率模型被称为“流”，这种模型的主要思想就是通过一系列的变换将一个初始分布逐步“调整”为我们想要的分布。

$$
\frac{d}{dt}\log p(t) = -\operatorname{tr}\left(\frac{\partial f_{\theta}}{\partial z}\right)
$$

在[论文](https://arxiv.org/pdf/1806.07366)中给出了以上这个重要的定理（Instantaneous change of variables）的证明；这个定理的作用是，我们不需要计算完整的 Jacobian 矩阵，而只需要计算它的迹；

