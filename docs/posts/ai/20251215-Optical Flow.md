---
date: 2025-12-15
---

# 空明流传

## 题记

最近在阅读[论文](https://arxiv.org/abs/2512.00425)的时候遇到了 Optical Flow（光流）这个名词。据导师说，光流可以达成一种“外观和动作的解耦”，“从动作之中提取更纯粹的物理规律，但不是唯一的方式”。我觉得这个算法在计算机视觉中还是挺重要的，有必要更深入地了解。这篇文章记录我的学习过程。

<!-- more -->

> 桂棹兮兰桨，击空明兮溯流光。——《赤壁赋》

空明流的名字更有诗意，听起来也很好玩的样子。很巧的是，光流中有一个很有名的算法叫做 RAFT，也和《赤壁赋》的背景比较贴切。

!!! info "关于该文章的书写进度"

    这片文章的内容可能有些支离破碎，以后也不一定能完成，也许再也不会更新；欢迎在评论区留言交流，也许这会激励我继续写下去。

## 可能有用的资料

- Stanford CS231A 课程中的[光流笔记](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)
- RAFT 论文：[RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- 一个 Github 仓库：[Awesome Optical Flow](https://github.com/hzwer/Awesome-Optical-Flow)
- OpenCV 的[光流教程](https://docs.opencv.org/4.12.0/d4/dee/tutorial_optical_flow.html)
- [Open MMlab](https://github.com/open-mmlab/mmflow)：一个光流工具箱

### 一些不太有用的资料

- [Bad Apple!! 暂停就看不见了](https://www.bilibili.com/video/BV1gs411278Y)：虽然是 2015 年的老视频，但是思想却极其先进，可以体现 Warped Noise 的基本思想；不知道 Wraped Noise 是怎么被发明的，是不是刷 B 站猎奇视频突然想到的；
- [Go with the flow](https://eyeline-labs.github.io/Go-with-the-Flow/)；

## 光流入门

### 连续性方程

> 对，亮度也可以看成一种密度。

视频中连续帧的图像满足什么样的性质呢？用物理上的思路，我们尝试在其中构造出一种“守恒”的量。

比如“亮度守恒”（brightness constancy），假设图片中的物体只是移动而没有发生任何其他的变化（这当然是非常理想的情况），在这个情况下，移动前的像素点亮度和移动后的像素点亮度不发生变化：

$$
\begin{aligned}
I(x, y, t) &= I(x + \Delta x, y +  \Delta y, t + \Delta t) \\
&= I(x, y, t) + \nabla I\cdot (\Delta x, \Delta y, \Delta t) + \cdots \\
&\approx I + \nabla I \cdot (\Delta x, \Delta y, \Delta t)
\end{aligned}
$$

在小运动假设下（small motion assumption），$\Delta t \to 0$，上面的一阶近似是可行的，可以推出：

$$
\begin{aligned}
0 &= \nabla I \cdot (\Delta x, \Delta y, \Delta z)\\
&= \frac{\partial I}{\partial x}\frac{\Delta x}{\Delta t} + \frac{\partial I}{\partial y} \frac{\Delta y}{\Delta t} + \frac{\partial I}{\partial t} \\
&= I_x u + I_y v + I_t \\
\implies & \nabla I \cdot \boldsymbol{u}+\frac{\partial I}{\partial t} = 0
\end{aligned}
$$

这个散度的形式可以看作物理中各种“流函数”的一种推广，可以类比流体力学中的连续性方程：

$$
\nabla \cdot (\rho\boldsymbol{v}) + \frac{\partial\rho}{\partial t} = 0
$$

### 连续性方程的局限

当然，光靠这个连续性方程解不出这个 $I$，事实上，它只确定了**法向流**（normal flow），即，$\boldsymbol{u}$ 在 $\nabla I$ 方向的分量。

??? notes "怎么求法向流？"

    $$
    \frac{\nabla I}{\| \nabla I \|} \cdot \boldsymbol{u} = \frac{-I_t}{\| \nabla I \|}
    $$

    左边是 $\boldsymbol{u}$ 在梯度方向的法向分量，而右边就是我们想求的值。

我们难以确定切向分量，这个问题被称为光圈问题（aperture problem）。为了解决这个问题，很多不同的方法都被提出。

### 额外的方程？

类比流体的粘滞性，这些方程大致的出发点都是——相邻的像素点的速度具有某种接近的关系。

> 简要概括：
>
> - 连续性 + 空域光滑性：Lucas-Kanade 方法
> - 连续性 + 全局光滑性：Horn-Schunck 方法
> - 把一阶近似换成二阶近似：Gunnar-Farneback 方法

Lucas-Kanade 和 Gunnar-Farneback 方法都在 [OpenCV](https://opencv.org/) 的[标准库](https://docs.opencv.org/4.12.0/d4/dee/tutorial_optical_flow.html)中有实现，而 Horn-Schunck 方法由于效果不佳，OpenCV 已经不再支持。

Lucas-Kanade 由于其模型过于理想化，只有在角点处才能得到较好的效果，而 Gunnar-Farneback 方法则可以得到稠密的光流场。

#### Lucas-Kanade

!!! notes "翻译提醒"

    注意翻译成中文应翻译成“卢卡斯-金出方法”，因为 Kanade 指日本计算机学家[金出武雄](https://en.wikipedia.org/wiki/Takeo_Kanade)。

引入空域光滑性前提（spatial smoothness assumption），我们认为对于一个小的“窗口”中的所有像素，它们共享一个光流 $\boldsymbol{u}$，而我们要对这个光流做最小二乘估计，形式化地：

$$
\begin{gather}
\arg \min_{\boldsymbol{u}} \|A\boldsymbol{u} - \boldsymbol{b}\| \\
A \triangleq
  \begin{bmatrix}
    I_x(p_1) & I_y(p_1) \\
    \vdots & \vdots \\
    I_x(p_n) & I_y(p_n)
  \end{bmatrix}, \quad
B \triangleq
  \begin{bmatrix}
    - I_t(p_1) \\ \vdots \\ - I_t(p_n)
  \end{bmatrix}
\end{gather}
$$

这可以看作让窗口中的每个点都尽可能满足“连续性方程”（由于假设了光流相等，这些方程一般不能同时被满足），其最小二乘估计为：

$$
\boldsymbol{u} = (A^TA)^{-1}A^T\boldsymbol{b}
$$

通常我们对每个像素点取一个 $N\times N$ 邻域。这样，我们就得到 Lucas-Kanade 算法的全流程了。这个算法有一些值得深究的地方，详见[附录](#Lucas-Kanade-与结构张量)

#### Gunnar-Farneback

> 使用了二阶近似，但失去了“连续性方程”的形式。

有时候，一阶近似并不够好，Gunnar-Farneback 方法使用二阶近似：

$$
\begin{aligned}
I(\boldsymbol{x}_0 + \boldsymbol{\delta}) &\approx I(\boldsymbol{x}_0) + \nabla I(\boldsymbol{x}_0)^T \boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^T \nabla^2 I(\boldsymbol{x}_0) \boldsymbol{\delta} \\
&\triangleq \boldsymbol{\delta}^T A \boldsymbol{\delta} + b^T \boldsymbol{\delta} + c
\end{aligned}
$$

对于经过平移 $\boldsymbol{d}$ 后的图像，有：

$$
\begin{gather}
\begin{aligned}
I_2(\boldsymbol{x}) &= I_1(\boldsymbol{x} - \boldsymbol{d}) \\
&\approx (\boldsymbol{x} - \boldsymbol{d})^T A_1 (\boldsymbol{x} - \boldsymbol{d}) + b_1^T (\boldsymbol{x} - \boldsymbol{d}) + c_1 \\
&= \boldsymbol{x}^T A_1 \boldsymbol{x} + (b_1 - 2A_1\boldsymbol{d})^T \boldsymbol{x} + \left(\boldsymbol{d}^T A_1 \boldsymbol{d} - b_1^T \boldsymbol{d} + c_1\right) \\
&= \boldsymbol{x}^T A_2 \boldsymbol{x} + b_2^T \boldsymbol{x} + c_2
\end{aligned} \\
\implies A_2 \approx A_1, \quad b_2 \approx b_1 - 2A_1\boldsymbol{d} \\
\implies \boldsymbol{\hat{d}} = -\frac{1}{2} A_1^{-1}(b_2 - b_1)
\end{gather}
$$

这样我们就得到了 Gunnar-Farneback 算法的计算公式。在实践中，类似 Lucas-Kanade 方法，对每个像素点取一个邻域对梯度和 Hessian 矩阵进行平滑处理。

由于引入了二阶近似，Gunnar-Farneback 方法可以用于处理稠密的光流场，而不需要像 Lucas-Kanade 方法那样依赖角点检测。

#### Horn-Schunck

在实践中，Horn-Schunck 算法的表现并不优秀，往往不如 Gunnar-Farneback 算法，所以，OpenCV 在新的版本中只保留了 Gunnara-Farneback 算法。尽管如此，它的数学原理还是很有启发性的。

> 水很深，关于 Euler-Lagrange 方程的内容，之后有缘再说吧。

Horn-Schunck 方法则是引入了全局光滑性前提（global smoothness assumption），它假设整幅图像的光流场都是光滑的。

它最小化一个能量泛函（energy functional）：

$$
E(u, v) = \iint \left( (I_x u + I_y v + I_t)^2 + \lambda \left( \|\nabla u\|^2 + \|\nabla v\|^2 \right) \right) \mathrm d x \mathrm d y
$$

由变分法理论可知，能量泛函的极小值满足欧拉-拉格朗日方程（Euler-Lagrange equation）：

$$
\begin{aligned}
I_x (I_x u + I_y v + I_t) - \lambda \nabla^2 u &= 0 \\
I_y (I_x u + I_y v + I_t) - \lambda \nabla^2 v &= 0
\end{aligned}
$$

这个方程组可以通过 Gauss-Seidel 迭代法或者 Jacobi 迭代法来求解。

==TBC==

## 深度学习中的光流

我们刚才讲了那么半天，都离不开“亮度守恒”、“连续性方程”这个条件，其实不妨更深入地想一想更本质的是什么。

更普遍的来说，光流可以看作是从两帧图像 $I_1, I_2$ 到一个位移场 $\boldsymbol{u}$ 的映射：

$$
\boldsymbol{u} = f_{\theta}(I_1, I_2)
$$

在传统方法中，$f_\theta$ 由一些假设（亮度守恒、光滑性等）和优化方法（最小二乘、变分法等）所决定，而基于深度学习的光流方法中 $f_\theta$ 是一个神经网络，而我们通过设计损失函数来约束传统方法中的那些假设。

### 损失函数的设计

损失函数的设计主要是给出约束条件，可以类比传统方法中的各种“假设”“方程”。

#### 光度损失

光度损失（photometric loss）对应传统方法中的亮度守恒假设。“光度”和“亮度”是同义词：

$$
\mathcal{L}_{\text{photo}} = \sum_{x}\rho(I_1(x) - I_2(x + \boldsymbol{u}(x)))
$$

#### 光滑损失

光滑损失对应 Horn-Schunck 方法中的全局光滑性假设：

$$
\mathcal{L}_{\text{smooth}} = \sum_{x}\left(|\nabla u(x)| + |\nabla u(x)|\right)
$$

对比 Horn-Schunck 方法中的能量泛函，我们发现这是一个总变差正则项（total variation regularizer），使用的是 L1 范数。传统方法使用 L2 范数的主要原因是数学上的方便（它是一个简单扩散方程的形式）；深度学习方法中不需要考虑这种问题，可以单纯考虑效果。

为什么 L2 光滑项不好呢？

- 运动场（motion field）是分段连续，而不是全局连续的，L2 范数会过度惩罚大梯度，从而导致边缘处的模糊；
- L2 范数假设 $\nabla u\sim \mathcal{N}(0, \sigma^2)$，但这在该问题下是错误的；

归根结底来说，这个范数选择是 Bayes 统计中先验分布选择的问题，L2 范数对应高斯先验，而 L1 范数对应拉普拉斯先验，Bayes 统计在《概率论与数理统计》中并没有涵盖，这一部分详见[附录](#最大后验估计)。

在实际应用中，光滑损失通常会结合图像的边缘信息进行加权，称为边缘觉知光滑损失（edge-aware smoothness loss）：

$$
\mathcal{L}_{\text{smooth}} = \sum_{x} |\nabla u(x)| e^{-|\nabla I(x)|}
$$

边缘附近，$|\nabla I|$ 较大，光滑损失的权重较小，从而允许较大的梯度，这就是“边缘觉知”的含义。

这可以看作一种对各向异性扩散（anisotropic diffusion）的模拟；相比之下，传统的 Horn-Schunck 方法使用的是各向同性扩散（isotropic diffusion）。

### 光流的匹配视角

光流的定义是一个两帧图像到位移场的映射，比较早期的深度学习方法大多是直接建立这种映射关系，比如 FlowNet。 FlowNet (2015) 采取的是建立直接的端到端的卷积神经网络架构：

$$
\operatorname{CNN}(I_1, I_2) \to \boldsymbol{u}
$$

然而，光流也可以从一种匹配（matching）的视角来理解。定义代价函数 $C(x, d)$ 来表示像素 $x$ 在视差 $d$ 下的匹配代价，光流预测的问题就变成了在每个像素点 $x$ 处寻找一个最优的视差 $d^*$：

$$
C(x, d) = \operatorname{sim}(F_1(x), F_2(x + d))
$$

其中 $F_1, F_2$ 是从图像 $I_1, I_2$ 中提取的特征，而不是像素本身。代价函数作为一个三维张量被称为代价体（cost volume），它的三个维度分别是图像的宽、高和视差。

从这种匹配的视角出发，研究光流的求解方法就更为方便。很多后续的方法都是基于该思想提出的。

## RAFT 简介

!!! info "该部分由于作者才疏学浅，只做了非常浅表的介绍"

由于这一部分内容太多，太深，我感觉自己目前不是有很多精力把它钻研透，在这里只写一个简单的介绍。

> RAFT 使用一个在全分辨率、全对关系的代价体上推理的循环神经网络对一个稠密的光流场进行迭代更新。

RAFT 算法的核心思想是不对每两帧图像对光流给出独立的预测，而是对其进行迭代更新。

### RAFT 的架构

- 特征提取器（Feature Extractor）：使用卷积神经网络从输入的两帧图像 $I_1, I_2\in \mathbb{R}^{H\times W\times 3}$ 中提取特征 $F_1, F_2\in \mathbb{R}^{h\times w\times c}$。
- 全对关系体（All-Pairs Correlation Volume）：$\operatorname{Corr}(i,j,i^\prime, j^\prime) = F_1(i,j)^T F_2(i^\prime, j^\prime)$，它们构成一个四维张量 $\mathbb{R}^{h\times w\times h\times w}$；
- 循环更新算子（Recurrent Update Operator）：使用一个循环神经网络（RNN）对光流场进行迭代更新。

$$
\begin{gather}
\Delta \boldsymbol{u}_k = \operatorname{UpdateNet}(\boldsymbol{u}_k, \operatorname{Corr}, \text{其他信息}) \\
\boldsymbol{u}_{k+1} = \boldsymbol{u}_k + \Delta \boldsymbol{u}_k
\end{gather}
$$

RAFT 的良好表现很大程度就来源于这个迭代、渐进的过程。

### RAFT 与传统方法的联系

| 特点   | Horn-Schunck | RAFT         |
| ------ | ------------ | ------------ |
| 匹配   | 全局         | 全对         |
| 正则化 | 显式         | 隐式         |
| 优化   | 偏微分方程   | 循环神经网络 |
| 分辨率 | 像素级别     | 全分辨率     |

## 光流的实现和可视化

### OpenCV

## 附录

### Lucas-Kanade 与结构张量

其实 Lucas-Kanade 算法还有不少值得深究的地方，一个值得注意的就是最小二乘估计中的“求逆”操作，$A^TA$ 并不一定是一个性态很好的矩阵。

$$
G = A^T A = \sum_{p\in W}\begin{bmatrix}I^2_x & I^xI^y \\ I_xI_y & I^2_y\end{bmatrix}
$$

矩阵 $G$ 被称为结构张量（structure tensor）或二阶矩矩阵，事实上，$G$ 就是 $W$ 所有像素 Hessian 矩阵之和。

只有当 $G$ 可逆且为良态矩阵时，这个最小二乘估计才可以进行。这就要求其两个特征值 $\lambda_1, \lambda_2$ 都不是太小。

而对结构矩阵 $G$ 进行分析也是很多边缘检测器的原理，比如，Shi-Tomasi 角检测器使用的就是 $\min(\lambda_1, \lambda_2) > \tau$ 来判断当前点是否为角点。Harris 角检测器使用的则是 $\det(G) - k(\operatorname{tr}(G))^2 > \tau$ 来判断，它们都反映着结构矩阵 $G$ 是否良态。

因此 Lucas-Kanade 算法不适用与图像平坦的区域，只适用于角点的情况；Lucas-Kanade 算法的应用基本上会与角点检测结合在一起。

其实这还是一个很深的话题，但是出于本文目的，我就不再深入了。

### 最大后验估计

!!! warning "该部分尚未完成"

我们很熟悉最大似然估计（Maximum Likelihood Estimation, MLE），而最大后验估计（Maximum A Posteriori Estimation, MAP）则是在提供先验分布的情况下，用贝叶斯定理进行的修正。

$$
x_{\text{MAP}} = \arg \max_{x} p(x | y) = \arg \max_{x} \frac{p(y | x)p(x)}{p(y)} = \arg \max_{x} p(y | x)p(x)
$$

与 MLE 不同的是，MAP 中要最大化的是后验概率，而不是“似然”，它的对数后验概率为：

$$
\log p(x | y) = \log p(y | x) + \log p(x) + C
$$

形式上，这等同于最大似然估计加上一个先验的正则项，告诉我们未知的 $x$ 的某些认知。

我们假设观测值 $y$ 由真实值 $x$ 加上噪声 $\epsilon$ 得到，现在考虑噪声服从不同的分布时，MAP 的形式。在光流这个任务中，我们想要预测的是 $\boldsymbol{u}$，而噪声是 $\nabla \boldsymbol{u}$。

$$
\begin{gather}
y = f(x) + \epsilon \\
p(\epsilon) \triangleq p(\epsilon = y - f(x)) = p(y | x)
\end{gather}
$$

#### Gaussian 先验

我们现在认为参数 $\epsilon$ 服从高斯分布 $\mathcal{N}(0, \sigma^2)$，则：

$$
\begin{gather}
p(\epsilon) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\epsilon^2}{2\sigma^2}\right) \\
\implies -\log p(\epsilon) = \frac{\epsilon^2}{2\sigma^2} + C \\
\implies -\log p(\epsilon) \propto \epsilon^2
\end{gather}
$$

注意 Gaussian 先验中的“先验”对应的其实是 MAP 中的“似然”部分。我们下面要说明，最小化 $\|x-y\|

#### Laplace 先验
