---
date: 2025-12-15
---

# 空明流传

!!! warning "本文正在编写中"

## 题记

最近在阅读[论文](https://arxiv.org/abs/2512.00425)的时候遇到了 Optical Flow（光流）这个名词。据导师说，光流可以达成一种“外观和动作的解耦”，“从动作之中提取更纯粹的物理规律，但不是唯一的方式”。我觉得这个算法在计算机视觉中还是挺重要的，有必要更深入地了解。这篇文章记录我的学习过程。

<!-- more -->

> 桂棹兮兰桨，击空明兮溯流光。——《赤壁赋》

空明流的名字更有诗意，听起来也很好玩的样子。很巧的是，光流中有一个很有名的算法叫做 RAFT，也和《赤壁赋》的背景比较贴切。

## 可能有用的资料

- Stanford CS231A 课程中的[光流笔记](https://web.stanford.edu/class/cs231a/course_notes/09-optical-flow.pdf)

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

通常我们对每个像素点取一个 $N\times N$ 邻域。这样，我们就得到 Lucas-Kanade 算法的全流程了。

???+notes "几个需要注意的地方"

    其实 Lucas-Kanade 算法还有不少值得深究的地方，一个值得注意的就是最小二乘估计中的“求逆”操作，$A^TA$ 并不一定是一个性态很好的矩阵。

    $$
    G = A^T A = \sum_{p\in W}\begin{bmatrix}I^2_x & I^xI^y \\ I_xI_y & I^2_y\end{bmatrix}
    $$

    矩阵 $G$ 被称为结构张量（structure tensor）或二阶矩矩阵，事实上，$G$ 就是 $W$ 所有像素 Hessian 矩阵之和。

    只有当 $G$ 可逆且为良态矩阵时，这个最小二乘估计才可以进行。这就要求其两个特征值 $\lambda_1, \lambda_2$ 都不是太小。

    而对结构矩阵 $G$ 进行分析也是很多边缘检测器的原理，比如，Shi-Tomasi 角检测器使用的就是 $\min(\lambda_1, \lambda_2) > \tau$ 来判断当前点是否为角点。Harris 角检测器使用的则是 $det(G) - k(trace(G))^2 > \tau$ 来判断，它们都反映着结构矩阵 $G$ 是否良态。

    因此 Lucas-Kanade 算法不适用与图像平坦的区域，只适用于角点的情况；Lucas-Kanade 算法的应用基本上会与角点检测结合在一起。

    其实这还是一个很深的话题，但是出于本文目的，我就不再深入了。

#### Horn-Schunck

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

这个方程组可以通过迭代法求数值解。

### 效果

==TBC==

## 深度学习中的光流

我们刚才讲了那么半天，都离不开“亮度守恒”、“连续性方程”这个条件，其实不妨更深入地想一想更本质的是什么。

更普遍的来说，光流可以看作是从两帧图像 $I_1, I_2$ 到一个位移场 $\boldsymbol{u}$ 的映射：

$$
\boldsymbol{u} = f_{\theta}(I_1, I_2)
$$

==TBC==
