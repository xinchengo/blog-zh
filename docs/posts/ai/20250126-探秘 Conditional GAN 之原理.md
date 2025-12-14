---
date: 2024-01-26
---

# Conditional GAN

!!! warning "本文正在编写中"

!!! info "声明"

    由于该文章为本人大一时为[“科学与社会”研讨课](https://www.teach.ustc.edu.cn/?attachment_id=17309)学习需要而创作。由于作者才疏学浅，在严谨性上可能存在缺陷。

本篇博客是我“科学与社会”研讨课的一部分。本文作为个人的学习笔记可能不是非常详细，具体请参考[链接](https://keras.io/examples/generative/conditional_gan/)。

本文介绍了 Conditional GAN，Wasserstein 比较**浅要的理解**，以及 WGAN-GP (添加梯度惩罚项的一种 Wasserstein GAN) 的**实现心得**。

<!-- more -->

## 为什么要 Conditional GAN

- GAN (Generative Adversarial Network，生成式对抗网络) 让我们很方便地使用随机信号生成各种数据（图片、视频、音频）等。
- 但是对于**有分类的数据**，比如生成手写数字 (0～9)，一个普通的 DCGAN (Deep Convolutional GAN) 就没有办法处理了。
- 当然，对于 0 到 9 每个数字都训练一个 GAN 也是可以的，但是 DCGAN 还有一个优势：比如随机生成不同年龄的人像照片，如何指定生成人像的年龄、性别等参数？
- 当然，指定生成人像的年龄和性别也可以这样做：由于生成的图像是隐向量 (latent vector) 决定的，将“老年”图片的隐向量减去“婴儿”图片的隐向量，之后就可以把图片按这个方向移动增大生成图片的“年龄”。
- Conditional GAN 的优势在于：可以支持更多的分类，可以建立参数与生成图像间的联系。

## 什么是 Conditional GAN？

### 什么是 GAN？

具体参考[Goodfellow 等人的论文](https://arxiv.org/abs/1406.2661)。

定义：

- $p_{\boldsymbol z}(\boldsymbol z)$：生成器的随机噪声
- $G(\boldsymbol z;\theta_{g})$：一个由参数 $\theta_{g}$ 定义的生成器，是一个随机向量空间到某个空间（生成数据）的映射。
- $D(\boldsymbol x;\theta_{d})$：一个由参数 $\theta_{d}$ 定义的鉴别器，是一个从某个空间（待鉴别数据）到一个 $(0,1)$ 实数的映射，表示“该数据真实的可能性”。

要求解一个最优化问题：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1 - D(G(\boldsymbol{z})))].
$$

直观地理解，就是找到这样一个 $G$，让最厉害的 $D$ 都很难分辨出分辨出数据是否是 $G$ 伪造的。

- 在实践中，往往采用“训练一步 $G$”——“训练 k 步 $D$”这样的迭代方法。
- 在训练的初期，由于生成的数据杂乱无章，$D$ 很容易分辨出假的数据，此时可以采用最大化 $\log(D(G(\boldsymbol z)))$ 来代替最小化 $\log(1-D(G(\boldsymbol z)))$。

论文中作者证明了这个最优化问题取得最小值当且仅当生成数据的分布等于真实数据的分布，而且这个函数关于生成器的参数是凸的，所以优化算法可以得到比较好的结果。

### 正题

具体参考[Mirza 的论文](https://arxiv.org/abs/1411.1784)。

- 我们想让 GAN 生成指定分类（或特征）的数据，也就是说，这个生成器**不仅**要接受一个**随机向量**，还要接受一个**与分类（特征）有关的向量**。
- 比如说，我需要指定生成衬衫、卫衣、长裤的图片，这个生成器不仅要接受随机向量，还要接受是不是衬衫，是不是卫衣，是不是长裤的向量。鉴别器也接受到了是不是衬衫，是不是卫衣，是不是长裤的这个向量。
- 所以，在 GAN 中，判别器是这样的：“这张图片好像是真的”
- 而在 Condition GAN 中：“那人告诉我这张图片是衬衫，但它不像，所以是假的”

形式化地，我们要改变一些定义：

- $G(\boldsymbol z|\boldsymbol y;\theta_{g})$：一个由参数 $\theta_{g}$ 定义的生成器，$\boldsymbol z$ 是随机向量，$\boldsymbol y$ 是附加信息。
- $D(\boldsymbol x|\boldsymbol y;\theta_{d})$：一个由参数 $\theta_{d}$ 定义的鉴别器，$\boldsymbol x$ 是待鉴别数据，$\boldsymbol y$ 是附加信息。

类似的，我们要最优化的是：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1 - D(G(\boldsymbol{z})))].
$$

### 遇到的一些问题

我按照 [Keras 网站上的教程](https://keras.io/examples/generative/conditional_gan/)书写了一个 Conditional GAN 用于模仿 [SPOTS-10](https://github.com/Amotica/SPOTS-10) 的图片，用于“科学与社会”研讨课的课题，然而训练的效果却[不尽如人意](https://github.com/xinchengo/spots/blob/0c745ad9f465861911691c4f846fd2970c0a42d7/conditional_gan.ipynb)——鉴别器的损失函数开始收敛到 0，生成器的损失函数开始发散到正无穷。

我尝试用“暂时锁住鉴别器”的方法训练若干轮的方法，让生成器的值恢复正常，之后在进行了几百轮的训练，但是结果就是生成了一批灰蒙蒙、模糊不清的图像，损失函数是收敛的，但收敛的值并不是一个最优解。

Update：

- 由于 [SPOTS-10](https://github.com/Amotica/SPOTS-10) 的数据是 $32\times 32$，而 [MNIST](https://en.wikipedia.org/wiki/MNIST_database) 的数据是 $28\times 28$ 的，所以在把适用 MNIST 的代码修改成 SPOTS-10 的版本时，我改动了鉴别器和生成器的结构。在该次训练中，损失函数发散了，
- 之后我又尝试了把 $32\times 32$ 的图片裁剪成 $28\times 28$，再在**不改动结构**的情况下直接训练。这次训练中，损失函数收敛了，但是没有收敛到最优解。

## Wasserstein GAN

以下是我读[Arjovsky 等人的原论文](https://arxiv.org/abs/1701.07875)的笔记

### 引入

根据 [Google 的教程](https://developers.google.com/machine-learning/gan/problems)，GAN 的训练常常遇到三类问题：

- 梯度消失 (vanishing gradients)：生成器不知道怎么调整——当鉴别器太好，生成器的参数不管往什么方向调整一点点，都不会改变它在鉴别器眼中的真实程度时，会发生这种情况。
- 模式坍缩 (mode collapse)：生成器只知道出几张牌，而不是生成随机图片；
- 无法收敛 (failure to converge)：损失函数发散

Wasserstein GAN 旨在解决的就是**梯度消失**和**模式坍缩**这两个问题。

### 概率密度方法与参数化方法

!!! info "以下部分内容使用 [Deepseek](https://chat.deepseek.com/) 翻译"

    由于本人对一些数学名词的正确翻译并不熟悉，以下部分内容使用 [DeepSeek](https://chat.deepseek.com/) 翻译。可能存在一些名词上的不严谨。

传统的**无监督学习**方法往往是去学习一个未知的概率密度 (probability density)。具体来说，给定一组真实数据的样本 $\{x^{(i)}\}^m_{i=1}$，传统方法会定义一个参数化的密度族 (parametric family of densities) $(P_{\theta})_{\theta\in\mathbb R^d}$，并寻找能最大化数据似然 (likelihood) 的参数：

$$
\max_{\theta\in\mathbb R^d}\frac{1}{m}\sum_{i=1}^m\log P_{\theta}(x^{(i)})
$$

如果真实数据分布 $\mathbb P_r$ 存在密度，而 $\mathbb P_{\theta}$ 是参数化密度 $\mathbb P_{\theta}$ 对应的分布，那么在大样本的情况下，这等价于最小化 **Kullback-Leibler 散度** (Kullback-Leibler divergence) $KL(\mathbb P_r||\mathbb P_{\theta})$。

然而，这种方法的一个关键问题是，**模型密度 $P_\theta$ 必须存在**。在许多实际情况下，数据分布 $\mathbb{P}_r$ 是由低维流形 (manifold) 支撑的（例如，图像数据通常位于高维像素空间中的低维流形上）。在这种情况下，模型流形和真实分布的支撑集 (support) 不太可能有显著的交集，这意味着 KL 散度可能没有定义（或者为无穷大）。

与其去估计 $\mathbb P_r$ 的密度，我们可以定义一个随机变量 $Z$，它具有固定的分布 $p(z)$，并通过一个参数化函数 $g_\theta: \mathcal{Z} \rightarrow \mathcal{X}$（通常是某种神经网络）直接生成样本，从而生成某种分布 $\mathbb{P}_\theta$。通过调整 $\theta$，我们可以改变这个分布，使其接近真实数据分布 $\mathbb{P}_r$。这种方法有两个优点：首先，与密度不同，这种方法可以表示局限于低维流形上的分布；其次，生成样本的能力通常比知道密度的数值更有用。

**变分自编码器** (Variational Auto-Encoders, VAEs) 和**生成对抗网络**都是后者的著名例子。

### 距离的选择在 GAN 中的重要性

对于 GAN，我们可以选择各种距离作为目标函数，包括 Jensen-Shannon 散度，所有 $f$-散度，以及一些奇特的组合。

[Arjovsky 等人的原论文](https://arxiv.org/abs/1701.07875)探讨的就是**衡量模型分布和真实分布接近程度**的方法，或者说，如何定义分布之间的距离或散度 $\rho(\mathbb P_{\theta},\mathbb P_r)$。这些距离的最根本区别在于它们对**概率分布序列敛散性**的影响。一个分布序列 $(\mathbb P_t)_{t\in\mathbb N}$ 收敛，当且仅当存在一个分布 $\mathbb P_{\infty}$，使得 $\rho(\mathbb P_t,\mathbb P_{\infty})$ 趋近于 $0$，这取决于距离 $\rho$ 的具体定义。非正式地说，当距离 $\rho$ 使得分布序列更容易收敛时，它诱导的拓扑结构 (topology) 更弱。

为了方便优化模型的参参数，我们希望这个模型能够让 $\theta \mapsto \mathbb P_{\theta}$ 这个映射连续，即，对于所有收敛于 $\theta$ 的子列 $\left\{\theta_t\right\}$ 都满足 $\left\{\mathbb P_{\theta_t}\right\}$ 收敛于 $\mathbb P_{\theta}$。但是，别忘了，$\left\{\mathbb P_{\theta_t}\right\}$ 是否收敛取决与我们使用的是什么距离。**距离越弱，这个分布的序列就越容易收敛**，事实上，$\rho$ 诱导拓扑结构强弱就是由收敛序列集合的关系定义的。

Wasserstein GAN 最小化的目标就是（近似的）Wasserstein 距离，这是它各种优点的来源。

### 一维情形下的不同距离

由于我的数学知识积累还比较有限，对这些特殊的数学符号还不够熟悉，所以将考虑简单情况下这些距离的定义。

1. [**总变差距离**](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures) (**Total Variation**, TV)：

   如果 $\mathbb P,\mathbb Q$ 这两个分布有着概率密度函数 $p,q$，那么它们的**总变差距离**定义为：

   $$
   \delta(\mathbb P,\mathbb Q)=\frac{1}{2}\int_{-\infty}^{+\infty} |p(x)-q(x)|\mathrm dx
   $$

   从直观上理解，如果把 $y=p(x),y=q(x)$ 画出来，它们的总变差距离就是两个函数之间包围的面积。在后面为了方便，我们又记 $\delta(\mathbb P,\mathbb Q)=||\mathbb P - \mathbb Q||_{TV}$。

2. [**Kullback-Leibler 散度**](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (KL)：

   如果 $\mathbb P,\mathbb Q$ 这两个分数有着概率密度函数 $p,q$，那么 Q 到 P 的相对熵（即 **KL 散度**）被定义为：

   $$
   \begin{aligned}
   D_{KL}(\mathbb P||\mathbb Q)&=\int_{-\infty}^{+\infty}p(x)\log\left(\frac{p(x)}{q(x)}\right)\mathrm dx \\
   &=\mathbb E_{x\sim \mathbb P}\left[\log\frac{p(x)}{q(x)}\right]
   \end{aligned}
   $$

   在这个定义下，KL 散度在很多情况下都不存在，而且即使存在也很可能不是对称的（考虑 $\exists x,q(x)=0$）。为了让这个散度变得对称，我们又定义了一个**Jensen-Shannon 散度**，即：

   $$
   \begin{gather*}
   D_{JS}(\mathbb P||\mathbb Q)=D_{KL}(\mathbb P||\mathbb M)+D_{KL}(\mathbb Q||\mathbb M)\\
   M=(\mathbb P+\mathbb Q)/2
   \end{gather*}
   $$

3. [**动土者距离**](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) (Earth Mover's distance, EMD)：

   如果 $\mathbb P,\mathbb Q$ 这两个分布有着概率密度函数 $p,q,P(x)=\int_{-\infty}^xp(t)\mathrm dt,Q(x)=\int_{-\infty}^q(t)\mathrm dt$，则它们的**动土者距离**为：

   $$
   W_1(\mathbb P,\mathbb Q)=\int_{-\infty}^{\infty}|P(x)-Q(x)|\mathrm dx
   $$

   动土者距离是计算机科学上的称谓，在数学领域，一般把这种距离称为 [**Wasserstein-1 距离**](https://en.wikipedia.org/wiki/Wasserstein_metric)或者 **Kantorovich–Rubinstein 距离**。从直观上理解，如果把 $y=p(x),y=q(x)$ 的图像画出来，动土者距离就相当于把 $p(x)$ “搬运”成 $q(x)$ 最小需要耗费的代价（只计水平方向上的）。

### 不同距离的强弱

#### 定理 1

记 $\mathbb P$ 是一个一维的概率分布，$\left\{\mathbb P_n\right\}$ 是一个概率分布的序列，$\mathbb P$ 有着概率密度函数 $p$，$\mathbb P_i$ 有着概率密度函数 $p_i$，则，考虑它们在 $n\to\infty$ 的极限：

1. 以下两个命题等价：
   - $\delta(\mathbb P_n,\mathbb P)\to 0$
   - $D_{JS}(\mathbb P_n,\mathbb P)\to 0$
2. 以下两个命题等价：
   - $W_1(\mathbb P_n,\mathbb P)\to 0$
   - $\mathbb P_n\xrightarrow{\mathcal D}\mathbb P$，其中 $\xrightarrow{\mathcal D}$ 代表**依分布收敛** (convergence in distribution)，即，$p_n(x)$ **逐点收敛**于 $p(x)$
3. 若 $D_{KL}(\mathbb P_n||\mathbb P)\to 0$ 或 $D_{KL}(\mathbb P_n||\mathbb P)\to 0$，则 1 中命题成立。
4. 若 1 中命题成立，则 2 中命题成立。

为了定理证明的方便，我们增加 $p,p_i$ 在 $\mathbb R$ 上可积的条件。

#### 证明 1

1. $(\delta(\mathbb P_n,\mathbb P)\Rightarrow D_{JS}(\mathbb P_n,\mathbb P)\to 0):$

   记 $\mathbb P_m=\frac{1}{2}\mathbb P_n+\frac{1}{2}\mathbb P$（注意这个 $\mathbb P_m$ 和 $Pn$ 是有关的），则：

   $$
   \begin{aligned}
   \delta(\mathbb P_m,\mathbb P_n) &= ||\mathbb P_m - \mathbb P_n||_{TV}\\
   &= ||\frac{1}{2}\mathbb P+\frac{1}{2}\mathbb P_n-\mathbb P_n||_{TV}\\
   &= \frac{1}{2}||\mathbb P-\mathbb P_n||_{TV}\\
   &= \frac{1}{2}\delta(\mathbb P_n,\mathbb P) \le \delta(\mathbb P_n, \mathbb P)
   \end{aligned}
   $$

   !!! warning "此处证明待完成"

### 如何使用 EM 距离

以上的定理使用[测度论](<https://en.wikipedia.org/wiki/Measure_(mathematics)>)的知识，可以扩展到更高维度的情形下仍然成立。所以，在 GAN 的训练中，使用 Wasserstein-1 距离比起 Kullback-Leibler 散度显然是一个更好的选择。现在的问题就是，如何训练一个 GAN 使得它近似地让 Wasserstein-1 距离达到最小。

在一般情况下，上文中 Wasserstein-1 距离的定义可以扩展为：(Kantorovich-Rubinstein 对偶性)

$$
W_1(\mathbb P_r,\mathbb P_{\theta}) = \sup_{||f||_L\le 1}\mathbb E_{x\sim\mathbb P_r}[f(x)]-\mathbb E_{x\sim\mathbb P_{\theta}}[f(x)]
$$

其中 $||f||_{L}\le 1$ 表示 $f$ 符合 1-Lipschitz 条件，所以，如果我们有符合 K-Lipschitz 条件的参数化函数族 $\left\{ f_{w}\right\}_{w\in\mathcal W}$，我们可以考虑最大化：

$$
\max_{w \in \mathcal W}\mathbb E_{x\sim\mathbb P_r}[f_{w}(x)]-\mathbb E_{x\sim p(x)}[f_{w}(g_{\theta}(z))]
$$

以这个作为目标函数，就相当于让 Wasserstein 距离达到最小——但是，前提是要保证鉴别器的神经网络**能够 Lipschitz 连续**。关于如何让鉴别器的神经网络 Lipschitz 连续是一个值得单独拿出来讨论的话题。在原论文中，作者的做法是把边权约束在一个指定的区间内。不过，后来也出现了效果更好的方法，比如，下面介绍的 [WGAN-GP](https://arxiv.org/abs/1704.00028) 所用的就是增添一个“梯度惩罚” (gradient penalty) 项——即，保证生成器的 L2 范数 (norm) 接近 1。

## WGAN-GP 的实现

!!! note "本文参考 [Keras 官网的教程](https://keras.io/examples/generative/wgan_gp/)"

    本文只记录作者在实现 [WGAN-GP](https://arxiv.org/abs/1704.00028) 过程中的**一些想法和体会**，以及在实现上与原教程不同的地方。原教程是一篇非常优秀的教程，如果想要系统地学习如何实现 WGAN-GP，建议从原教程而不是我浅薄的见解入手。

在[论文](https://arxiv.org/abs/1704.00028)第 4 节，作者在目标函数中加入了一项“梯度损失”，即：

$$
L=\mathbb E_{\tilde{\boldsymbol{x}}\sim\mathbb P_g}[D(\tilde{\boldsymbol{x}})]-\mathbb E_{\boldsymbol{x}\sim\mathbb P_r}[D(\boldsymbol{x})]+\lambda\mathbb E_{\hat{\boldsymbol{x}}\sim\mathbb P_{\hat{\boldsymbol{x}}}}\left[(||\nabla_{\hat{\boldsymbol{x}}}D(\hat{\boldsymbol{x}}||_2-1)^2\right]
$$

其中这个 $\hat{\boldsymbol{x}}$ 是一个随机的在真实和虚假照片中插值得到的照片。

## 总结

- Conditonal GAN：和一般的 GAN 一样，但是在生成器和鉴别器的输入中**加入附加信息**。
- Wasserstein GAN：
  - 传统的 GAN 训练的**目标函数**相当于让生成器分布与真实分布的 **Kullback-Leibler 散度**最小。
  - 但是 Kullback-Leibler 散度是一个很强的距离，实践中很难对其进行优化。
  - 使用 **Wasserstein-1 距离**作为替代会让训练简单很多。
  - 实践上，需要在**改变目标函数**的同时，保障鉴别器的神经网络符合 **Lipschitz 条件**。

