---
title: 概率机器人 PART I
date: 2022-10-05 22:57:59
tags:
- SLAM
mathjax: true
---

## 第二章 递归状态估计
### 2.2 概率的基本概念

**高斯分布**
$$
\begin{aligned}
    p(x) &= \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \\
    p(\mathbf{x}) &= \frac{1}{\det {\sqrt{2 \pi \boldsymbol{\Sigma}}}}e^{-\frac{1}{2}{(\mathbf{x} - \boldsymbol{\mu})^T}\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})} 
\end{aligned}
$$

**条件独立与绝对独立之间既不充分也不必要**
$$
\begin{aligned}
    & p(x, y | z ) = p(x | z) p(y | z) \nRightarrow p(x, y) = p(x) p(y) \\
    & p(x, y) = p(x) p(y)  \nRightarrow p(x, y | z ) = p(x | z) p(y | z) 
\end{aligned}
$$

**期望与协方差**
$$
E[X] = \int xp(x)dx \\
E[aX + b] = a E[X] + b \\
D[X] = E[(X - E[X])^2] = E[X^2] - E^2[X]
$$

**熵**
$$
H(x) = -E[\log_2p(x)] 
$$

### 2.3 机器人环境交互
#### 2.3.1 状态
机器人系统中典型状态:
- 机器人位姿
- 机器人运动学状态，一般指关节的转动角度
- 机器人速度、角速度
- 环境中物体的位置和特征
- 移动的物体和人的位置和状态
- 潜在的其他状态量，如传感器是否故障

#### 2.3.2 环境交互
两种交互类型: 机器人通过执行机构影响环境的状态与它通过传感器收集有关状态的信息

#### 2.3.3 概率生成法则
马尔可夫性，可以称为隐马尔可夫模型(HMM)或者动态贝叶斯网络(DBN)
$$
\begin{aligned}
    p(x_t | x_{0:t-1}, z_{1:t-1}, u_{1:t}) &= p(x_t | x_{t - 1}, u_t) \\
    p(z_t | x_{0:t-1}, z_{1:t-1}, u_{1:t}) & = p(z_t | x_t)
\end{aligned}
$$

#### 2.3.4 置信分布
置信分布可以理解为状态量的后验分布，有两种后验分布
$$
\begin{aligned}
\overline{\mathrm{bel}}(x_t) &= p(x_t | z_{1:t-1}, u_{1:t}) \\
\mathrm{bel}(x_t) &= p(x_t | z_{1:t}, u_{1:t})
\end{aligned}
$$
第一种后验计算过程被称为预测，而第二种后验被称为修正或者测量更新，这两个概率为卡尔曼滤波的核心

### 2.4 贝叶斯滤波

#### 2.4.1 贝叶斯滤波算法

```
Algo Bayes_filter(bel(x(t-1)), u(t), z(t)):
    for all x(t) do:
        bel_overline(x(t)) = integrate(p(x(t) | u(t), x(t - 1)) bel(x(t - 1)) dx(t-1))
        bel_(x(t)) = eta p((z(t) | x(t)))  bel_overline(x(t))
    endfor
    return bel_(x(t))
```

#### 2.4.2 实例

假设有一个估计门开关的传感器，初始的门开关与否的先验相同，即
$$
p(X_0 = \mathrm{open}) = 0.5 \\
p(X_0 = \mathrm{close}) = 0.5
$$
且有测量模型
$$
p(Z_t = \mathrm{sense\_open} | X_t = \mathrm{open}) = 0.6 \\ 
p(Z_t = \mathrm{sense\_close} | X_t = \mathrm{open}) = 0.4 \\
p(Z_t = \mathrm{sense\_open} | X_t = \mathrm{close}) = 0.2 \\
p(Z_t = \mathrm{sense\_close} | X_t = \mathrm{close}) = 0.8
$$
机器人有推门或者啥也不干的操作，有如下的状态转移概率
$$
p(X_t = \mathrm{open} | U_t = \mathrm{push}, X_{t-1} = \mathrm{open})  = 1 \\
p(X_t = \mathrm{close} | U_t = \mathrm{push}, X_{t-1} = \mathrm{open})  = 0 \\
p(X_t = \mathrm{open} | U_t = \mathrm{push}, X_{t-1} = \mathrm{close})  = 0.8 \\
p(X_t = \mathrm{close} | U_t = \mathrm{push}, X_{t-1} = \mathrm{close})  = 0.2 \\
p(X_t = \mathrm{open} | U_t = \mathrm{do \ nothing}, X_{t-1} = \mathrm{open})  = 1 \\
p(X_t = \mathrm{close} | U_t = \mathrm{do \ nothing}, X_{t-1} = \mathrm{open})  = 0 \\
p(X_t = \mathrm{open} | U_t = \mathrm{do \ nothing}, X_{t-1} = \mathrm{close})  = 0 \\
p(X_t = \mathrm{close} | U_t = \mathrm{do \ nothing}, X_{t-1} = \mathrm{close})  = 1
$$

假设在$t_1$时刻，机器人没有采用任何控制动作，但传感器检测到门时开的，此时有贝叶斯滤波计算结果如下
$$
\begin{aligned}
\overline{\mathrm{bel}}(x_1) = &p(x_1 | U_1 = \mathrm{do \ nothing}, X_0 = \mathrm{open}) \mathrm{bel}(x_0 = \mathrm{open}) + \\ 
&p(x_1 | U_1 = \mathrm{do \ nothing}, X_0 = \mathrm{close}) \mathrm{bel}(x_0 = \mathrm{close})
\end{aligned}
$$

那么有
$$
\begin{aligned}
\overline{\mathrm{bel}}(X_1 = \mathrm{open}) &= 1 \times 0.5 + 0 \times 0.5 = 0.5 \\
\overline{\mathrm{bel}}(X_1 = \mathrm{close}) & = 0 \times 0.5 + 1 \times 0.5 = 0.5
\end{aligned}
$$

接着计算更新后的后验概率
$$
\mathrm{bel}(x_1) = \eta p(Z_1 = \mathrm{open} | x_1) \overline{\mathrm{bel}}(x_1)
$$

那么有
$$
\begin{aligned}
\mathrm{bel}(X_1 = \mathrm{open}) &= \eta 0.6 \times 0.5 = 0.3 \eta \\
\mathrm{bel}(X_1 = \mathrm{open}) &= \eta 0.2 \times 0.5 = 0.1 \eta
\end{aligned}
$$

归一化因子$\eta = (0.3 + 0.1)^-1 = 2.5$，所以有
$$
\begin{aligned}
\mathrm{bel}(X_1 = \mathrm{open}) &= 0.75 \\
\mathrm{bel}(X_1 = \mathrm{open}) &= 0.25
\end{aligned}
$$

对于$u_2 = \mathrm{push}$并且$Z_2 = \mathrm{open}$的情况下，此时可以计算得到
$$
\begin{aligned}
\overline{\mathrm{bel}}(X_2 = \mathrm{open}) &= 1 \times 0.75 + 0.8 \times 0.25 = 0.95 \\
\overline{\mathrm{bel}}(X_2 = \mathrm{close}) & = 0 \times 0.75 + 0.2 \times 0.25 = 0.05 \\
\mathrm{bel}(X_2 = \mathrm{open}) &= \eta 0.6 \times 0.95 \approx 0.983 \\
\mathrm{bel}(X_2 = \mathrm{open}) &= \eta 0.4 \times 0.05 \approx 0.017
\end{aligned}
$$

#### 2.4.3 贝叶斯滤波的数学推导
首先根据贝叶斯法则后验概率有
$$
\begin{aligned}
p(x_t | z_{1:t}, u_{1:t}) &= \frac{p(z_t| x_t,  z_{1:t-1},  u_{1:t}) p(x_t |  z_{1:t-1},  u_{1:t})}{p(z_t | z_{1:t-1},  u_{1:t})} \\
& = \eta p(z_t| x_t,  z_{1:t-1},  u_{1:t}) p(x_t |  z_{1:t-1},  u_{1:t}) \\
& = \eta p(z_t| x_t) p(x_t |  z_{1:t-1},  u_{1:t}) \\
& = \eta p(z_t| x_t) \overline{\mathrm{bel}}(x_{t})
\end{aligned}
$$

接着计算$\overline{\mathrm{bel}}(x_{t})$:
$$
\begin{aligned}
\overline{\mathrm{bel}}(x_{t}) &=  p(x_t |  z_{1:t-1},  u_{1:t}) \\
& = \int p(x_t | x_{t-1}, z_{1:t-1},  u_{1:t}) p(x_{t-1} | z_{1:t-1},  u_{1:t}) dx_{t-1} \\
& = \int p(x_t | x_{t-1}, u_{t})  p(x_{t-1} | z_{1:t-1},  u_{1:t-1}) dx_{t-1} \\
& = \int p(x_t | x_{t-1}, u_{t}) \mathrm{bel}(x_{t-1})
\end{aligned}
$$


### 2.8 习题
1.  
    解:
    $$
    p(x_1 = 0) = \frac{1}{34} \approx 0.0294 \\
    p(x_2 = 0) = \frac{1}{12} \approx 0.0833 \\
    \vdots\\
    p(x_N = 0) = \frac{1}{1 + 99 \frac{1}{3^N}}
    $$

2. 
    解:
    (a) $p = 0.2 * 0.4 * 0.2 = 0.016$
    (b) Day 0: 随机出一个天气; Day 1: 按照状态转移表的概率进行随机转移
    (c) ...
    (d) 设
        $$
        \begin{aligned}
        Y_t &= \begin{bmatrix}
            P(X_t = 1) \\
            p(X_t = 2) \\
            p(X_t = 3)
        \end{bmatrix} \\
        A &= \begin{bmatrix}
        P(X_{t+1} = 1 | X_t = 1) & P(X_{t+1} = 1 | X_t = 2) & P(X_{t+1} = 1 | X_t = 3) \\
        P(X_{t+1} = 2 | X_t = 1) & P(X_{t+1} = 2 | X_t = 2) & P(X_{t+1} = 2 | X_t = 3) \\
        P(X_{t+1} = 3 | X_t = 1) & P(X_{t+1} = 3 | X_t = 2) & P(X_{t+1} = 3 | X_t = 3)
         \end{bmatrix} \\
         & =  \begin{bmatrix}
         0.8 & 0.4 & 0.2\\
         0.2 & 0.4 & 0.6 \\
         0.0 & 0.2 & 0.2
           \end{bmatrix} 
         \end{aligned}
        $$
        有 $Y_{t+1} = A Y_t$，那么$Y_{n} = A^n Y_0$，将$A$进行相似分解为对角阵$A = P D P^{-1}$，其中$D$为
        $$
        D = \begin{bmatrix}
        1 & 0 & 0 \\
        0 & \frac{1 + \sqrt{2}}{5} & 0 \\
        0  & 0& \frac{1 - \sqrt{2}}{5}
         \end{bmatrix} 
        $$
        由于 $$
        \lim_{n \rightarrow \infty} D^n = \begin{bmatrix}
        1 & 0 & 0 \\
        0 & 0 & 0 \\
        0  & 0& 0
         \end{bmatrix} 
        $$
        所以最终
        $$
        \begin{aligned}
         \lim_{n \rightarrow \infty} Y^n &= P \begin{bmatrix}
        1 & 0 & 0 \\
        0 & 0 & 0 \\
        0  & 0& 0
         \end{bmatrix}  P^{-1} X_0 \\
         & = \begin{bmatrix}
        \frac{9}{14}  \\
        \frac{2}{7} \\
        \frac{1}{14}  
         \end{bmatrix}
         \end{aligned}
        $$
    (e) 
        $$
        \begin{aligned}
        H(x) = -(\frac{9}{14} \log_2\frac{9}{14} + \frac{2}{7} \log_2\frac{2}{7} + \frac{1}{14} \log_2\frac{1}{14})
        \end{aligned}
        $$
    (f) 根据贝叶斯法则，用稳态时候的概率，可以计算出概率为
        $$
         \begin{aligned}
        p(Y_{t-1} | Y_t) &= \frac{p(Y_t | Y_{t-1}) p(Y_{t-1})}{p(Y_t)} \\
         &=A \cdot \begin{bmatrix}
            1 & 2.25 & 9 \\
            0.444 & 1 & 4 \\
            0.111 & 0.25 & 1
         \end{bmatrix} \\
         & = 
         \begin{bmatrix}
         0.8 & 0.45 & 0 \\
         0.18 & 0.40 & 0.80 \\
         0.02 & 0.15 & 0.20
          \end{bmatrix} 
          \end{aligned}
        $$

    (g) 状态转移矩阵依赖季节的话会丧失马尔可夫性，需要将季节变量引入到状态变量中恢复马尔可夫性

3. 
   解:
   (a)

    |     | $z_t$ | $\overline{\mathrm{bel}}(x_t)$                 | ${\mathrm{bel}}(x_t)$                       | $\eta$ |
    | --- | ----- | ---------------------------------------------- | ------------------------------------------- | ------ |
    | 1   |       |                                                | $\begin{bmatrix}1&0&0\end{bmatrix}^T$       | 1      |
    | 2   | 多云  | $\begin{bmatrix}0.8&0.2&0\end{bmatrix}^T$      | $\begin{bmatrix}0.32&0.14&0\end{bmatrix}^T$ | 2.17   |
    | 3   | 多云  | $\begin{bmatrix}0.67&0.26&0.06\end{bmatrix}^T$ | $\begin{bmatrix}0.26&0.18&0\end{bmatrix}^T$ | 2.27   |
    用同样的方法向后计算，第五天时晴天的后验概率为0.4
   (b) 如果只依据以往的数据，计算后验概率，2-4最有可能的天气为晴、晴、雨，概率分别为0.89、0.87、1。利用所有数据计算后验概率，2-4最有可能的天气为晴、多云、雨，概率分别为0.8, 1.0, 1.0

4. 
   解:
   (a): $p(x)$ 为一个高斯分布，测量$p(z|x)$也为一个高斯分布
   (b): 后验也为一个高斯分布

## 第三章 高斯滤波
### 3.2 卡尔曼滤波
#### 3.2.1 线性高斯系统
状态转移概率以及测量概率均符合高斯分布
#### 3.2.2 卡尔曼滤波算法

```
Algorithm Kalman_filter(mu(t-1), Sigma(t-1), u(t), z(t)):
    overline_mu(t) = A(t) mu(t-1) + B(t) u(t)
    overline_Sigma(t) = A(t)  Sigma(t-1) A(t).T + R(t)

    K(t) = overline_sigma(t) C(t).T (C(t) overline_sigma(t) C(t).T + Q(t)).inverse()
    mu(t) = overline_mu(t) + K(t) (z(t) - C(t) overline_mu(t))
    Sigma(t) = (I -  K(t) C(t)) overline_Sigma(t)
```
#### 3.2.4 卡尔曼滤波的数学推导
首先根据贝叶斯滤波的预测步骤可知
$$
\begin{aligned}
    \overline{\mathrm{bel}}(x_t) &= \int p(x_t | x_{t-1}, u_t) {\mathrm{bel}}(x_t-1) dx_{t-1} \\
    &= \int N(A_t x_{t-1} + B_t u _t, R_t) N(\mu_{t-1}, \Sigma_{t-1}) dx_{t-1} \\
    &= \eta \int \exp \left\{ -\frac{1}{2}(x_t - A_t  x_{t-1} - B_t u_t)^T R_t^{-1}(x_t - A_t  x_{t-1} - B_t u_t)\right\} \\
    & \quad \exp \left\{-\frac{1}{2} (x_{t-1} - \mu_{t-1})^T \Sigma^{-1}_{t-1} (x_{t-1} - \mu_{t-1})\right\} dx_{t-1} \\
    & = \eta \int \exp \left\{ -L_t\right\} dx_{t-1}
\end{aligned}
$$

其中
$$
\begin{aligned}
L_t &= \frac{1}{2}(x_t - A_t  x_{t-1} - B_t u_t)^T R_t^{-1}(x_t - A_t  x_{t-1} - B_t u_t) + \\
    & \quad \frac{1}{2} (x_{t-1} - \mu_{t-1})^T \Sigma^{-1}_{t-1} (x_{t-1} - \mu_{t-1})
\end{aligned}
$$
该式子可以分解为仅包含$x_t$的部分以及剩余部分，经过推导可以得知 $\overline{\mathrm{bel}}(x_t)$ 也为一个高斯分布，其均值为 $\overline{\mu}_t = A_t \mu_{t-1} + B_t u_t$，方差为 $\overline{\Sigma}_t = A_t \Sigma_{t-1}^{-1} A_t^T + R_t$

更新步骤有
$$
\begin{aligned}
{\mathrm{bel}}(x_t) &= \eta p(z_t | x_t) \overline{\mathrm{bel}}(x_t) \\
& = N(C_tx_t, Q_t) N(\overline{\mu}_t, \overline{\Sigma}_t)
\end{aligned}
$$
该高斯的乘积也是一个高斯分布，且
$$
\mu_t = \overline{\mu}_t + K_t(z_t - C_t \overline{\mu}_t) \\
\Sigma_t = (I - K_t C_t) \overline{\Sigma}_t
$$

### 3.3 扩展卡尔曼滤波
#### 3.3.3 扩展卡尔曼滤波算法
```
Algorithm Extended_Kalman_filter(mu(t-1), Sigma(t-1), u(t), z(t)):
    overline_mu(t) = g(u(t), mu(t-1))
    overline_Sigma(t) = G(t)  Sigma(t-1) G(t).T + R(t)

    K(t) = overline_sigma(t) H(t).T (H(t) overline_sigma(t) H(t).T + Q(t)).inverse()
    mu(t) = overline_mu(t) + K(t) (z(t) - h(overline_mu(t)))
    Sigma(t) = (I -  K(t) H(t)) overline_Sigma(t)
```

### 3.4 无迹卡尔曼滤波
UKF采用一些采样点来计算高斯分布，避免求导

### 3.5 信息滤波
使用正则参数表达高斯分布，与KF是对偶关系