# JFM
This is a demo of a Jacobian-free method (JFM) for solving the nearest doubly stochastic matrix problem

\begin{align}\label{1.1}

\begin{array}{ll}

\min\limits_{X\in \mathbb{R}^{n\times n}} & \frac{1}{2}|X-\hat{X}|^2 \\

{\rm s.t.} & Xe=e,\\

& X^{\top}e=e, \\

& X\geq 0,\\

\end{array}

\end{align}

using a Jacobian-free method based on a scaling memoryless DFP formula, described in the following paper

Yin J, Li Y, Tang C. A Jacobian-free method for the nearest doubly stochastic matrix problem. Submitted to JOTA.

% -----------------------------------------------------------------------

% Copyright (2023): Jianghua Yin

% ----------------------------------------------------------------------

The first version of this code by Jianghua Yin, Oct., 1, 2023

If you use/modify this code, please cite the following papers appropriately:

[1] Yin J, Jian J, Jiang X, et al. A family of inertial-relaxed DFPM-based algorithms for solving large-scale monotone nonlinear equations with application to sparse signal restoration. Journal of Computational and Applied Mathematics, 2023, 419: 114674.

[2] Jian J, Yin J, Tang C, et al. A family of inertial derivative-free projection methods for constrained nonlinear pseudo-monotone equations with applications. Computational and Applied Mathematics, 2022, 41(7): 1-21.

[3] Yin J, Jian J, Jiang X, et al. A hybrid three-term conjugate gradient projection method for constrained nonlinear monotone equations with applications. Numerical Algorithms, 2021, 88(1): 389-418.

[4] Yin J, Jian J, Jiang X. A generalized hybrid CGPM-based algorithm for solving large-scale convex constrained equations with applications to image restoration. Journal of Computational and Applied Mathematics, 2021, 391: 113423.ons to image restoration. Journal of Computational and Applied Mathematics, 2021, 391: 113423.

[5.1] 尹江华, 简金宝, 江羡珍. 凸约束非光滑方程组基于自适应线搜索的谱梯度投影算法. 计算数学, 2020, 42(4): 457-471.

[5.2] Yin, J.H., Jian, J.B., Jiang, X.Z.: A spectral gradient projection algorithm for convex constrained nonsmooth equations based on an adaptive line search. Math. Numer. Sin. (Chinese) 2020, 42(4), 457-471.

Questions/comments/suggestions about the codes are welcome.

Jianghua Yin, jianghuayin1017@126.com, jhyin@gxmzu.edu.cn

Chunming Tang, cmtang@gxu.edu.cn
