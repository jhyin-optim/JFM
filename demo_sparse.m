% This is a demo of the Jacobian-free method (JFM) for solving the nearest doubly stochastic matrix problem
% \begin{align}\label{1.1}
% \begin{array}{ll}
% \min\limits_{X\in \mathbb{R}^{n\times n}} & \frac{1}{2}\|X-\hat{X}\|^2 \\
% {\rm s.t.} & Xe=e, \\
% & X^{\top}e=e, \\
% & X\geq 0,
% \end{array}
% \end{align}
% using a Jacobian-free method based on a scaling memoryless DFP formula, described in the following paper
% Yin J, Li Y, Tang C. A Jacobian-free method for the nearest doubly stochastic matrix problem. Submitted to JOTA.
%
% -----------------------------------------------------------------------
% Copyright (2023): Jianghua Yin
% ----------------------------------------------------------------------
%
% The first version of this code by Jianghua Yin, Oct., 1, 2023

% If you use/modify this code, please cite the following papers appropriately:
% [1] Yin J, Jian J, Jiang X, et al. A family of inertial-relaxed DFPM-based algorithms for solving large-scale monotone nonlinear equations 
% with application to sparse signal restoration. Journal of Computational and Applied Mathematics, 2023, 419: 114674.
% [2] Jian J, Yin J, Tang C, et al. A family of inertial derivative-free projection methods for constrained nonlinear pseudo-monotone equations with applications. 
% Computational and Applied Mathematics, 2022, 41(7): 1-21.
% [3] Yin J, Jian J, Jiang X, et al. A hybrid three-term conjugate gradient projection method for constrained nonlinear monotone equations with applications.
% Numerical Algorithms, 2021, 88(1): 389-418.
% [4] Yin J, Jian J, Jiang X. A generalized hybrid CGPM-based algorithm for solving large-scale convex constrained equations with applications to image restoration. Journal of Computational and Applied Mathematics, 2021, 391: 113423.ons to image restoration. 
% Journal of Computational and Applied Mathematics, 2021, 391: 113423.
% [5.1] 尹江华, 简金宝, 江羡珍. 凸约束非光滑方程组基于自适应线搜索的谱梯度投影算法. 
% 计算数学, 2020, 42(4): 457-471.
% [5.2] Yin, J.H., Jian, J.B., Jiang, X.Z.: A spectral gradient projection algorithm for convex constrained
% nonsmooth equations based on an adaptive line search. Math. Numer. Sin. (Chinese) 2020, 42(4), 457-471.

%% reshape(A,[],1)  % stack the columns of A
% B = repmat(A,M,N) % creates a large matrix B consisting of an M-by-N 
% tiling of copies of A.If A is a matrix, the size of B is [size(A,1)*M, size(A,2)*N].
%% repmat(eye(n),1,n) 
%% kron(X,Y) is the Kronecker tensor product of X and Y.
clc
clear all
close all
%set parameters
% randn('seed',1); % 1 for the experiments in the paper
% rand('seed',1); % 1 for the experiments in the paper
rng(2024);
% rng('default')

% global A b Hatx

ITR_max = 40000;
% parameters for DFP
para1.Itr_max = ITR_max;
para1.gamma = 1;         % the initial guess
para1.sigma = 0.0001;    % the coefficient of line search 
para1.tau = 0.9;         % the compression ratio
para1.rho = 1;           % the relaxation factor 

% parameters for JFM
para2.Itr_max = ITR_max;
para2.gamma = 1;         % the initial guess  
para2.sigma = 0.5;       % the coefficient of line search in (0,1)
para2.tau = 0.5;         % the compression ratio in (0,1)
para2.rho = 1.95;        % the relaxation factor in (0,2)

fid_tex=fopen('mytext.txt','w'); 
% set the value of n
problem_set = [10:10:100 150:50:700]; %10:10:100 150:50:800

np = length(problem_set);
ns = 5;   % the number of the test algorithms
T = zeros(np,ns);
F = zeros(np,ns);
N = zeros(np,ns);
for index=1:np
    disp('generate the original data');
    n = problem_set(index);
    HatX = randn(n,n);
%     HatX = (HatX+HatX')/2; % make HatX symmetric
    Hatx = reshape(HatX,[],1);
    et = ones(1,n);
    I = speye(n);
    A1 = kron(I,et);
    A2 = repmat(I,1,n);
    A2(n,:) = []; % delete the last row in A2
    A = [A1;A2]; % generate a sparse matrix
    b = ones(2*n-1,1);
    y0 = zeros(2*n-1,1); % set the initial point 0.1*ones(2*n-1,1);
    % define function handle
%     Ay = @(y) A*y;
%     AT = @(y) (y'*A)';
    Fun = @(y) A*(max(Hatx+(y'*A)',0))-b;

    disp('Starting DFP') % Rao, J., Huang, N.: A derivative-free scaling memoryless DFP method
    % for solving large scale nonlinear monotone equations. Journal of Global
    % Optimization (2022). DOI 10.1007/s10898-022-01215-2
    [T1,NFF1,NI1,G1] = JFM(y0,Fun,'DFP',4,para1);
    
    disp('Starting SMBFGS') % Ullah, N., Sabi’u, J., Shah, A.: A derivative-free scaling memoryless 
    % Broyden-Fletcher-Goldfarb-Shanno method for solving a system of monotone nonlinear equations. 
    % Numerical Linear Algebra with Applications 28(5), e2374 (2021)
    [T2,NFF2,NI2,G2] = JFM(y0,Fun,'SMBFGS',4,para1);

    disp('Starting PDFP') % UR Rehman, M., Sabi’u, J., Sohaib, M., Shah, A.: A projection-based derivative free
    % DFP approach for solving system of nonlinear convex constrained monotone equations with image restoration          
    % applications. Journal of Applied Mathematics and Computing 69(5), 3645–3673 (2023)
    [T3,NFF3,NI3,G3] = JFM(y0,Fun,'PDFP',4,para1); 
    
    disp('Starting JFM1') % the proposed method with model=4
    [T4,NFF4,NI4,G4] = JFM(y0,Fun,'JFM1',4,para2);
    
    disp('Starting JFM2') % the proposed method with model=2
    [T5,NFF5,NI5,G5] = JFM(y0,Fun,'JFM1',2,para2);
    
    fprintf(fid_tex,'%d & %.0f/%.0f/%.3f/%.2e & %.0f/%.0f/%.3f/%.2e & %.0f/%.0f/%.3f/%.2e\n & %.0f/%.0f/%.3f/%.2e & %.0f/%.0f/%.3f/%.2e\\\\ \n', ... 
                n,NI1,NFF1,T1,G1,NI2,NFF2,T2,G2,NI3,NFF3,T3,G3,NI4,NFF4,T4,G4,NI5,NFF5,T5,G5); % & %.0f/%.0f/%.3f/%.2e & %.0f/%.0f/%.3f/%.2e\n
    T(index,:) = [T1,T2,T3,T4,T5]; 
    F(index,:) = [NFF1,NFF2,NFF3,NFF4,NFF5];
    N(index,:) = [NI1,NI2,NI3,NI4,NI5]; 
end
%% 关闭文件
fclose(fid_tex);

%% 画图
clf;   %clf删除当前图形窗口中、
       %%句柄未被隐藏(即它们的HandleVisibility属性为on)的图形对象
figure(1);
%subplot(2,2,1);
perf(T,'logplot');
%title('时间性能');
%set(gca,'ylim',[0.3,1]);
xlabel('\tau','Interpreter','tex');
ylabel('\rho(\tau)','Interpreter','tex');
legend('DFP','SMBFGS','PDFP','JFM1','JFM2');
% %subplot(2,2,2);
figure(2);
perf(F,'logplot');
%title('目标函数计算性能');
% set(gca,'ylim',[0.1,1]);
xlabel('\tau','Interpreter','tex');                     %% 给x轴加注
ylabel('\rho(\tau)','Interpreter','tex');               %% 给y轴加注
legend('DFP','SMBFGS','PDFP','JFM1','JFM2');
%subplot(2,2,3);
figure(3);
perf(N,'logplot');
%title('迭代次数性能');
%set(gca,'ylim',[0.5,1]);
xlabel('\tau','Interpreter','tex');
ylabel('\rho(\tau)','Interpreter','tex');
legend('DFP','SMBFGS','PDFP','JFM1','JFM2');
% %hold on
% %text
% %axes
% %set(gca,'xtick',[],'ytick',[]) 
% %figure(2);
% %perf(T,'logplot');% 此“logplot”任何一个确定的常数都可以，只要保证有两个输入变量