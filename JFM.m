% Matlab Model by Jianghua Yin (Oct.,2023, Nanning)
% Copyright (C) 2023 Jian Group
% All Rights Reserved
%
%% a Jacobian-free method (JFM) for solving nonsmooth 
%% monotone equations of the form
% \begin{align*}
%  F(y)=A(\hat{x}+A^{\top}y)_{+}-b=0, 
% \end{align*}
% where $A\in \mathbb{R}^{(2n-1)\times n^{2}},\ \hat{x}\in
% \mathbb{R}^{n^{2}}$ and $b=ones(2*n-1,1)$.

function [Tcpu,NF,Itr,NormF] = JFM(y0,Fun,method,model,para) 
 
format long

% start the clock
tic; 

%% the stopping criterion
epsilon = 1e-5;     % 1e-5
epsilon1 = 1e-6;    % 1e-6

%% the line search parameters and relaxation factor
k_max = para.Itr_max;   % the maximum number of iterations
gamma = para.gamma;     % the initial guess
sigma = para.sigma;     % the coefficient of line search 
tau = para.tau;         % the compression ratio
rho = para.rho;         % the relaxation factor 

fprintf('%s & LSmodel=%d & gamma=%.1f & sigma=%.4f & tau=%.1f & rho=%.2f\n', ... 
    method,model,gamma,sigma,tau,rho);


%% compute the search direction
Fk0 =Fun(y0);           % evaluate the function value at y0
NF = 1;                 % the total number of function evaluations
dk = -Fk0;
NormFk0 = norm(Fk0);
L1 = 0;
     
for k=1:k_max
    
    if k==1 && NormFk0<=epsilon
        L1 = 1;
        NormF = NormFk0;    % the final norm of equations
        break; 
    end
    
    %%% Start Armijo-type line search  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % model=1 means -F(zk)'*dk ¡Ý sigma*tk*norm(dk)^2
    % model=2 means -F(zk)'*dk ¡Ý sigma*tk*norm(F(zk))*norm(dk)^2
    % model=3 means -F(zk)'*dk ¡Ý sigma*tk*norm(F(zk))/(1+norm(F(zk)))*norm(dk)^2
    % model=4 means -F(zk)'*dk ¡Ý sigma*tk*norm(F(zk))*norm(dk)
    % model=5 means -F(zk)'*dk ¡Ý sigma*tk*min(nu,norm(Fzk))*norm(dk)^2
    if model==1
        t = gamma;
        z_new = y0+t*dk;
        Fz_new = Fun(z_new);
        NF = NF+1;
        Normdk2 = norm(dk)^2;
        % check the Armijo-type line search condition
        while -Fz_new'*dk < sigma*t*Normdk2 && t>10^-10  
            % the Armijo-type line search condition violated
            t = t*tau;
            z_new = y0+t*dk;
            Fz_new = Fun(z_new);
            NF = NF+1;
        end %%% End Armijo-type line search %%%
    elseif model==2
        t = gamma;
        z_new = y0+t*dk;
        Fz_new = Fun(z_new);
        NF = NF+1;
        NormFzk = norm(Fz_new);
        Normdk2 = norm(dk)^2;
        % check the Armijo-type line search condition
        while -Fz_new'*dk < sigma*t*NormFzk*Normdk2 && t>10^-6
            % the Armijo-type line search condition violated
            t = t*tau;
            z_new = y0+t*dk;
            Fz_new = Fun(z_new);
            NF = NF+1;
            NormFzk = norm(Fz_new);
        end %%% End Armijo-type line search %%%
    elseif model==3
        t = gamma;
        z_new = y0+t*dk;
        Fz_new = Fun(z_new);
        NF = NF+1;
        NormFzk = norm(Fz_new);
        Normdk2 = norm(dk)^2;
        % check the Armijo-type line search condition
        while -Fz_new'*dk < sigma*t*NormFzk/(1+NormFzk)*Normdk2 && t>10^-10  
            % the Armijo-type line search condition violated
            t = t*tau;
            z_new = y0+t*dk;
            Fz_new = Fun(z_new);
            NF = NF+1;
            NormFzk = norm(Fz_new);
        end %%% End Armijo-type line search %%%
    elseif model==4
        t = gamma;
        z_new = y0+t*dk;
        Fz_new = Fun(z_new);
        NF = NF+1;
        NormFzk = norm(Fz_new);
        Normdk = norm(dk);
        % check the Armijo-type line search condition
        while -Fz_new'*dk < sigma*t*NormFzk*Normdk && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z_new = y0+t*dk;
            Fz_new = Fun(z_new);
            NF = NF+1;
            NormFzk = norm(Fz_new);
        end %%% End Armijo-type line search %%%
    else
        nu = 0.8;
        t = gamma;
        z_new = y0+t*dk;
        Fz_new = Fun(z_new);
        NF = NF+1;
        NormFzk = norm(Fz_new);
        Normdk2 = norm(dk)^2;
        % check the Armijo-type line search condition
        while -Fz_new'*dk < sigma*t*min(nu,NormFzk)*Normdk2 && t>10^-6  
            % the Armijo-type line search condition violated
            t = t*tau;
            z_new = y0+t*dk;
            Fz_new = Fun(z_new);
            NF = NF+1;
            NormFzk = norm(Fz_new);
        end %%% End Armijo-type line search %%%
    end 
    zk = z_new;
    Fzk = Fz_new;
    if model==1
        NormFzk = norm(Fzk);
    end
    if NormFzk<=epsilon
        L1 = 1;
        NormF = NormFzk; % the final norm of equations
        break;
    end
    xik = Fzk'*(y0-zk)/NormFzk^2;
    % compute the next iteration 
    y1 = y0-rho*xik*Fzk;
    Fk1 = Fun(y1);
    NF = NF+1;
    NormFk1 = norm(Fk1);
    if NormFk1<=epsilon
        L1 = 1;
        NormF = NormFk1;
        break;
    end
    % update the search direction
    switch method
        case 'DFP'
            sk = y1-y0;
            wk = Fk1-Fk0+0.06*sk;
            betak = 0.88*wk'*Fk1/norm(wk)^2;
            thetak = sk'*Fk1/norm(sk)^2;
            dk = -1.08*Fk1+betak*wk-thetak*sk; 
        case 'JFM1'
            sk = zk-y0;
            wk = Fk1-Fk0+0.06*sk; 
            if model==4
                Normdk2 = Normdk^2;
            end
            denom = max(0.6*norm(wk)^2,NormFk0^2)*Normdk2; 
            betak = (Fk1'*wk)*(dk'*wk)/denom-dk'*Fk1/Normdk2;
            dk = -Fk1+betak*dk; 
        case'SMBFGS'
            sk = y1-y0;
            wk = Fk1-Fk0+0.06*sk;
            wkTsk = wk'*sk;
            betak = wk'*Fk1/wkTsk-2*norm(wk)^2*sk'*Fk1/wkTsk^2;
            thetak = sk'*Fk1/wkTsk;
            dk = -Fk1+betak*sk+thetak*wk;
        case'PDFP'
            sk = zk-y0;
            wk = Fk1-Fk0+0.06*sk; 
            nsk = norm(sk);
            nwk = norm(wk);
            nsknwk = nsk/nwk;
            betak  = nsknwk*wk'*Fk1/nwk^2;
            thetak = sk'*Fk1/(wk'*sk);
            dk = -nsknwk*Fk1+betak*wk-thetak*sk;
    end
    if norm(dk)<epsilon1
        L1 = 1;
        NormF = norm(Fk1);
        break;
    end
%     tused = toc;
%     if tused>1800
%         break;
%     end

    % update the iteration
    y0 = y1;
    Fk0 = Fk1;
    NormFk0 = NormFk1;
end
if L1==1
    Itr = k;
    Tcpu = toc;
else
    NF = NaN;
    Itr = NaN;
    Tcpu = NaN;
    NormF = NaN;
end
fprintf('%s & NI=%d & NFF=%.d & Tcpu=%.3f & NG=%.2e\n', ... 
    method,Itr,NF,Tcpu,NormF);
