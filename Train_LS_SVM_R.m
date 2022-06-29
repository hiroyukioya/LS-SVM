   function [alpha, beta, W, NP, PRESS, resid, rsq, rsqp]=Train_LS_SVM_R(X,Y,rambda, kernel, sigma)

%% ///*****   Train Least-Squares SVM   [LS-SVM]   *****///
%     
%  [USAGE]
%   [alpha, beta, W, NP, PRESS, LOOE]=...
%                     Train_LS_SVM(X,Y,rambda, kernel, sigma)
%
%  [INPUTS]
%    X= input data [n x p]  (n=observations, p=dimension)
%    y= respone vector [n x1]
%    rambda:  Regularization parameter (smaller value means stronger
%    regularization)
%    kernel:     Enter strings: 'Linear', 'RBF' or 'Polynomial'
%    sigma= parameter for non-linear svm (sigma or degree for the
%    polynomial)
%
%  [OUTPUTS]
%    PRESS:  LOO prediction residuals sum-of-squares
%    LOOE:  Leave-one-out error rate
%    alpha:   Lagrange multipliers (positive real)
%    beta:    bias parameter
%    W:       Primal space Weights (Linear kernel case)
%    resid:   LOO cv prediction residuals
%    NP:      Decision score
%    
%
%    Refs:  Suykens:  Neural process Letters, 1999
%             Gestel:      Machine Learn,   2004
%             Adankon;   Machine Learn, 2008
%             Senjian;     Pattern Recog, 2007   
%              Cavin;       Neural Network, 2004 
%              An;            Pattern recog. 2007 
%              Ojeda;       Neural Network, 2008
%
%    ///**********************************************************///        

%     Created by  H. Oya (2009)


%%  PCA dimentional reduction (optional) --------
pca=0;

if pca==1
    %  Take "all" non-zero eigenvectors,
    %  so that complete recon. is ensured.
    %  v will be used for reconstruction of gamma
    [v,nx,latent]=princomp(X,'econ');  
end

%% 
Y=Y(:);
[a,b]=size(X);
[c,d]=size(Y);
if a~=c;
    error('Two input matrices must have same number of rows.....')
end

o=strncmpi(kernel,'rbf',3);
o3=strncmpi(kernel,'gauss',4);
o1=strncmpi(kernel,'polynomial',4);
o2=strncmpi(kernel,'linear',3);

if o2==1
    mode=1;
%     disp(sprintf('   Linear kernel'))
    % Linear ls-svm
     K=X*X';
elseif o==1 | o3==1
    mode=2;
%      disp(sprintf('   RBF kernel'))
    % RBF ls-svm
    [tK,D,K]=KernelMatrix(X,sigma,'gaus');
elseif o1==1 
    mode=3;
%      disp(sprintf('   Polynomial kernel'))
    % Polonomial ls-svm
    [tK,D,K]=KernelMatrix(X,sigma,'poly');
end

% Construct H matrix ...
H=K+(eye(a)./rambda);
IN=ones(a,1);

% //**** Solve systems of equations... ****//
U=[H IN; IN' 0];
I=[Y;0];
tem=U\I;
beta=tem(end);        % beta= bias term
alpha=tem(1:end-1);% alpha= Lagrange multipliers

%  <This may be useful for data with large observation>
%                           < OPTION>
%     % Solve for eta
%     [eta]=sv_conj_grad_descent(H,Y);
%     % Solve for mu
%     [mu]=sv_conj_grad_descent(H,ones(a,1));
%     % Compute s
%     s=Y'*eta;
%     % Final Solutions .....
%     beta=eta'*ones(a,1)/s;
%     alpha=mu-eta*beta;

% For linear kernel case, we can go to primal space ....
if strncmpi(kernel,'Linear',4)==1
    % Primal Space Weights (Linear kernel case)
     W=alpha'*X;
else 
    W=0;
end

%  /****  Exact LOO-CV errors evaluation ****/
% ls-svm block structure...
% NH=[H  ones(a,1);ones(1,a) 0]; 
% iNH=inv(NH);
iNH=inv(H);

% Compute prediction residuals.....
mu=iNH*ones(a,1);
s=-1*ones(1,a)*iNH*ones(a,1);
iA=diag(iNH)+mu.^2./s;
resid=alpha./iA(1:a);

% PRESS statistic ...
PRESS=sum(resid.^2);

% Fitted values
NP=alpha'*K+beta;

% Predicted labels
yn=sign(NP);

% R squares (fitted)
tem= sum((Y-NP').^2);
rsq=1-tem/sum(Y.^2);
% predicted R-square
rsqp=1-PRESS/sum(Y.^2);
% PRESS=PRESS/length(resid);

% LOO error rate 
% f=find((yn.*Y)==1);
% Acc=length(f)/a;
% LOOE=1-Acc;

%/***********************************/%
function [K,D,CK]=KernelMatrix(data,sigma,kernel)
%%  Kernel matrix 
% Compute kernel matrix using Gaussian kernel with sigma value
 
if sigma<=0;
    error('       sigma must be positive     ');
end

[a,b]=size(data);
K=zeros(a,a);

o=strncmpi(kernel,'gaussian',4);
o1=strncmpi(kernel,'polynomial',4);

if (o==1) & (o1==0)
    for n=1:a
            for m=n:a
                  [K(n,m)]=gaussian_kernel_function(data,n,m,sigma);
            end
    end
elseif (o1==1) & (o==0)
    for n=1:a
            for m=n:a
               [K(n,m)]=polynomial_kernel_function(data,n,m,sigma);
            end
    end
end
    

% Kernel matrix
for n=1:a
      m=n:a;
      K(m,n)=K(n,m)';
end

% Normalize
D=diag(1./sqrt(diag(K)));
NK=D*K*D;

%Centering
d=sum(K)/a;
e=sum(d)/a;
J=ones(a,1)*d;
CK=K-J-J'+e*ones(a,a);

function [a]=gaussian_kernel_function(data,ind1,ind2,sigma)

k1=dot(data(ind1,:),data(ind1,:));
k2=dot(data(ind2,:),data(ind2,:));
kb=dot(data(ind1,:),data(ind2,:));

% Gaussian Kernel Function
a=exp(-(k1+k2-2*kb)/(2*sigma^2));

function [a]=polynomial_kernel_function(data,ind1,ind2,sigma)

kb=dot(data(ind1,:),data(ind2,:));
% polynomial kernel od degree sigma
a=(kb).^sigma;

%%     solver for ls-svm
function [x,n]=sv_conj_grad_descent(A,B)
%%               sv_conj_grad_descent.m
%               <<<< Solve Linear System
%   with Conjugate Gradient Descent algorithm >>>>
% 
%    Solve Ax=B
%    A has to be square positive definite matrix
%    
%

%    Created H. Oya (2009)
%%

[a,b]=size(A);
c=length(B);

% check matrices -------------
if a~=b
    error(' Matrix must be square .....')
end
if a~=c
    error(' Response vector must have the same number of colums ....');
end
if min(real(eig(A)))<=0
    error( ' Matrix must be positive definite ...')
end
    
% Initialization----------
x=zeros(c,1);
r=B-A*x;
rho_old=r'*r;
rho_new=rho_old;
maxiter=251;
n=0;
tol=10^-10.0;

% Main loop------------
while n<=maxiter
    n=n+1;
    if n==1
        p=r;
    else
        % improvement
        beta=rho_new/rho_old;
        % direction
        p=r+beta*p;
    end
    q=A*p;
    % step length
    alpha=rho_new/(p'*q);
    % approx. solution
    x=x+alpha*p;
    rho_old=rho_new;
    % update rho
    r=r-alpha*q;
    rho_new=r'*r;
    
    if norm(r)<tol
        break
    end
    if n==maxiter
       error('Iteration was not converged...')
    end
       
end
    
%%  Centering and Normalizing 
function [newX, sx, mx] = centernormalize(X)
% Centering and Scaling the matrix X

[L,N]=size(X);
% Scaling ...
sx=std(X,[],1);
newX=X./sx(ones(L,1),:);

% Centering ...
 mx=mean(newX);
 mmx=repmat(mx,[L 1]);
 newX=newX-mmx;

