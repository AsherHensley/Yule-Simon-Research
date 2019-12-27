function [Chain,History,BestFit] = yulesimon(rt,varargin)
%YULESIMON(...) MCMC Sampler for Latent Yule-Simon Processes
%   [CHAIN,HISTORY,BESTFIT] = YULESIMON(RT) Runs Gibbs sampling algorithm
%   based on log return series RT, where
%
%       RT(t) = log(Price(t)) - log(Price(t-1))
%
%   to sample the posterior of the latent Yule-Simon generative model:
%
%             alpha|a,b ~ Gamma(a,b)
%         lambda(j)|c,d ~ Gamma(c,d)
%                 d|u,v ~ Gamma(u,v)
%   s(t)|x(1:t-1),alpha ~ Bernoulli(alpha/(nxt+alpha))
%      w(t)|x(t),lambda ~ Gaussian(0,1/lambda(x(t)))
%                v(t)|q ~ Gaussian(0,q^2)
%                  x(t) = x(t-1) + s(t)
%                 mu(t) = mu(t-1) + v(t)
%                  r(t) = mu(t) + w(t)
%
%   where nxt = sum(x(t)==x(1:t-1)) and log returns are given by r(t). The
%   final state of the Markov chain is returned in the CHAIN struct,
%   the samples from each iteration are returned in the HISTORY struct, 
%   and the best fit MCMC sample is returned in the BESTFIT struct.
%
%   Required: Statistics/Machine Learning Toolbox
%
%   [CHAIN,HISTORY,BESTFIT] = YULESIMON(...,'param',value) runs 
%   Gibbs sampler using optional parameter/input value pairs:
%
%       'a' = alpha shape hyperparameter [default = 2]
%       'b' = alpha rate hyperparameter [default = 2]
%       'c' = lambda shape hyperparameter [default = 5]
%       'd' = initial state of lambda rate hyperparameter [default = 5]
%       'u' = d shape hyperparameter [default = 1]
%       'v' = d rate hyperparameter [default = 1]
%       'alpha' = initial state of Yule-Simon parameter [default = 10]
%       'q' = initial state of plant noise sigma [default = 5e-3]
%       'mu0' = initial state of log return mean process [default = 0] 
%       'V0' = initial variance of mean process [default = 1e-1] 
%       'Chain' = initial state of Markov Chain. The Chain struct from a 
%           previous run can be used as the starting point for the current
%           run. If no Chain is passed in, the sampler will initialize 
%           Chain structure automatically. Note, that when this option is
%           used, all other optional sampler parameters (above) will be
%           ignored.
%       'sample_alpha' = flag to enable 'alpha' sampler [default = true]
%       'sample_cd' = flag to enable 'c,d' sampler [default = true]
%       'estimate_q' = flag to enable 'q' estimator [default = false]
%       'waitbar' = flag to enable sampler waitbar [default = true]
%       'show_warnings' = print sampler warnings [default = false]
%       'mh_sigma' = Metropolis-Hastings Gaussian proposal sigma for 
%           sampling lambda hyperparamaeters c,d [default = [0.25,0.5]]
%       'pause' = algorithm pause time between iterations to avoid 
%           overheating in seconds [default = 0]
%       'ngibbs' = number of gibbs iterations [default = 2000]
%
%   Revisions:
%   1.0     13-Aug-2018     Hensley     Initial release
%
%   MIT License
% 
%   Copyright (c) 2018 Asher A. Hensley Research
%   $Revision: 1.0 $  $Date: 2018/08/13 12:00:01 $
%   
%   Permission is hereby granted, free of charge, to any person obtaining a 
%   copy of this software and associated documentation files (the 
%   "Software"), to deal in the Software without restriction, including 
%   without limitation the rights to use, copy, modify, merge, publish, 
%   distribute, sublicense, and/or sell copies of the Software, and to 
%   permit persons to whom the Software is furnished to do so, subject to 
%   the following conditions:
% 
%   The above copyright notice and this permission notice shall be included 
%   in all copies or substantial portions of the Software.
% 
%   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
%   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
%   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
%   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
%   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
%   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
%   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

%Force Input to Column & Convert to "Percent"
rt = rt(:) * 100;

%Default Parameters
Config.a = 2;
Config.b = 2;
Config.c = 5;             
Config.d = 5;             
Config.u = 1;
Config.v = 1;
Config.alpha = 10;
Config.q = 5e-3;          %This is a std dev, not a variance
Config.mu0 = 0;
Config.V0 = 1e-1;
Flag.sample_alpha = true;
Flag.sample_cd = true;
Flag.estimate_q = false;
Flag.waitbar = true;
Flag.show_warnings = false;
mh_sigma = [0.25,0.5];
pause_time = 0;
ngibbs = 2000;

%Optional Inputs
nvararg = length(varargin);
for kk = 1:2:nvararg
    switch varargin{kk}
        case 'a'
            Config.a = varargin{kk+1};
        case 'b'
            Config.b = varargin{kk+1};
        case 'c'
            Config.c = varargin{kk+1};
        case 'd'
            Config.d = varargin{kk+1};
        case 'u'
            Config.u = varargin{kk+1};
        case 'v'
            Config.v = varargin{kk+1};
        case 'alpha'
            Config.alpha = varargin{kk+1};
        case 'q'
            Config.q = varargin{kk+1};
        case 'mu0'
            Config.mu0 = varargin{kk+1};
        case 'V0'
            Config.V0 = varargin{kk+1};
        case 'Chain'
            Chain = varargin{kk+1};
        case 'sample_alpha'
            Flag.sample_alpha = varargin{kk+1};
        case 'sample_cd'
            Flag.sample_cd = varargin{kk+1};
        case 'estimate_q'
            Flag.estimate_q = varargin{kk+1};
        case 'waitbar'
            Flag.waitbar = varargin{kk+1};
        case 'show_warnings'
            Flag.show_warnings = varargin{kk+1};
        case 'mh_sigma'
            mh_sigma = varargin{kk+1};
        case 'pause'
            pause_time = max(0,varargin{kk+1});
        case 'ngibbs'
            ngibbs = varargin{kk+1};
        otherwise
            error('Unknown Input Parameter')  
    end
end

%Init Markov Chain
if ~exist('Chain','var')
    Chain = gibbsInit(rt,Config,Flag);
end

%Init History Struct
History = historyInit(rt,ngibbs,Chain);

%Run
if Flag.waitbar
    hw = waitbar(0,'Running MCMC Update 0 of 0');
    pos = get(hw,'position');
    set(hw,'position',[pos(1:3),pos(4)*1.25])
end
ns = num2str(ngibbs);

for it = 2:ngibbs+1
    
    %Mean Removal
    wt = rt - Chain.mu;

    %Sample Yule-Simon Partitions
    Chain = sampleYuleSimonPartitions(wt,Chain);
    
    %Sample Lambda Posterior
    Chain = sampleLambdaPosteriorBatch(wt,Chain);
    
    %Sample Alpha Posterior
    if Flag.sample_alpha
        Chain = sampleAlphaPosterior(Chain);
    end
    
    %Sample Lambda Hyperparameters (c,d)
    if Flag.sample_cd
        icond = [Chain.c,Chain.d];
        reject = 1;
        timeOut = 0;
        while reject
            timeOut = timeOut+1;
            [Chain.c,Chain.d,reject] = metropolisHastings(Chain.lambda,1,'icond',icond,'sigma',mh_sigma);
            if timeOut==20 && Flag.show_warnings==true
                disp(['WARNING: Metropolis-Hastings Sampler Steps > 20 (it=',num2str(it),')'])
            end
        end
    end
    
    %Update State-Space Model
    Chain = kalman(rt,Chain,Flag);

    %Update History
    History.alpha(it) = Chain.alpha;
    History.c(it) = Chain.c;
    History.d(it) = Chain.d;
    History.q(it) = Chain.q;
    History.x(:,it) = Chain.x;
    History.lambda(:,it) = Chain.lambda(Chain.x);
    History.lambdac{it} = Chain.lambda;
    History.mu(:,it) = Chain.mu;
    History.muvar(:,it) = Chain.muvar;
    History.LP(it) = logPosterior(rt,Chain);
    History.mu0(it) = Chain.mu0;
    History.V0(it) = Chain.V0;
    Z = (rt-Chain.mu).*sqrt(Chain.lambda(Chain.x));
    [~,pval1] = chi2gof(Z);
    [~,pval2] = kstest(Z);
    History.pval(it) = (pval1+pval2)/2;
    
    %Update Waitbar
    if Flag.waitbar
        waitbar(it/ngibbs,hw,...
            {['Running MCMC Update ',num2str(it),' of ' ns];...
            ['Log-Posterior = ',num2str(round(History.LP(it)))];...
            ['Partitions: ',num2str(max(Chain.x)),' | p-value: ', sprintf('%0.3f',History.pval(it))]})
    end
    
    %Pause
    if pause_time>0 && mod(it,1000)==0
        disp(['Pausing for ',num2str(pause_time),' seconds to cool down'])
        pause(pause_time)
    end
    
end
if Flag.waitbar
    delete(hw)
end

%Best Fit MCMC Sample
[~,BestFit.pmask] = max(History.pval);
BestFit.mu = History.mu(:,BestFit.pmask);
BestFit.sigma = 1./sqrt(History.lambda(:,BestFit.pmask));
BestFit.rt_whitened = (rt(:)-BestFit.mu(:))./BestFit.sigma(:);
[~,BestFit.pval.kstest] = kstest(BestFit.rt_whitened);
[~,BestFit.pval.chi2test] = chi2gof(BestFit.rt_whitened);

%**************************************************************************
function Chain = sampleYuleSimonPartitions(wt,Chain)

%Sample First Point
Chain = sampleCurrentPoint(wt,1,Chain);

%Sample Internal Points
N = length(Chain.x);
for t = 2:N-1
    boundary = Chain.x(t)~=Chain.x(t-1) || Chain.x(t)~=Chain.x(t+1);
    if boundary
        Chain = sampleCurrentPoint(wt,t,Chain);
    end
end

%Sample Last Point
Chain = sampleCurrentPoint(wt,N,Chain);

%**************************************************************************
function Chain = sampleCurrentPoint(wt,t,Chain)

%Get Transition Weights
[w,type] = getTransitionWeights(wt,t,Chain);

%All Zeros Chk
if all(w==0)
    w = ones(1,length(w));
    disp('WARNING: All Transition Weights Are Zero -- Setting to Uniform')
end

%Return If Interior Point
if strcmp(type,'interiorPoint')
    return
end

%Update Markov Chain
Chain = updateMarkovChain(w,type,wt,t,Chain);

%**************************************************************************
function [w,type] = getTransitionWeights(wt,t,Chain)

%Parameters
alpha = Chain.alpha;
lambda = Chain.lambda;
c = Chain.c;
d = Chain.d;
x = Chain.x;

%In-Line Functions
F = @(z,idx) normpdf(z,0,1/sqrt(lambda(idx))); 
H = @(z) alpha/(1+alpha) * student(z,0,c/d,2*c);

%Counts
n = diff(find([1;diff(x);1]));
N = size(wt,1);
L = length(n);
j = x(t);

%Init
w = nan;
type = 'interiorPoint';

%Determine Case
if t==1 %First Sample
    
    if x(t+1)==1
        w(1) = (n(1)-1)/(n(1)+alpha)*F(wt(1),1);
        w(2) = H(wt(1));
        type = 't1_noBoundary';
        
    else
        w(1) = n(2)/(n(2)+alpha+1)*F(wt(1),2);
        w(2) = H(wt(1));
        type = 't1_rightBoundary';
        
    end
    
elseif t==N %Last Sample
    
    if x(t-1)==L
        w(1) = (n(L)-1)/(n(L)+alpha)*F(wt(N),L);
        w(2) = H(wt(N));
        type = 'tN_noBoundary';
     
    else
        w(1) = n(L-1)/(n(L-1)+alpha+1)*F(wt(N),L-1);
        w(2) = H(wt(N));
        type = 'tN_leftBoundary';

    end
    
elseif x(t-1)~=j && x(t+1)==j %Left Boundary
    
    w(1) = (n(j)-1)/(n(j)+alpha)*F(wt(t),j);
    w(2) = n(j-1)/(n(j-1)+alpha+1)*F(wt(t),j-1);
    w(3) = H(wt(t));
    type = 'leftBoundary';
    
elseif x(t-1)==j && x(t+1)~=j %Right Boundary
    
    w(1) = (n(j)-1)/(n(j)+alpha)*F(wt(t),j);
    w(2) = n(j+1)/(n(j+1)+alpha+1)*F(wt(t),j+1);
    w(3) = H(wt(t));
    type = 'rightBoundary';
    
elseif x(t-1)~=j && x(t+1)~=j %Double Boundary
    
    w(1) = n(j-1)/(n(j-1)+alpha+1)*F(wt(t),j-1);
    w(2) = n(j+1)/(n(j+1)+alpha+1)*F(wt(t),j+1);
    w(3) = H(wt(t));
    type = 'doubleBoundary';

end

%**************************************************************************
function Chain = updateMarkovChain(w,type,wt,t,Chain)

%Setup
j = Chain.x(t);
u = discreteSelect(w,1);

%Update Chain Struct
switch type
    
    case 't1_noBoundary'
        
        if u==1 %No Change
            Chain.x(1) = 1;  
            
        elseif u==2 %Add New Partition
            Chain.x = Chain.x+1;
            Chain.x(1) = 1;
            newLambda = sampleLambdaPosterior(wt(1),Chain);
            Chain.lambda = [newLambda;Chain.lambda];  
            
        else
            error([type ' error'])
            
        end
      
    case 't1_rightBoundary'
        
        if u==1 %Merge Right
            Chain.x(1) = 2;
            Chain.x = Chain.x-1; 
            Chain.lambda(1) = [];
            
        elseif u==2 %Add New Partition
            Chain.x(1) = 1;
            newLambda = sampleLambdaPosterior(wt(1),Chain);
            Chain.lambda(1) = newLambda;
            
        else
            error([type ' error'])
            
        end
        
    case 'leftBoundary'
        
        if u==1 %no change
            Chain.x(t) = j;
            
        elseif u==2 %merge left
            Chain.x(t) = j-1;
            
        elseif u==3 %add new partition
            Chain.x(t+1:end) = Chain.x(t+1:end)+1;
            newLambda = sampleLambdaPosterior(wt(t),Chain);
            Chain.lambda = [Chain.lambda(1:j-1);newLambda;Chain.lambda(j:end)];
            
        else
            error([type ' error'])
            
        end
        
    case 'rightBoundary'
        
        if u==1 %no change
            Chain.x(t) = j;
            
        elseif u==2 %merge right
            Chain.x(t) = j+1;
            
        elseif u==3 %add new partition
            Chain.x(t) = j+1;
            Chain.x(t+1:end) = Chain.x(t+1:end)+1;
            newLambda = sampleLambdaPosterior(wt(t),Chain);
            Chain.lambda = [Chain.lambda(1:j);newLambda;Chain.lambda(j+1:end)];
            
        else
            error([type ' error'])
            
        end
        
    case 'doubleBoundary'
        
        if u==1 %merge left
            Chain.x(t) = j-1;
            Chain.x(t+1:end) = Chain.x(t+1:end)-1;
            Chain.lambda(j) = [];
            
        elseif u==2 %merge right
            Chain.x(t+1:end) = Chain.x(t+1:end)-1;
            Chain.lambda(j) = [];
            
        elseif u==3 %add new partition
            newLambda = sampleLambdaPosterior(wt(t),Chain);
            Chain.lambda(j) = newLambda;
            
        else
            error([type ' error'])
        end
      
    case 'tN_noBoundary'
        
        if u==1 %no change
            Chain.x(t) = j;
        
        elseif u==2 %add new partition
            Chain.x(t) = j+1;
            newLambda = sampleLambdaPosterior(wt(t),Chain);
            Chain.lambda = [Chain.lambda;newLambda];
            
        else
            error([type ' error'])
        end

    case 'tN_leftBoundary'
        
        if u==1 %merge left
            Chain.x(t) = j-1;
            Chain.lambda(j) = [];
            
        elseif u==2 %add new partition
            newLambda = sampleLambdaPosterior(wt(t),Chain);
            Chain.lambda(j) = newLambda;

        else
            error([type ' error'])
            
        end
end

%**************************************************************************
function x = discreteSelect(w,n)

%Draw n samples from discrete PDF
x = zeros(1,n);
for ii = 1:n
    cdf = [0;cumsum(w(:)/sum(w))];
    u = rand;
    for kk = 2:length(cdf)
        if u>=cdf(kk-1) && u<=cdf(kk)
            x(ii) = kk-1;
            break
        end
    end
end

%**************************************************************************
function History = historyInit(rt,ngibbs,Chain)

%Setup
T = length(rt);
zvec = zeros(1,ngibbs+1);
zmat = zeros(T,ngibbs+1);

%Init
History.alpha = zvec;
History.c = zvec;
History.d = zvec;
History.q = zvec;
History.x = zmat;
History.lambda = zmat;
History.lambdac = cell(1,ngibbs+1);
History.mu = zmat;
History.muvar = zmat;
History.LP = zvec;
History.mu0 = zvec;
History.V0 = zvec;
History.pval = zvec;

%Set First Value
History.alpha(1) = Chain.alpha;
History.c(1) = Chain.c;
History.d(1) = Chain.d;
History.q(1) = Chain.q;
History.x(:,1) = Chain.x;
History.lambda(:,1) = Chain.lambda(Chain.x);
History.lambdac{1} = Chain.lambda;
History.mu(:,1) = Chain.mu;
History.muvar(:,1) = Chain.muvar;
History.LP(1) = logPosterior(rt,Chain);
History.mu0(1) = Chain.mu0;
History.V0(1) = Chain.V0;
History.pval(1) = 0;

%**************************************************************************
function LP = logPosterior(rt,Chain)

%p(r|mu,x,lambda)
L1 = sum(log(normpdf(rt,Chain.mu,1./sqrt(Chain.lambda(Chain.x)))));

%p(mu|mu0)
T = length(rt);
L2 = log(normpdf(Chain.mu(1),Chain.mu0,Chain.q));
for t = 2:T
   L2 = L2 + log(normpdf(Chain.mu(t),Chain.mu(t-1),Chain.q));
end

%p(x|alpha)
n = hist(Chain.x,unique(Chain.x));
L3 = sum(log(Chain.alpha * beta(n,Chain.alpha+1)));

%p(lambda|c,d)
L4 = sum(log(gampdf(Chain.lambda,Chain.c,1/Chain.d)));

%Total
LP = L1+0*(L2+L3+L4);

%Error Chk
if any(isnan(LP)) || any(isinf(LP))
    error('Calculation problem in logPosterior')
end

%**************************************************************************
function Chain = gibbsInit(rt,Config,Flag)

%Setup
Chain.a = Config.a;
Chain.b = Config.b;
Chain.c = Config.c;
Chain.d = Config.d;
Chain.u = Config.u;
Chain.v = Config.v;
Chain.alpha = Config.alpha;
Chain.x = [1;rt(2:end)*0];
Chain.lambda = sampleLambdaPosterior(rt(1),Chain);
Chain.q = Config.q;
Chain.mu0 = Config.mu0;
Chain.V0 = Config.V0;

%Assume Zero-Mean
wt = rt;

%Init Chain
k = 1;
for t = 2:length(Chain.x)
    
    %Transition Probabilities
    w = forwardTransitionWeights(wt(t),Chain,t);
    
    %Sample Transition
    u = rand;
    regimeChange = u<w(2);
    if regimeChange
        k = k+1;
        newLambda = sampleLambdaPosterior(wt(t),Chain);
        Chain.lambda = [Chain.lambda;newLambda];
    end
    Chain.x(t) = k;
end

%Init Mean
Chain = kalman(rt,Chain,Flag);

%**************************************************************************
function Chain = kalman(rt,Chain,Flag)

%Setup
T = length(rt);
R = 1./Chain.lambda(Chain.x);
Q = Chain.q^2;
mu0 = Chain.mu0;
V0 = Chain.V0;
mu = zeros(T,1);
P = zeros(T,1);
V = zeros(T,1);

%Forward Recursion
K = V0 / (V0 + R(1));
mu(1) = mu0 + K * (rt(1) - mu0);
V(1) = (1 - K) * V0;
P(1) = V(1) + Q;
for t = 2:T
    K = P(t-1) / (P(t-1) + R(t));
    mu(t) = mu(t-1) + K * (rt(t) - mu(t-1));
    V(t) = (1 - K) * P(t-1);
    P(t) = V(t) + Q;
end

%Backward Recursion
mu_hat = mu * 0;
V_hat = V * 0;
J = V * 0;
mu_hat(end) = mu(end);
V_hat(end) = V(end);
for t = T-1:-1:1
    J(t) = V(t)/P(t);
    mu_hat(t) = mu(t) + J(t) * (mu_hat(t+1) - mu(t));
    V_hat(t) = V(t) + J(t) * (V_hat(t+1) - P(t)) * J(t);
end

%Estimate Plant Noise (Maximum Likelihood)
qhat = Chain.q;
if Flag.estimate_q
    Q = 0;
    for t = 2:T
        Qcur = 0;
        Qcur = Qcur + (V_hat(t) + mu_hat(t)^2);
        Qcur = Qcur - 2 * (V_hat(t)*J(t-1) + mu_hat(t)*mu_hat(t-1));
        Qcur = Qcur + (V_hat(t-1) + mu_hat(t-1)^2);
        Q = Q + Qcur;
    end
    qhat = sqrt(Q/(T-2));
end

%Update Markov Chain
Chain.mu = mu_hat;
Chain.muvar = V_hat;
Chain.q = qhat;
Chain.mu0 = Chain.mu(1);
Chain.V0 = Chain.muvar(1);

%**************************************************************************
function Chain = sampleLambdaPosteriorBatch(wt,Chain)

%Error Chk
N = max(Chain.x);
if Chain.x(end)~=N
    error('Partition Sampling Error')
end

%Loop
for kk = 1:N
    mask = Chain.x==kk;
    data = wt(mask);
    Chain.lambda(kk) = sampleLambdaPosterior(data,Chain);
end

%**************************************************************************
function lambda = sampleLambdaPosterior(data,Chain)

%Compute Posterior Parameters
N = length(data);
cN = Chain.c + N/2;
dN = Chain.d + 0.5 * sum(data.^2);

%Sample Posterior
lambda = gamrnd(cN,1/dN,1);

%**************************************************************************
function Chain = sampleAlphaPosterior(Chain)

%Setup
n = hist(Chain.x,unique(Chain.x));
N = length(n);
w = -log(betarnd(Chain.alpha+1,n));

%Draw
Chain.alpha = gamrnd(Chain.a+N,1/(Chain.b+sum(w)));

%**************************************************************************
function w = forwardTransitionWeights(rt,Chain,ptr)

%Get Run-Length
xlast = Chain.x(ptr-1);
n = sum(Chain.x(1:ptr-1)==xlast);

%Compute Prior
alpha = Chain.alpha;
prior = [n./(n+alpha),alpha./(n+alpha)];

%Compute Transition Weights
sigma = 1/sqrt(Chain.lambda(xlast));
L0 = normpdf(rt,0,sigma);
L1 = student(rt,0,Chain.c/Chain.d,2*Chain.c);
w = [L0,L1].*prior;
w = w./sum(w);

%**************************************************************************
function p = student(data,mu,prec,dof)

%Student's-t PDF
C = gamma(dof/2+0.5) / gamma(dof/2) * sqrt(prec/pi/dof);
p = C * (1 + prec / dof * (data-mu).^2).^(-dof/2-0.5);

%**************************************************************************
function [a,b,reject] = metropolisHastings(x,n,varargin)
%METROPOLISHASTINGS(X,N) Sample gamma hyperparameters w/ Metropolis-Hastings
%   [A,B] = SAMPLEGAMMAPARM(X,N) returns N samples from the posterior of
%   the gamma distribution's shape and rate parameters A and B given a set 
%   of observations X using the Metropolis-Hastings algorithm. The proposal
%   distribution used is an isotropic truncated Gaussian distribution
%   centered on the previous [A,B] sample. Truncation is used to make sure 
%   no negative values of [A,B] are proposed. 
%
%   The observations are assumed to be drawn from the following model:
%
%   [A,B] ~ p(A,B)
%       X ~ Gamma(X|A,B)
%
%   The program then returns N samples from the posterior:
%
%   [A,B] ~ p(A,B|X)
%
%   [A,B,REJECT] = SAMPLEGAMMAPARM(X,N) returns the REJECT mask and
%   indicating which samples of A and B were rejected by the 
%   Metropolis-Hastings update.
%
%   [A,B,REJECT] = SAMPLEGAMMAPARM(...,'PARAM',VALUE) allows the use
%   of optional input parameters:
%       'sigma' - Use custom sigma for the proposal distribution [default = 0.1]. 
%       'icond' - Set the initial state of the Markov chain (default = [0;0]).
%

%Setup
X = zeros(2,n);
SIGMA = 0.1;
reject = zeros(1,n);
ICOND = [0;0];

%Optional Input Args
if nargin>2
    narg = length(varargin);
    for kk = 1:2:narg
        switch varargin{kk}
            case 'sigma'
                SIGMA = varargin{kk+1};   
            case 'icond'
                ICOND = varargin{kk+1}; 
            otherwise
                error('Unknown parameter')
        end
    end
end

%Set Initial State
X(:,1) = ICOND(:);

%Proposal Distribution
if length(SIGMA)==2
    SIGMA = [SIGMA(1)^2,0;0,SIGMA(2)^2];
    Q = @(z,mu,sig)mvnpdf(z,mu,sig)/(mvncdf([0;0],inf(2,1),mu,sig));
else
    Q = @(z,mu,sig)mvnpdf(z,mu,sig^2*eye(2))/(mvncdf([0;0],inf(2,1),mu,sig^2*eye(2)));
end

%Target Distribution
logP = @(z)sum(log(x.^(z(1)-1).*exp(-z(2)*x) / gamma(z(1)) / z(2)^(-z(1))));

%Run
for t = 1:n

    %Propose Move
    while(1)
        temp = X(:,t)+SIGMA*randn(2,1);
        if all(temp>=0)
            Y = temp;
            break
        end
    end
    
    %Accept/Reject Move
    ratio = exp(logP(Y)-logP(X(:,t))) * Q(X(:,t),Y,SIGMA)/Q(Y,X(:,t),SIGMA);
    if isnan(ratio)
        disp('WARNING: NaNs found MH Sampler')
    end
    A = min(1,ratio);
    U = rand;
    if U<=A
        X(:,t+1) = Y;
        reject(t+1) = false;
    else
        X(:,t+1) = X(:,t);
        reject(t+1) = true;
    end  
end

%Configure Output
reject = reject(2:end);
a = X(1,2:end);
b = X(2,2:end);


