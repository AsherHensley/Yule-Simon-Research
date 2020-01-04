function rt = sample_generator(Chain,m)
%SAMPLE_GENERATOR(...) Simulate Yule-Simon Generative Model
%   RT = SAMPLE_GENERATOR(CHAIN,M) Returns M log-return samples RT given 
%   the state of the Markov chain CHAIN from Yule-Simon generative model  
%   given by:
%
%         lambda(j)|c,d ~ Gamma(c,d)
%   s(t)|x(1:t-1),alpha ~ Bernoulli(alpha/(nxt+alpha))
%      w(t)|x(t),lambda ~ Gaussian(0,1/lambda(x(t)))
%                v(t)|q ~ Gaussian(0,q^2)
%                  x(t) = x(t-1) + s(t)
%                 mu(t) = mu(t-1) + v(t)
%                  r(t) = mu(t) + w(t)
%
%   where nxt = sum(x(t)==x(1:t-1)) and log returns are given by r(t).
%
%   Revisions:
%   1.0     04-Jan-2020     Hensley     Initial release
%
%   MIT License
% 
%   Copyright (c) 2020 Asher A. Hensley Research
%   $Revision: 1.0 $  $Date: 2020/01/04 12:00:01 $
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

%Init
n = 1;
rt = zeros(n,m);
nxt = sum(Chain.x==Chain.x(end));
p = nxt/(nxt+Chain.alpha);

for ii = 1:m
    
    %Yule Simon Partitions
    u = rand<p;
    if u
        lambda = Chain.lambda(end);
    else
        lambda = gamrnd(Chain.c,1/Chain.d,1,1);
    end
    w = 1/sqrt(lambda) .* randn(1,n);
    
    %Mean Process
    mu = Chain.mu(end)/100 + 5e-5 * randn;
    
    %Log Returns
    rt(:,ii) = mu + w/100;
    
end

