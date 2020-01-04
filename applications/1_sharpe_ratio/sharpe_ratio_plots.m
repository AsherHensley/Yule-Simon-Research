function sharpe_ratio_plots(S,x,BestFit,History)
%SHARPE_RATIO_PLOTS(...) Make Sharpe-Ratio Analysis Plots
%   SHARPE_RATIO_PLOTS(S,X,BESTFIT,HISTORY) make analysis plots based on 
%   Sharpe-ratio posterior samples S, price struct X, BESTFIT, and HISTORY
%   structs returned by yulesimon.m
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

%Set Fontsize
fs = 11;

%Apply Train Mask
dd = datevec(x.date{1})*[1,1/12,1/365,0,0,0]';
dd = dd(x.train_mask);
x.return = x.return(x.train_mask);

%Sharpe-Ratio
figure,
set(gcf,'outerposition',[392,6,578,860],'paperpositionmode','auto')
yyaxis left
subplot(4,2,1:2)
[H,B] = hist(S,50);
bar(B,H,'edgecolor','k','facecolor','w')
title('Sharpe-Ratio','fontsize',fs)
grid on
sample_s = mean(x.return(end-252:end)*100)/std(x.return(end-252:end)*100)*sqrt(252);
ax = axis;
hold on
plot(sample_s*[1,1],ax(3:4),'r')
legend('Posterior Samples','Historical Point Estimate')

%Log Returns
subplot(4,2,3:4)
plot(dd,x.return,'b')
axis tight
ax = axis;
hold on
plot(dd,BestFit.mu/100 + BestFit.sigma/100*2,'r')
plot(dd,BestFit.mu/100 - BestFit.sigma/100*2,'r')
plot(dd,x.return,'b')
set(gca,'fontsize',fs)
xlabel('Year')
ylabel('Log-Return')
grid on
ylim(1*ax(3:4))
title('Volatility Clustering')
legend('Observation','Volatility (2\sigma)','location','best')

%Model-Fit
subplot(4,2,5)
normplot(BestFit.rt_whitened)
set(gca,'box','on','fontsize',12)
xlabel('Standardized Log Returns')
text(-3,3,['p-value = ',num2str(BestFit.pval.chi2test)],...
    'fontsize',fs,'fontweight','bold')
title('Model Fit')

%Log-Posterior
subplot(4,2,6)
ngibbs = length(History.q);
plot(History.LP)
set(gca,'fontsize',fs)
grid on
axis tight
xlabel('Iteration')
ylabel('Log-Posterior')
ax = axis;
xlim([-0.05*ngibbs,ax(2)])
title('Convergence')

%Process Noise & Alpha
subplot(4,2,8)
yyaxis left
plot(History.q)
set(gca,'fontsize',fs)
grid on
axis tight
xlabel('Iteration')
ylabel('Process Noise')
ax = axis;
xlim([-0.05*ngibbs,ax(2)])
ylim([0,ax(4)])
title('State-Space Parameters')

yyaxis right
plot(History.alpha)
set(gca,'fontsize',fs)
grid on
axis tight
xlabel('Iteration')
ylabel('Power-Law Exponent')
ax = axis;
xlim([-0.05*ngibbs,ax(2)])
ylim([0,ax(4)])

%Gamma Hyper-Parameters
subplot(4,2,7)
plot(History.c,History.d)
set(gca,'fontsize',fs)
grid on
axis tight
xlabel('Shape')
ylabel('Rate')
ax = axis;
axis([0,ax(2),0,ax(4)])
title('Volatility Parameters')

