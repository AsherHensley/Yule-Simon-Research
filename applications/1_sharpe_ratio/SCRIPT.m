%SCRIPT() Example Script for Computing Sharpe-Ratio
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

%Clean Up
close all
clear
clc

%Import Closing Prices
x = read_yahoo_csv('GSPC.csv');

%Compute Log-Returns
x.return = diff(log(x.close));

%Plot 
x.train_mask = 1:999;
figure
dd = datevec(x.date{1})*[1,1/12,1/365,0,0,0]';
plot(dd,x.close,'r'),hold on,grid on
plot(dd(x.train_mask),x.close(x.train_mask),'b','linewidth',2)
axis tight
ax = axis;
plot(dd(x.train_mask(end))*[1,1],ax(3:4),'k--')
set(gca,'fontsize',11);
xlabel('Year')
ylabel('Daily Closing Price')
title('S&P 500')
legend('Out-of-Sample','In-Sample')

%Set Random Number Generator
seed = 7;
rng(seed);

%Sample Yule-Simon Posterior
[Chain,History,BestFit] = yulesimon(x.return(x.train_mask));

%Compute Sharpe-Ratio
burn_in = 500;
risk_free_rate = 0;
S = (History.mu(end,burn_in:end) - risk_free_rate/sqrt(252)) .* sqrt(History.lambda(end,burn_in:end)) * sqrt(252);

%Plot Sampler Results
sharpe_ratio_plots(S,x,BestFit,History)



