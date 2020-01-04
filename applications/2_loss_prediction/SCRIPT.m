%SCRIPT() Example Script for Loss Prediction Simulation
%
%   Revisions:
%   1.0     03-Jan-2020     Hensley     Initial release
%
%   MIT License
%
%   Copyright (c) 2020 Asher A. Hensley Research
%   $Revision: 1.0 $  $Date: 2020/01/03 12:00:01 $
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

%Path Setup
addpath(genpath('../../'))

%Flag to Load Preprocessed Results (This takes awhile to run...)
load_results_from_mat = true;

%Begin
if load_results_from_mat
    load results.mat
else
    
    %Import Closing Prices
    x = read_yahoo_csv('MSFT.csv');
    
    %Compute Log-Returns
    x.return = diff(log(x.close));
    
    %Set Random Number Generator
    seed = 7;
    rng(seed);
    
    %Init Chain
    train_mask = 1:500;
    [Chain0,History0,BestFit0] = yulesimon(x.return(train_mask));
    
    %Setup
    ngibbs = 20;
    nsteps = 8000;
    hits01 = zeros(1,nsteps);
    hits05 = zeros(1,nsteps);
    var01 = zeros(1,nsteps);
    var05 = zeros(1,nsteps);
    cvar01 = zeros(1,nsteps);
    cvar05 = zeros(1,nsteps);
    Chain = Chain0;
    
    %Run
    for kk = 1:nsteps
        disp(kk)
        [Chain,History,BestFit] = yulesimon(x.return(train_mask),'Chain',Chain,'ngibbs',ngibbs,'waitbar',false);
        rt = sample_generator(Chain,50000);
        var01(kk) = prctile(rt,1);
        cvar01(kk) = mean(rt(rt<var01(kk)));
        if x.return(train_mask(end)+1)<var01(kk)
            hits01(kk) = 1;
        end
        var05(kk) = prctile(rt,5);
        cvar05(kk) = mean(rt(rt<var05(kk)));
        if x.return(train_mask(end)+1)<var05(kk)
            hits05(kk) = 1;
        end
        train_mask = train_mask + 1;
    end
    
end

%Compute Plain Vanilla Estimates 
idx01 = find(hits01)+499;
ref01 = zeros(size(idx01));
for kk = 1:length(idx01)
    win = x.return(idx01(kk)-499:idx01(kk));
    ref01(kk) = prctile(win,1);
end
idx05 = find(hits05)+499;
ref05 = zeros(size(idx05));
for kk = 1:length(idx05)
    win = x.return(idx05(kk)-499:idx05(kk));
    ref05(kk) = prctile(win,5);
end

%Print Threshold Exceedances
disp(['  Percent of Data < 1% Threshold: ',num2str(mean(hits01)*100),'%'])
disp(['  Percent of Data < 5% Threshold: ',num2str(mean(hits05)*100),'%'])

%Plot Results
figure
subplot(211)
ret = x.return(501:8500);
plot(ret(hits05==1))
hold on
plot(cvar05(hits05==1),'r')
plot(ref05,'k')
grid on
axis tight
legend('Actual Loss','Predicted Loss (Yule-Simon)','Predicted Loss (Plain Vanilla)','location','southeast')
title('5% Tail')
ylabel('Log Return')
xlabel('Sample')

subplot(212)
ret = x.return(501:8500);
plot(ret(hits01==1))
hold on
plot(cvar01(hits01==1),'r')
plot(ref01,'k')
grid on
axis tight
legend('Actual Loss','Predicted Loss (Yule-Simon)','Predicted Loss (Plain Vanilla)','location','southeast')
title('1% Tail')
ylabel('Log Return')
xlabel('Sample')

