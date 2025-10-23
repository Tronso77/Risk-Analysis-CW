clear; close all; clc; format short;

% Daily adjusted close stock prices from 22/02/2022 to 20/02/2024 (both inclusive) 
% AA INTC JPM PG ^GSPC

filename = 'q4_dataset.csv';
dataset = readtable(filename,"VariableNamingRule","preserve");

colLabels = dataset.Properties.VariableNames;
tickers = colLabels(:,2:end);

histDates = dataset{:, 1};
histPrices = dataset{:, 2:end-1};
histMktPrices = dataset{:, end};

% Log-returns of stocks
logStockRet = log(histPrices(2:end,:) ./ histPrices(1:end-1,:));
% Log-returns of the market (S&P 500 Index)
logMktRet = log(histMktPrices(2:end,:) ./ histMktPrices(1:end-1,:));

[nObs, nStocks] = size(logStockRet);

% Estimating the covariance matrix

% (1) Sample estimator

sampleSIGMA = cov(logStockRet);

% (2) Shrinkage estimator with the single-index model as the prior

beta = zeros(nStocks,1);
resVar = zeros(nStocks,1);
for i = 1:nStocks
    % Linear regression model: r_i = a_i + b_i * rm + eps_i
    model = fitlm(logStockRet(:, i), logMktRet);
    b = model.Coefficients.Estimate;
    beta(i,1) = b(2);
    resVar(i,1) = model.MSE;
    % Check model fit
    model.Rsquared.Adjusted;
    model.Coefficients;
end

F = beta.*beta'.*var(logMktRet) + diag(resVar);
shrinkage = optimalShrinkage(logStockRet,logMktRet,nObs,nStocks,sampleSIGMA,F);
singleIndexModelSIGMA = shrinkage*F+(1-shrinkage)*sampleSIGMA;

% (3) Shrinkage estimator with constant correlation model as the prior

[LedoitWolfSIGMA,shrinkageLW] = get_LedoitWolfCov(logStockRet);

% Analysis between the different covariance matrices

CN = @(SIGMA) max(eig(SIGMA)) / min(eig(SIGMA));
CN(singleIndexModelSIGMA);
CN(sampleSIGMA);
CN(LedoitWolfSIGMA);

%% NORMALITY TEST
% qqplot(logStockRet(:,1));
% ylabel('AA Log-Return Quantiles');
% title('QQ Plot of AA Log-returns');
% exportgraphics(gca, 'qqplot-aa-log-returns.pdf', 'ContentType', 'vector', 'Resolution', 300);

% qqplot(logStockRet(:,2));
% ylabel('INTC Log-Return Quantiles');
% title('QQ Plot of INTC Log-returns');
% exportgraphics(gca, 'qqplot-intc-log-returns.pdf', 'ContentType', 'vector', 'Resolution', 300);

% qqplot(logStockRet(:,3));
% ylabel('JPM Log-Return Quantiles');
% title('QQ Plot of JPM Log-returns');
% exportgraphics(gca, 'qqplot-jpm-log-returns.pdf', 'ContentType', 'vector', 'Resolution', 300);

% qqplot(logStockRet(:,4));
% ylabel('PG Log-Return Quantiles');
% title('QQ Plot of PG Log-returns');
% exportgraphics(gca, 'qqplot-pg-log-returns.pdf', 'ContentType', 'vector', 'Resolution', 300);

% mu = mean(logStockRet);
% vol = std(logStockRet);
% sk = skewness(logStockRet);
% excessK = kurtosis(logStockRet) - 3;
% for i=1:nStocks
%     [h,~,jbstat,critval] = jbtest(logStockRet(:,i),0.01);
% end

%% FULL REVALUATION

% Start inputs -----------------------------------------------------

OPTION_CONTRACT_SIZE = 1;

% Adj. Close Feb 20,2024
S0 = histPrices(end,:);

% B-S pricing
rf = 0.04;
q = 0;

K_AA = fix(1.05.*S0(1)); % 105% of the stock price
K_INTC = fix(0.9.*S0(2)); % 90% of the stock price
K_JPM = fix(S0(3)); % ATM
K_PG = fix(1.1.*S0(4)); % 110% of the stock price

T_AA = 12/12;
T_INTC = 9/12;
T_JPM = 6/12;
T_PG = 9/12;

wAA = 6; % Long 6 option contracts
wINTC = -3; % Short 3 option contracts
wJPM = 6; % Long 6 option contracts
wPG = -2; % Short 2 option contracts

K = [K_AA K_INTC K_JPM K_PG];
T = [T_AA T_INTC T_JPM T_PG];
w = [wAA wINTC wJPM wPG];

% Number of MC or bootstrap simulations
M = 10000;

n = 10; % VaR holding period horizon (in days) 
confLevel = 0.99; % VaR confidence level

% End inputs -----------------------------------------------------

vol = sqrt(diag(sampleSIGMA).*252);

[C0, P0] = blsprice(S0, K, rf, T, vol', q);
Vi0 = [w(1,1:2).*C0(1,1:2) w(1,3:4).*P0(1,3:4)];
Vp0 = sum(Vi0);

fprintf('\n');
for i=1:1:nStocks
    fprintf('Intial Value of Option Position on %s: %.2f$\n', tickers{i}, Vi0(i));
end
fprintf('\nIntial Value of the Options Portfolio: %.2f$',Vp0);

% Simulate log-returns via Historical Simulation (non-parametric approach)
rng(123);
bVaR = zeros(M,1);
bES = zeros(M,1);

bMVaR = zeros(M,nStocks);
bMES = zeros(M,nStocks);

function [MVaR, MES] = calcMarginalRisk(VaR,VpPnL,ViPnL,N)
    MVaR = 999*ones(N,1);
    MES = 999*ones(N,1);
        
    eps = 0.001*VaR;
    lb = -VaR - eps;
    ub = -VaR + eps;
    posVaR = (VpPnL <= ub) & (VpPnL >= lb);
    
    posES = VpPnL <= -VaR;
    
    for i=1:1:N
        MVaRcondPnL = ViPnL(posVaR,i);
        MEScondPnL = ViPnL(posES,i);
        MVaR(i,1) = -mean(MVaRcondPnL);
        MES(i,1) = -mean(MEScondPnL)';

    end
end

function [VaR, ES] = calcRiskMeasure(port,cl)
    VaR = -prctile(port, (1-cl).*100);
    ES = -mean(port((port <= -VaR)));
end

for j=1:1:M
    bStockLogRet = zeros(nObs,nStocks);

    for i=1:1:nStocks
        u = randi(nObs, nObs, n);
        ret = logStockRet(:,i);
        temp = cumsum(ret(u));
        bStockLogRet(:,i) = temp(:,n);
    end
    
    bST = S0.*exp(bStockLogRet);
    % Revaluate each option position
    [bCT, bPT] = blsprice(bST, K.*ones(size(bST)), rf, T-(n./252).*ones(size(bST)), vol'.*ones(size(bST)), q);
    bViT = [w(1,1:2).*bCT(:,1:2) w(1,3:4).*bPT(:,3:4)];
    
    % Obtain the P&L of each option and of the portfolio
    bViPnL =  bViT-Vi0;
    bVpPnL = sum(bViPnL,2);
    
    [bVaR(j,1), bES(j,1)] = calcRiskMeasure(bVpPnL,confLevel);
    [bMVaR(j,:), bMES(j,:)] = calcMarginalRisk(bVaR(j,1),bVpPnL,bViPnL,nStocks);
end

nonParam_VaR = mean(bVaR,'omitmissing');
nonParam_ES = mean(bES,'omitmissing');
nonParam_MVaR = abs(mean(bMVaR,'omitmissing'));
nonParam_MES = abs(mean(bMES,'omitmissing'));

CVaR = nonParam_MVaR;
CES = nonParam_MES;
MVaR = [nonParam_MVaR(:,1:2)./w(:,1:2) nonParam_MVaR(:,3:4)./w(:,3:4)];
MES = [nonParam_MES(:,1:2)./w(:,1:2) nonParam_MES(:,3:4)./w(:,3:4)];

% Simulate log-returns via multivariate Gaussian distribution (parametric)
rng(123);
MU = mean(logStockRet);
SIGMA = sampleSIGMA;
mcStockLogRet = mvnrnd(MU'.*n, SIGMA.*n, M);
mcST = S0.*exp(mcStockLogRet);

% Revaluate the position of the portfolio
[mcCT, mcPT] = blsprice(mcST, K.*ones(size(mcST)), rf, T-(n./252).*ones(size(mcST)), vol'.*ones(size(mcST)), q);
mcViT = [w(1,1:2).*mcCT(:,1:2) w(1,3:4).*mcPT(:,3:4)];

% Obtain the P&L of each option and the portfolio
mcViPnL =  mcViT-Vi0;
mcVpPnL = sum(mcViPnL,2);

% Calculate the VaR and ES using the quantiles approach
[Gauss_VaR, Gauss_ES] = calcRiskMeasure(mcVpPnL,confLevel);

fprintf('\n\n');

fprintf('VaR (non-parametric): %.2f$\n', nonParam_VaR);
fprintf('ES  (non-parametric): %.2f$\n', nonParam_ES);
fprintf('VaR (Gaussian): %.2f$\n', Gauss_VaR);
fprintf('ES  (Gaussian): %.2f$\n', Gauss_ES);

fprintf('\n');
for i=1:1:nStocks
    fprintf('MVaR of Position on %s: %.2f$\n', tickers{i}, MVaR(i));
    fprintf('CVaR of Position on %s: %.2f$\n', tickers{i}, CVaR(i));
    fprintf('CVaR (%%) of Position on %s: %.2f%%\n', tickers{i}, (CVaR(i)./sum(CVaR))*100);
    fprintf('\n');
end

fprintf('\n');

for i=1:1:nStocks
    fprintf('MES of Position on %s: %.2f$\n', tickers{i}, MES(i));
    fprintf('CES of Position on %s: %.2f$\n', tickers{i}, CES(i));
    fprintf('CES (%%) of Position on %s: %.2f%%\n', tickers{i}, (CES(i)./sum(CES))*100);
    fprintf('\n');
end
