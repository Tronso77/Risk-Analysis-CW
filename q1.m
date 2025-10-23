clc; clear; close all;

%% Q1(a). LOAD DATA & BASIC STAT ANALYSIS
filename = 'cleaned_adj_close_data.xlsx'; 
stockData = readtable(filename, 'PreserveVariableNames', true);

dates = stockData.Date;  
prices = stockData{:, 2:end};  
[numDays, numStocks] = size(prices);

% 1) Compute daily log-returns
logPrices = log(prices);
dailyReturns = diff(logPrices);       % (numDays-1) x numStocks
returnDates = dates(2:end);          % Align with dailyReturns

% 2) Equally Weighted Portfolio
weights = ones(1, numStocks) / numStocks;
portfolioReturns = dailyReturns * weights';  % T x 1

% 3) Covariance & correlation among the 6 stocks
covMatrix  = cov(dailyReturns);
corrMatrix = corrcoef(dailyReturns);

% 4) Descriptive Stats (each stock)
stockMean   = mean(dailyReturns);   
stockStd    = std(dailyReturns);    
stockSkew   = skewness(dailyReturns);
stockKurt   = kurtosis(dailyReturns);

disp('--- Descriptive Stats for Each Stock (Daily Returns) ---');
for i = 1:numStocks
    fprintf('%s: Mean=%f, Std=%f, Skew=%f, Kurt=%f\n', ...
        stockData.Properties.VariableNames{i+1}, ...
        stockMean(i), stockStd(i), stockSkew(i), stockKurt(i));
end

% 5) Stats on the Portfolio
meanReturn = mean(portfolioReturns);
stdReturn  = std(portfolioReturns);
skewReturn = skewness(portfolioReturns);
kurtReturn = kurtosis(portfolioReturns);
minReturn  = min(portfolioReturns);
maxReturn  = max(portfolioReturns);

fprintf('\n--- Portfolio Descriptive Statistics ---\n');
fprintf('Mean Daily Return:       %f\n', meanReturn);
fprintf('Standard Deviation:      %f\n', stdReturn);
fprintf('Skewness:                %f\n', skewReturn);
fprintf('Excess Kurtosis:         %f\n', kurtReturn);
fprintf('Minimum Daily Return:    %f\n', minReturn);
fprintf('Maximum Daily Return:    %f\n', maxReturn);

% Extended correlation with portfolio
allReturns = [dailyReturns, portfolioReturns];
corrMatrixAll = corrcoef(allReturns);

% 6) Jarque-Bera normality test on portfolio
[h_jb, p_jb] = jbtest(portfolioReturns);
if h_jb == 1
    disp('Jarque-Bera test: Reject normality at the 5% level.');
else
    disp('Jarque-Bera test: Cannot reject normality at the 5% level.');
end
fprintf('p-value for JB test: %f\n', p_jb);

% 7) Plot distributional characteristics
figure;
histogram(portfolioReturns, 50, 'Normalization', 'pdf');
hold on;
mu_p = meanReturn; 
sigma_p = stdReturn;
xgrid = linspace(minReturn, maxReturn, 200);
pdfVals = normpdf(xgrid, mu_p, sigma_p);
plot(xgrid, pdfVals, 'r-', 'LineWidth', 2);
hold off;
title('Portfolio Returns Distribution');
xlabel('Daily Return'); ylabel('Density');
legend('Histogram', 'Normal PDF Overlay','Location','best');
exportgraphics(gca, 'Portfolio Returns Distribution.pdf', 'ContentType', 'vector', 'Resolution', 300);

grid on;

figure;
qqplot(portfolioReturns);
title('Q-Q Plot of Portfolio Returns vs. Normal Distribution');
exportgraphics(gca, 'Q-Q Plot of Portfolio Returns vs. Normal Distribution.pdf', 'ContentType', 'vector', 'Resolution', 300);

figure;
plot(returnDates, portfolioReturns);
datetick('x','yyyy');
xlabel('Date'); ylabel('Daily Return');
title('Time Series of Portfolio Daily Returns');
exportgraphics(gca, 'Time Series of Portfolio Daily Returns.pdf', 'ContentType', 'vector', 'Resolution', 300);

grid on;
%% Q1(b). Rolling 6-month VaR Computation 
T = length(portfolioReturns);
desiredStartDate = datetime(2014,7,1);  % July 1, 2014
nMonths = 6;  
alphaLevels = [0.90, 0.99];  % Two confidence levels
nAlphas = numel(alphaLevels);

% Find earliest index >= July 1, 2014
t0 = find(returnDates >= desiredStartDate, 1, 'first');
if isempty(t0)
    error('No trading days on or after July 1, 2014 in the dataset.');
end
fprintf('\nStarting VaR estimation on row %d => %s\n', t0, datestr(returnDates(t0)));

% We'll have 6 methods x 2 alpha => 12 columns of VaR
% (HS, Normal, MC, CF, BSHS, StudT)
varLabels = { ...
    'HS_90','HS_99',...
    'Normal_90','Normal_99',...
    'MC_90','MC_99',...
    'CF_90','CF_99',...
    'BSHS_90','BSHS_99',...
    'T_90','T_99' ...
};
varTable = array2table(NaN(T,numel(varLabels)), 'VariableNames', varLabels);
varTable.Date = returnDates; 
varTable = movevars(varTable, 'Date', 'Before', varLabels{1});

% Precompute normal inverses
z90 = norminv(0.10);  % -1.28155
z99 = norminv(0.01);  % -2.32635

% For Student-t MLE
opts = statset('MaxIter',2000,'MaxFunEvals',4000);
nSim = 10000;  % for MC + BSHS

% Rolling
muVec = NaN(T,1);
sigmaVec = NaN(T,1);

for t = t0 : T
    thisDate = returnDates(t);
    inSampleStart = thisDate - calmonths(nMonths);
    mask = (returnDates >= inSampleStart) & (returnDates < thisDate);
    windowData = portfolioReturns(mask);
    
    if length(windowData) < 30
        continue;
    end
    
    muNorm = mean(windowData);
    sigmaNorm = std(windowData);
    muVec(t) = muNorm;
    sigmaVec(t) = sigmaNorm;
    
    % 1) Historical Simulation (HS)
    sortedData = sort(windowData);
    hs_90 = prctile(sortedData, 10); % 10% => VaR(90%)
    hs_99 = prctile(sortedData, 1);  % 1% => VaR(99%)
    
    % 2) Parametric Normal
    muNorm    = mean(windowData);
    sigmaNorm = std(windowData);
    norm_90 = muNorm + z90*sigmaNorm;
    norm_99 = muNorm + z99*sigmaNorm;
    
    % 3) Monte Carlo (Normal-based)
    simData = muNorm + sigmaNorm .* randn(nSim,1);
    mc_90 = quantile(simData, 0.10);
    mc_99 = quantile(simData, 0.01);
    
    % 4) Cornish-Fisher
    sampleSkew = skewness(windowData);
    sampleKurt = kurtosis(windowData); 
    exKurt = sampleKurt - 3; 
    
    cornishFisherZ = @(zVal, skewVal, eKurtVal) ...
        zVal + (zVal^2 - 1)*skewVal/6 + ...
               (zVal^3 - 3*zVal)*eKurtVal/24;
    
    zcf_90 = cornishFisherZ(z90, sampleSkew, exKurt);
    zcf_99 = cornishFisherZ(z99, sampleSkew, exKurt);
    cf_90 = muNorm + zcf_90*sigmaNorm;
    cf_99 = muNorm + zcf_99*sigmaNorm;
    
    % 5) Bootstrap HS
    nWindow = length(windowData);
    bootIdx = randi(nWindow, nSim*nWindow, 1);
    bootSample = windowData(bootIdx);
    bhs_90 = prctile(bootSample, 10);
    bhs_99 = prctile(bootSample, 1);
    
    % 6) Student-t (Method of Moments)
    muMm = mean(windowData);       % location = sample mean
    kurtVal = kurtosis(windowData);% sample kurtosis

    % if kurtVal <= 3, the formula for nuMm will blow up or become <= 2
    if kurtVal <= 3
        % fallback => store NaN or infinite VaR
        t_90 = NaN;
        t_99 = NaN;
    else
        % degrees of freedom:
        nuMm = 4 + 6 / (kurtVal - 3); 
        % if nuMm <=2 => infinite or undefined variance => fallback
        if nuMm <= 2
            t_90 = NaN;
            t_99 = NaN;
        else
            % scale parameter via variance formula: sigma^2 = ((nu - 2)/nu) * sampleVar
            sampleVar = var(windowData);
            sigma2Mm = ((nuMm - 2)/nuMm) * sampleVar;
            if sigma2Mm <= 0
                % fallback if negative (which can happen if nuMm ~ 2)
                t_90 = NaN;
                t_99 = NaN;
            else
                sigmaMm = sqrt(sigma2Mm);

                % compute VaR at 90%, 99%
                % tinv(0.10,nuMm) is negative, so final t_90 will typically be less than muMm.
                t_90 = muMm + sigmaMm * tinv(0.10, nuMm);
                t_99 = muMm + sigmaMm * tinv(0.01, nuMm);
            end
        end
    end

    
    varTable.HS_90(t)    = hs_90;
    varTable.HS_99(t)    = hs_99;
    varTable.Normal_90(t)= norm_90;
    varTable.Normal_99(t)= norm_99;
    varTable.MC_90(t)    = mc_90;
    varTable.MC_99(t)    = mc_99;
    varTable.CF_90(t)    = cf_90;
    varTable.CF_99(t)    = cf_99;
    varTable.BSHS_90(t)  = bhs_90;
    varTable.BSHS_99(t)  = bhs_99;
    varTable.T_90(t)     = t_90;
    varTable.T_99(t)     = t_99;
end

firstValid = find(~isnan(varTable.HS_90),1,'first');
lastValid  = find(~isnan(varTable.HS_90),1,'last');
if ~isempty(firstValid)
    fprintf('\nFirst valid VaR date: row %d => %s\n', ...
        firstValid, datestr(varTable.Date(firstValid)));
    fprintf('Last valid VaR date: row %d => %s\n', ...
        lastValid, datestr(varTable.Date(lastValid)));
    
    disp(varTable(firstValid : min(firstValid+5,height(varTable)), :));
else
    disp('No valid VaR rows found. Possibly not enough data in the 6-month window.');
end

%% HERE PLOTS and Graphs for each methods
nonParam90 = {'HS_90','BSHS_90'};
param90    = {'Normal_90','MC_90','CF_90','T_90'};
nonParam99 = {'HS_99','BSHS_99'};
param99    = {'Normal_99','MC_99','CF_99','T_99'};

% Define a color mapping for consistency
colorMap = containers.Map;
colorMap('HS')     = [0.00 0.45 0.74]; % bluish
colorMap('BSHS')   = [0.85 0.33 0.10]; % reddish
colorMap('Normal') = [0.93 0.69 0.13]; % yellowish
colorMap('MC')     = [0.49 0.18 0.56]; % purple
colorMap('CF')     = [0.47 0.67 0.19]; % green
colorMap('T')      = [0.30 0.75 0.93]; % cyan

getMethodColor = @(mName) colorMap( extractBefore(mName, '_') );

figure('Color',[1 1 1]);

subplot(2,2,1);
hold on;
for i = 1:numel(nonParam90)
    mName = nonParam90{i};
    c = getMethodColor(mName);
    plot(varTable.Date, varTable.(mName), '-.', 'LineWidth',1.5, ...
         'Color', c, 'DisplayName', mName);
end
xlabel('Date'); ylabel('VaR (90%)');
title('Non-Parametric (90%)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

subplot(2,2,2);
hold on;
for i = 1:numel(param90)
    mName = param90{i};
    c = getMethodColor(mName);
    plot(varTable.Date, varTable.(mName), '-.', 'LineWidth',1.5, ...
         'Color', c, 'DisplayName', mName);
end
xlabel('Date'); ylabel('VaR (90%)');
title('Parametric (90%)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

subplot(2,2,3);
hold on;
for i = 1:numel(nonParam99)
    mName = nonParam99{i};
    c = getMethodColor(mName);
    plot(varTable.Date, varTable.(mName), '-.', 'LineWidth',1.5, ...
         'Color', c, 'DisplayName', mName);
end
xlabel('Date'); ylabel('VaR (99%)');
title('Non-Parametric (99%)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

subplot(2,2,4);
hold on;
for i = 1:numel(param99)
    mName = param99{i};
    c = getMethodColor(mName);
    plot(varTable.Date, varTable.(mName), '-.', 'LineWidth',1.5, ...
         'Color', c, 'DisplayName', mName);
end
xlabel('Date'); ylabel('VaR (99%)');
title('Parametric (99%)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

% Optionally, align y-axis limits for each row (adjust as needed)
subplot(2,2,1); ylim([-0.06 0]);
subplot(2,2,2); ylim([-0.06 0]);
subplot(2,2,3); ylim([-0.10 -0.02]);
subplot(2,2,4); ylim([-0.10 -0.02]);

exportgraphics(gcf, 'VaR_Estimates.pdf', 'ContentType', 'vector', 'Resolution',300);





%% Q1(c).VaR Violations for each method

finalCols = {'HS_90','HS_99','Normal_90','Normal_99','MC_90','MC_99',...
             'CF_90','CF_99','BSHS_90','BSHS_99','T_90','T_99'};
nMethods = numel(finalCols);

fullNames = { 'Historical Simulation (90%)', 'Historical Simulation (99%)',...
              'Parametric Normal (90%)',      'Parametric Normal (99%)',...
              'Monte Carlo Simulation (90%)', 'Monte Carlo Simulation (99%)',...
              'Cornish-Fisher (90%)',         'Cornish-Fisher (99%)',...
              'Bootstrapped HS (90%)',        'Bootstrapped HS (99%)',...
              'Student-t (90%)',              'Student-t (99%)'};

violationsCount = zeros(nMethods,1);
validDaysCount = zeros(nMethods,1);
violationFraction = zeros(nMethods,1);

for i = 1:nMethods
    colName = finalCols{i};
    thisVaR = varTable.(colName);
    % Consider only days with a valid (non-NaN) VaR forecast
    maskValid = ~isnan(thisVaR);
    validVaR = thisVaR(maskValid);
    validRet = portfolioReturns(maskValid);
    
    % A violation occurs if the actual return is less than the VaR threshold.
    % VaR is stored as a negative number representing a loss threshold
    maskViol = (validRet < validVaR);
    violationsCount(i) = sum(maskViol);
    validDaysCount(i) = sum(maskValid);
    violationFraction(i) = violationsCount(i) / validDaysCount(i);
end


methodCats = categorical(fullNames);
methodCats = reordercats(methodCats, fullNames);

% summary table 
resultsTable = table(fullNames', violationsCount, validDaysCount, violationFraction, ...
    'VariableNames', {'VaRMethod','NumViolations','NumValidDays','ViolationFraction'});
disp('Number of VaR Violations per Method');
disp(resultsTable);


%%
%  bar chart
figure('Color',[1 1 1]);
bar(methodCats, violationFraction, 'FaceColor', [0.2 0.6 0.8]);
xlabel('VaR Method');
ylabel('Violation Fraction');
title('Fraction of VaR Violations per Method');
grid on;

xtips = get(gca, 'XTick');
ytips = violationFraction;
labels = compose('%.2f%%', violationFraction * 100);
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
exportgraphics(gcf, 'VaR_viol.pdf', 'ContentType', 'vector', 'Resolution',300);

%% Q1(d) - Kupiec and Christoffersen tests
finalCols = {'HS_90','HS_99','Normal_90','Normal_99','MC_90','MC_99',...
             'CF_90','CF_99','BSHS_90','BSHS_99','T_90','T_99'};
nMethods = numel(finalCols);

violationCounts = zeros(nMethods,1);
validDays       = zeros(nMethods,1);
kupiec_LR       = zeros(nMethods,1);
kupiec_p        = zeros(nMethods,1);
christoffersen_LR_ind = zeros(nMethods,1);
christoffersen_p_ind  = zeros(nMethods,1);
condCover_LR    = zeros(nMethods,1);
condCover_p     = zeros(nMethods,1);

for i = 1:nMethods
    colName = finalCols{i};
    
    if contains(colName,'90')
        alpha = 0.90;
    elseif contains(colName,'99')
        alpha = 0.99;
    else
        alpha = 0.90;
    end
    
    VaR_series = varTable.(colName);
    maskValid = ~isnan(VaR_series);
    validVaR  = VaR_series(maskValid);
    validRet  = portfolioReturns(maskValid);
    validDays(i) = sum(maskValid);
    
    hits = double(validRet < validVaR);
    violationCounts(i) = sum(hits);
    
    T_valid = validDays(i);
    x = violationCounts(i);
    pStar = 1 - alpha;
    
    % ----------------- Kupiec (Unconditional Coverage) -------------------
    if x == 0 || x == T_valid
        LR_uc = Inf;
        p_uc = 0;
    else
        lnL_H0 = x*log(pStar) + (T_valid - x)*log(1 - pStar);
        lnL_H1 = x*log(x/T_valid) + (T_valid - x)*log(1 - x/T_valid);
        LR_uc  = -2 * (lnL_H0 - lnL_H1);
        LR_uc  = max(0, LR_uc);
        p_uc   = 1 - chi2cdf(LR_uc, 1);
    end
    kupiec_LR(i) = LR_uc;
    kupiec_p(i)  = p_uc;
    
    % ----------------- Christoffersen Independence Test ------------------
    n00 = 0; n01 = 0; n10 = 0; n11 = 0;
    for t_idx = 2:length(hits)
        prev = hits(t_idx-1);
        curr = hits(t_idx);
        if prev == 0 && curr == 0, n00 = n00 + 1; end
        if prev == 0 && curr == 1, n01 = n01 + 1; end
        if prev == 1 && curr == 0, n10 = n10 + 1; end
        if prev == 1 && curr == 1, n11 = n11 + 1; end
    end
    n0 = n00 + n01;
    n1 = n10 + n11;
    
    if (n0 + n1) == 0
        LR_ind = 0;
        p_ind  = 1;
    else
        if n0 == 0
            p01 = eps;
        else
            p01 = n01 / n0;
        end
        if n1 == 0
            p11 = eps;
        else
            p11 = n11 / n1;
        end
        pHat = (n01 + n11) / (n0 + n1);
        
        eps_val = 1e-8;
        p01 = max(min(p01,1-eps_val), eps_val);
        p11 = max(min(p11,1-eps_val), eps_val);
        pHat = max(min(pHat,1-eps_val), eps_val);
        
        lnL_H1 = n00*log1p(-p01) + n01*log(p01) + n10*log1p(-p11) + n11*log(p11);
        lnL_H0 = (n0 + n1 - (n01 + n11))*log1p(-pHat) + (n01 + n11)*log(pHat);
        
        if lnL_H1 == -Inf || lnL_H0 == -Inf
            LR_ind = Inf;
            p_ind  = 0;
        else
            LR_ind = -2*(lnL_H0 - lnL_H1);
            LR_ind = max(0, LR_ind);
            p_ind  = 1 - chi2cdf(LR_ind, 1);
        end
    end
    christoffersen_LR_ind(i) = LR_ind;
    christoffersen_p_ind(i)  = p_ind;
    
    % ----------------- Conditional Coverage ------------------------------
    LR_cc = LR_uc + LR_ind;
    p_cc  = 1 - chi2cdf(LR_cc, 2);
    condCover_LR(i) = LR_cc;
    condCover_p(i)  = p_cc;
end

backtestResults = table(finalCols', violationCounts, validDays, ...
    kupiec_LR, kupiec_p, ...
    christoffersen_LR_ind, christoffersen_p_ind, ...
    condCover_LR, condCover_p, ...
    'VariableNames', {'VaRMethod','NumViolations','NumValidDays',...
                      'Kupiec_LR','Kupiec_p',...
                      'Christoffersen_LR','Christoffersen_p',...
                      'CondCoverage_LR','CondCoverage_p'});

disp('=== Kupiec, Christoffersen (Independence), and Conditional Coverage Tests ===');
disp(backtestResults);
%%

%% Cumulative Violation Plots for All Methods (Grouped by Confidence Level)
methods90 = {'HS_90','Normal_90','MC_90','CF_90','BSHS_90','T_90'};
methods99 = {'HS_99','Normal_99','MC_99','CF_99','BSHS_99','T_99'};

figure('Color',[1 1 1]);

subplot(2,1,1);
hold on;
for i = 1:length(methods90)
    mName = methods90{i};
    mask = ~isnan(varTable.(mName));
    datesSel = varTable.Date(mask);
    hits = double(portfolioReturns(mask) < varTable.(mName)(mask));
    plot(datesSel, cumsum(hits), 'LineWidth',1.5, 'DisplayName', mName);
end
xlabel('Date');
ylabel('Cumulative Violations');
title('Cumulative Violations (90% VaR)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

subplot(2,1,2);
hold on;
for i = 1:length(methods99)
    mName = methods99{i};
    mask = ~isnan(varTable.(mName));
    datesSel = varTable.Date(mask);
    hits = double(portfolioReturns(mask) < varTable.(mName)(mask));
    plot(datesSel, cumsum(hits), 'LineWidth',1.5, 'DisplayName', mName);
end
xlabel('Date');
ylabel('Cumulative Violations');
title('Cumulative Violations (99% VaR)');
datetick('x','yyyy');
legend('Location','best','Interpreter','latex');
grid on;
hold off;

exportgraphics(gcf, 'Cumulative_Violations_AllMethods.pdf', 'ContentType', 'vector', 'Resolution',300);

%% PIT Visuals for Parametric Normal Model
T = length(portfolioReturns);
U = NaN(T,1);
for t = 1:T
    if ~isnan(muVec(t)) && ~isnan(sigmaVec(t)) && sigmaVec(t) > 0 && ~isnan(portfolioReturns(t))
        Z = (portfolioReturns(t) - muVec(t)) / sigmaVec(t);
        U(t) = normcdf(Z);
    end
end
U_valid = U(~isnan(U));

figure('Color',[1 1 1]);
histogram(U_valid, 'Normalization', 'pdf');
hold on;
xgrid = linspace(0,1,100);
plot(xgrid, ones(size(xgrid)), 'r', 'LineWidth',2);
xlabel('PIT Value');
ylabel('Density');
title('Histogram of PIT Values vs. Uniform(0,1)');
grid on;
hold off;
exportgraphics(gcf, 'PIT_Histogram.pdf', 'ContentType','vector', 'Resolution',300);

figure('Color',[1 1 1]);
qqplot(U_valid, makedist('Uniform','lower',0,'upper',1));
xlabel('Theoretical Quantiles (Uniform(0,1))');
ylabel('Empirical Quantiles');
title('PIT QQ Plot');
grid on;
exportgraphics(gcf, 'PIT_QQPlot.pdf', 'ContentType','vector', 'Resolution',300);