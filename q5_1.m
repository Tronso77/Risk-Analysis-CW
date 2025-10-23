clc; clear; close all;

% Q5(2). LOAD DATA & BASIC STAT ANALYSIS
filename = 'cleaned_adj_close_data.xlsx'; 
stockData = readtable(filename, 'PreserveVariableNames', true);

dates = stockData.Date;  
prices = stockData{:, 2:end};  
[numDays, numStocks] = size(prices);

% 1) Compute daily log-returns
logPrices = log(prices);
dailyReturns = diff(logPrices);       % (numDays-1) x numStocks
data=dailyReturns;
head(data)
[nrows ncols]=size(data)

% Bootstraping for Different Time Horizons
Nb = 10000; % Number of bootstrap simulations
T = nrows; % Number of observations
w = ones(6,1) / 6; % Equal portfolio weights
Ndays = 50; % Max time horizon (1 to 50 days)

% Initialize counter for probability estimation
probLoss = zeros(Ndays,1); % Store probability estimates for each horizon

% Bootstrap simulation loop
for i = 1:Nb
    % 1. Bootstrap the asset returns
    indices = randi(T, T, 1); % Random indices with replacement
    simRet = data(indices, :); % Bootstrapped individual asset returns

    % 2. Compute portfolio return from bootstrapped returns
    simRetPortfolio = simRet * w; % Weighted portfolio return

    % 3. Compute cumulative returns over different horizons (1 to Ndays)
    cumulativeReturns = cumsum(simRetPortfolio(1:Ndays)); % Compute cumulative sum for each horizon

    % 4. Count occurrences where cumulative return < -5% at each horizon
    probLoss = probLoss + (cumulativeReturns < -0.05);
end

% 5. Normalize to get probability estimates across all horizons
probLoss = probLoss / Nb;

% Plot the estimated probability of a 5% loss across horizons
figure;
plot(1:Ndays, probLoss, 'b', 'LineWidth', 2);
xlabel('Investment Horizon (Days)');
ylabel('Probability of Losing More Than 5%');
title('Bootstrap Estimated Probability of a 5% Loss at Different Horizons');
grid on;

