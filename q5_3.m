clc; clear; close all;

% Q5(3). LOAD DATA & BASIC STAT ANALYSIS
filename = 'cleaned_adj_close_data.xlsx'; 
stockData = readtable(filename, 'PreserveVariableNames', true);

dates = stockData.Date;  
prices = stockData{:, 2:end};  
[numDays, numStocks] = size(prices);

% 1) Compute daily log-returns
logPrices = log(prices);
dailyReturns = diff(logPrices);       % (numDays-1) x numStocks
data = dailyReturns;
[nrows, ncols] = size(data);

% Compute Portfolio Returns
w = ones(ncols, 1) / ncols;  % Equal weights
portRet = data * w;          % Portfolio returns

% Define Parameters
Nb = 10000;  % Number of bootstrap simulations
Ndays = 50;  % Maximum investment horizon (1 to 50 days)

% Initialize Probability Storage
probLossBoot = zeros(Ndays, 1);  % Bootstrapped probability
probLossTheory = zeros(Ndays, 1); % Theoretical probability

% Bootstrap Simulation
for i = 1:Nb
    % 1. Bootstrap the asset returns
    indices = randi(nrows, nrows, 1); % Random indices with replacement
    simRet = data(indices, :);        % Bootstrapped individual asset returns

    % 2. Compute portfolio return from bootstrapped returns
    simRetPortfolio = simRet * w;     % Weighted portfolio return

    % 3. Compute cumulative returns over different horizons (1 to Ndays)
    cumulativeReturns = cumsum(simRetPortfolio(1:Ndays)); % Cumulative sum for each horizon

    % 4. Count occurrences where cumulative return < -5% at each horizon
    probLossBoot = probLossBoot + (cumulativeReturns < -0.05);
end

% 5. Normalize to get probability estimates across all horizons
probLossBoot = probLossBoot / Nb;

% Compute Mean and Standard Deviation of Daily Returns
mu = mean(portRet); 
sigma = std(portRet);

% Compute Theoretical Probability Using Normal Distribution
for n = 1:Ndays
    mu_n = n * mu;          % Mean for n-day return
    sigma_n = sqrt(n) * sigma; % Standard deviation for n-day return
    probLossTheory(n) = normcdf(-0.05, mu_n, sigma_n); % Probability using Gaussian CDF
end

% Plot the Estimated Probability of a 5% Loss Across Horizons
figure;
plot(1:Ndays, probLossBoot, 'b', 'LineWidth', 2, 'DisplayName', 'Bootstrapped');
hold on;
plot(1:Ndays, probLossTheory, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical (Gaussian)');
xlabel('Investment Horizon (Days)');
ylabel('Probability of Losing More Than 5%');
title('Probability of a 5% Loss at Different Horizons');
legend('show');
grid on;


% Compute Error Metrics
MAE = mean(abs(probLossBoot - probLossTheory)); % Mean Absolute Error
MSE = mean((probLossBoot - probLossTheory).^2); % Mean Squared Error
RMSE = sqrt(MSE); % Root Mean Squared Error
correlation = corr(probLossBoot, probLossTheory); % Pearson correlation coefficient

% Display Results
fprintf('Mean Absolute Error (MAE): %.6f\n', MAE);
fprintf('Mean Squared Error (MSE): %.6f\n', MSE);
fprintf('Root Mean Squared Error (RMSE): %.6f\n', RMSE);
fprintf('Correlation Coefficient: %.6f\n', correlation);
