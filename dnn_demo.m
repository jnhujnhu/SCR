clear;
mex_all;
load 'mnist.scale.mat';

%% Parse Data
[N, Dim] = size(X);
X = full(X');

%% Normalize Data
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, Dim, 1);
% end
% clear sum1;

%% Set Params
% X, Y, CLASS, n_layers, stuc_layers, algorithm, lambda, batch_size,
% n_iteraions, n_save_interval, step_size
Class = 10;
n_layers = 2; % No. of Hidden Layers
stuc_layers = [20, 30]; % No. of Nodes in Each HL.
lambda = 1e-4; % lambda for L2 Regularizer
n_save_interval = 1;
batch_size = fix(N / n_save_interval);
passes = 50;
X_algos = [0:1:passes]';

is_plot = true;

% SGD
algorithm = 'SGD';
step_size = 0.5;
n_iteraions = passes * n_save_interval;
fprintf('Algorithm: %s\n', algorithm);
tic;
hist1 = interface(X, y, Class, n_layers, stuc_layers, algorithm, lambda...
    , batch_size, n_iteraions, n_save_interval, step_size);
time = toc;
fprintf('Time: %f seconds \n', time);
hist1 = [X_algos, hist1];

clear X_algos;
%% Plot
if(is_plot)
    aa1 = min(hist1(:, 2));
    % aa2 = min(hist2(:, 2));
    % minval = min([aa1]) - 2e-3;
    aa = max(max([hist1(:, 2)]));
    b = 1;

    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(hist1(1:b:end,1), abs(hist1(1:b:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'c--+','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of effective passes');
    ylabel('Loss');
    axis([0 passes 0  aa]);
    legend('SGD');
end
