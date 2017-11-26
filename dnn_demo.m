clear;
%mex_all;

%% Parse Test set
load 'satimage.scale.t.mat';
[NT, DIMT] = size(X);
XT = full(X');
yT = y;

%% Parse Train set
load 'satimage.scale.mat';
[N, DIM] = size(X);
X = full(X');

%% Normalize Data
% sum1 = 1./sqrt(sum(X.^2, 1));
% if abs(sum1(1) - 1) > 10^(-10)
%     X = X.*repmat(sum1, Dim, 1);
% end
% clear sum1;

%% Set Params
% X, y, XT, yT, CLASS, n_layers, stuc_layers, algorithm, lambda, batch_size,
% n_iterations, n_save_interval, step_size
Class = 6;
n_layers = 3; % No. of Hidden Layers
stuc_layers = [10, 20, 30]; % No. of Nodes in Each HL.
lambda = 1e-4; % lambda for L2 Regularizer
n_save_interval = 10;
batch_size = 100;
is_plot = true;

% SGD
algorithm = 'SGD';
step_size = 0.6;
n_iterations = 2000;
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss1, acc1] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size);
time = toc;
fprintf('Time: %f seconds \n', time);
X_SGD = [0:n_save_interval:n_iterations]';
loss1 = [X_SGD, loss1];
acc1 = [X_SGD, acc1];
clear X_SGD;

if(is_plot)
    %% Plot Loss
    la1 = min(loss1(:, 2));
    % aa2 = min(hist2(:, 2));
    % minval = min([la1]) - 2e-3;
    la = max(max([loss1(:, 2)]));
    b = 3;
    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(loss1(1:b:end,1), abs(loss1(1:b:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist2(1:b:end,1), abs(hist2(1:b:end,2) - minval),'g-.^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist3(1:b:end,1), abs(hist3(1:b:end,2) - minval),'c--+','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of Steps');
    ylabel('Loss');
    axis([0 n_iterations 0 la]);
    legend('SGD');

    %% Plot Accuracy
    aa1 = min(acc1(:, 2));
    % aa2 = min(hist2(:, 2));
    ma = min([aa1]);
    aa = max(max([acc1(:, 2)]));
    ba = 3;

    figure(102);
    set(gcf,'position',[600,100,386,269]);
    semilogy(acc1(1:ba:end,1), abs(acc1(1:ba:end,2)),'m--o','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist2(1:ba:end,1), abs(hist2(1:ba:end,2) - ma),'g-.^','linewidth',1.6,'markersize',4.5);
    % hold on,semilogy(hist3(1:ba:end,1), abs(hist3(1:ba:end,2) - ma),'c--+','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of Steps');
    ylabel('Accuracy');
    axis([0 n_iterations ma aa]);
    legend('SGD');
end
