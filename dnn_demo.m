clear;
mex_all;

%% Parse Test set
load 'mnist.scale.t.mat';
[NT, DIMT] = size(X);
X = [X zeros(size(X, 1), 1) zeros(size(X, 1), 1)]; % For stupid mnist testset
XT = full(X');
yT = y;

%% Parse Train set
load 'mnist.scale.mat';
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
Class = 10;
n_layers = 2; % No. of Hidden Layers
stuc_layers = [10, 20]; % No. of Nodes in Each HL.
lambda = 1e-6; % lambda for L2 Regularizer
batch_size = 20;
n_iterations = 40000;
n_save_interval = 2000;
is_plot = true;

%% SGD
algorithm = 'SGD';
step_size = 0.1;
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss1, acc1] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size);
time = toc;
fprintf('Time: %f seconds \n', time);
X_axis = [0:1:n_iterations / n_save_interval]';
loss1 = [X_axis, loss1];
acc1 = [X_axis, acc1];

%% Adam
algorithm = 'Adam';
step_size = 0.1;
% Adam_params: [Beta1, Beta2, epsilon]
adam_params = [0.9, 0.999, 1e-8];
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss2, acc2] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size, adam_params);
time = toc;
fprintf('Time: %f seconds \n', time);
loss2 = [X_axis, loss2];
acc2 = [X_axis, acc2];

%% SCR
algorithm = 'SCR';
n_iterations = 40;
n_save_interval = 2;
g_batch_size = 200;
% SCR_params: [hv_batch_size, sub_iterations, petb_interval, eta, rho, sigma]
scr_params = [200, 99, 0, 0.03, 3, 0.001];
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss3, acc3] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, g_batch_size, n_iterations, n_save_interval, step_size, scr_params);
time = toc;
fprintf('Time: %f seconds \n', time);
loss3 = [X_axis, loss3];
acc3 = [X_axis, acc3];
clear X_axis;

if(is_plot)
    %% Plot Loss
%     la1 = min(loss1(:, 2));
%     la2 = min(loss2(:, 2));
%     la3 = min(loss3(:, 2));
    % minval = min([la1]) - 2e-3;
    la = max(max([loss1(:, 2)]));
    b = 2;
    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(loss1(1:b:end,1), abs(loss1(1:b:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss2(1:b:end,1), abs(loss2(1:b:end,2)),'g-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss3(1:b:end,1), abs(loss3(1:b:end,2)),'k-.^','linewidth',1.6,'markersize',4.5);
    hold off;
    xlabel('Number of Steps');
    ylabel('Loss');
    axis([0 n_iterations/n_save_interval 0 la]);
    legend('SGD', 'Adam', 'SCR');

    %% Plot Accuracy
%     aa1 = min(acc1(:, 2));
%     aa2 = min(acc2(:, 2));
%     aa3 = min(acc3(:, 2));
%     ma = min([aa1, aa2]);
    aa = max(max([acc1(:, 2), acc2(:, 2)]));
    ba = 2;

    figure(102);
    set(gcf,'position',[600,100,386,269]);
    semilogy(acc1(1:ba:end,1), abs(acc1(1:ba:end,2)),'m--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(acc2(1:ba:end,1), abs(acc2(1:ba:end,2)),'c-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(acc3(1:ba:end,1), abs(acc3(1:ba:end,2)),'k--+','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of Steps');
    ylabel('Accuracy');
    axis([0 n_iterations/n_save_interval 0.5 aa]);
    legend('SGD', 'Adam', 'SCR');
end
