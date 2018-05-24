clear;
mex_all;

%% Parse Test set
%load 'mnist.scale.t.mat';
load 'satimage.scale.t.mat';
[NT, DIMT] = size(X);
%X = [X zeros(size(X, 1), 1) zeros(size(X, 1), 1)]; % For stupid mnist testset
XT = full(X');
yT = y;

%% Parse Train set
%load 'mnist.scale.mat';
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
Class = 6; % No. of Class for Classification
n_layers = 2; % No. of Hidden Layers
stuc_layers = [50, 50]; % No. of Nodes in Each HL.
lambda = 1e-6; % lambda for L2 Regularizer
batch_size = 100;
n_iterations = 1000;
n_save_interval = 10;
is_plot = true;
X_axis = [0:1:n_iterations / n_save_interval]';

%% SGD
algorithm = 'SGD';
step_size = 0.5;
using_pbatch = false;
using_piterate = false;
petb_radius = 1e-6;
decay = 1e-5; % Decay step size (step_size * 1 / (1 + decay * iter))
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss1, acc1] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size, decay, using_piterate...
    , using_pbatch, petb_radius);
time = toc;
fprintf('Time: %f seconds \n', time);
loss1 = [X_axis, loss1];
acc1 = [X_axis, acc1];

%% SGD (Using perturbed-batch)
algorithm = 'SGD';
% step_size = 0.5;
using_pbatch = true;
using_piterate = false;
petb_radius = 1;
% decay = 1e-5; % Decay step size (step_size * 1 / (1 + decay * iter))
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss2, acc2] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size, decay, using_piterate...
    , using_pbatch, petb_radius);
time = toc;
fprintf('Time: %f seconds \n', time);
loss2 = [X_axis, loss2];
acc2 = [X_axis, acc2];

%% SGD (Using perturbed-iterate)
algorithm = 'SGD';
% step_size = 0.5;
using_pbatch = false;
using_piterate = true;
petb_radius = 0.1;
% decay = 1e-5; % Decay step size (step_size * 1 / (1 + decay * iter))
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss3, acc3] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, batch_size, n_iterations, n_save_interval, step_size, decay, using_piterate...
    , using_pbatch, petb_radius);
time = toc;
fprintf('Time: %f seconds \n', time);
loss3 = [X_axis, loss3];
acc3 = [X_axis, acc3];

% %% Adam
% algorithm = 'Adam';
% step_size = 0.01;
% % Adam_params: [Beta1, Beta2, epsilon]
% adam_params = [0.9, 0.999, 1e-08];
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss2, acc2] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, batch_size, n_iterations, n_save_interval, step_size, adam_params);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss2 = [X_axis, loss2];
% acc2 = [X_axis, acc2];
% 
% %% AdaGrad
% algorithm = 'AdaGrad';
% step_size = 0.5;
% epsilon = 1e-08;
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss3, acc3] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, batch_size, n_iterations, n_save_interval, step_size, epsilon);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss3 = [X_axis, loss3];
% acc3 = [X_axis, acc3];

%% SCR
% algorithm = 'SCR';
% n_iterations = 1000;
% n_save_interval = 10;
% g_batch_size = 4435;
% % SCR_params: [hv_batch_size, sub_iterations, petb_interval, L, rho, sigma]
% % petb_interval = 0 stands for Perturbe Method in Paper
% % 1/(20L) for subsolver 
% scr_params = [4435, 35, 2, 0.5, 0.001, 0.0001];
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss3, acc3] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, g_batch_size, n_iterations, n_save_interval, step_size, scr_params);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss3 = [X_axis, loss3];
% acc3 = [X_axis, acc3];
% clear X_axis;

if(is_plot)
    %% Plot Loss
    la1 = min(loss1(:, 2));
    la2 = min(loss2(:, 2));
    la3 = min(loss3(:, 2));
%     la4 = min(loss4(:, 2));
    minval = min([la1, la2, la3]) - 2e-3;
    la = max(max([loss1(:, 2), loss2(:, 2), loss3(:, 2)]));
    b = 2;
    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(loss1(1:b:end,1), abs(loss1(1:b:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss2(1:b:end,1), abs(loss2(1:b:end,2)),'g-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss3(1:b:end,1), abs(loss3(1:b:end,2)),'k--+','linewidth',1.6,'markersize',4.5);
%     hold on,semilogy(loss4(1:b:end,1), abs(loss4(1:b:end,2)),'m--^','linewidth',1.6,'markersize',4.5);
    hold off;
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.YScale = 'linear';
    xlabel('Number of Traces');
    ylabel('Loss');
    axis([0 n_iterations/n_save_interval minval la]);
    legend('SGD', 'SGD-PB', 'SGD-PT');

    %% Plot Accuracy
    aa1 = min(acc1(:, 2));
    aa2 = min(acc2(:, 2));
    aa3 = min(acc3(:, 2));
    ma = min([aa1, aa2, aa3]);
    aa = max(max([acc1(:, 2), acc2(:, 2), acc3(:, 2)]));
    ba = 2;

    figure(102);
    set(gcf,'position',[600,100,386,269]);
    semilogy(acc1(1:ba:end,1), abs(acc1(1:ba:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(acc2(1:ba:end,1), abs(acc2(1:ba:end,2)),'c-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(acc3(1:ba:end,1), abs(acc3(1:ba:end,2)),'k--+','linewidth',1.2,'markersize',4.5);
%     hold on,semilogy(acc4(1:ba:end,1), abs(acc4(1:ba:end,2)),'m--^','linewidth',1.2,'markersize',4.5);
    hold off;
    xlabel('Number of Steps');
    ylabel('Accuracy');
    axis([0 n_iterations/n_save_interval 0.2 aa]);
    legend('SGD', 'SGD-PB', 'SGD-PT');
end
