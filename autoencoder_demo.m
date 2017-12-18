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
Class = 20; % No. of Code for Autoencoder
n_layers = 1; % No. of Hidden Layers for Encoder(Except Code Layer)
stuc_layers = [200]; % No. of Nodes in Each HL for Encoder(Except Code Layer)
lambda = 1e-6; % lambda for L2 Regularizer
batch_size = 100;
n_iterations = 6000;
n_save_interval = 200;
is_plot = true;
X_axis = [0:1:n_iterations / n_save_interval]';

% %% SGD
% algorithm = 'SGD';
step_size = 0.5;
% decay = 0; % Decay step size (step_size * 1 / (1 + decay * iter))
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss1] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, batch_size, n_iterations, n_save_interval, step_size, decay);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss1 = [X_axis, loss1];
% 
% %% Adam
% algorithm = 'Adam';
% step_size = 0.01;
% % Adam_params: [Beta1, Beta2, epsilon]
% adam_params = [0.9, 0.999, 1e-08];
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss2] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, batch_size, n_iterations, n_save_interval, step_size, adam_params);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss2 = [X_axis, loss2];
% 
% %% AdaGrad
% algorithm = 'AdaGrad';
% step_size = 0.5;
% epsilon = 1e-08;
% fprintf('Algorithm: %s\n', algorithm);
% tic;
% [loss4] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
%     , lambda, batch_size, n_iterations, n_save_interval, step_size, epsilon);
% time = toc;
% fprintf('Time: %f seconds \n', time);
% loss4 = [X_axis, loss4];

%% SCR
algorithm = 'SCR';
n_iterations = 1300;
n_save_interval = 100;
g_batch_size = 100;
% SCR_params: [hv_batch_size, sub_iterations, petb_interval, L, rho, sigma]
% petb_interval = 0 stands for Perturbe Method in Paper
% 1/(20L) for subsolver 
scr_params = [10, 20, 0, 1.0, 0.01, 0.0001];
fprintf('Algorithm: %s\n', algorithm);
tic;
[loss3] = interface(X, y, XT, yT, Class, n_layers, stuc_layers, algorithm ...
    , lambda, g_batch_size, n_iterations, n_save_interval, step_size, scr_params);
time = toc;
fprintf('Time: %f seconds \n', time);
loss3 = [X_axis, loss3];
clear X_axis;

if(is_plot)
    %% Plot Loss
    la1 = min(loss1(:, 2));
    la2 = min(loss2(:, 2));
    la3 = min(loss3(:, 2));
    la4 = min(loss4(:, 2));
    minval = min([la1, la2, la3, la4]) - 2e-3;
    la = max(max([loss1(:, 2)]));
    b = 1;
    figure(101);
    set(gcf,'position',[200,100,386,269]);
    semilogy(loss1(1:b:end,1), abs(loss1(1:b:end,2)),'b--o','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss2(1:b:end,1), abs(loss2(1:b:end,2)),'g-.^','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss3(1:b:end,1), abs(loss3(1:b:end,2)),'k--+','linewidth',1.6,'markersize',4.5);
    hold on,semilogy(loss4(1:b:end,1), abs(loss4(1:b:end,2)),'m--^','linewidth',1.6,'markersize',4.5);
    hold off;
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.YScale = 'linear';
    xlabel('Number of Traces');
    ylabel('Loss');
    axis([0 n_iterations/n_save_interval minval 0.1]);
    legend('SGD', 'Adam', 'SCR', 'AdaGrad');
end
