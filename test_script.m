% Image deblurring test problem, with geometrical test image
% S. Gazzola, S. Scott, A. Spence; September 2021

%% Generate test problem
% setting the test problem size
N = 256; 
n = N^2;
rng(0) % ensures reproducibility
noiselevel = 1e-2;
ProblemOptions = PRset('trueimage','pattern1');
[A, bexact, xexact, ProbInfo] = PRblur(N,ProblemOptions);
bn = PRnoise(bexact,noiselevel);

%% setting the solver options
K = 50;   % number of inner iterations
kout = 2; % number of outer iterations
ktot = K*kout;
x0 = zeros(n,1); % initial guess
RegParam = 'discrepit'; 
qnorm = 1;
options = IRset('x_true', xexact, 'NoStop', 'on','x0',x0,'hybridvariant','Rnew','RegParam', RegParam, 'NoiseLevel', noiselevel,'qnorm', qnorm, 'IterBar', 'on');

%% running the solvers

% LSQR `with priorconditioning'
optn_lsqr = options;
optn_lsqr.RegMatrix = 'Gradient2D';
[X_lsqr, info_lsqr] = IRhybrid_lsqr(A, bn, 1:ktot, optn_lsqr);

% FLSQR utilising isotropic TV penalisation
optn_ftv = options;
optn_ftv.SparsityTrans= 'tv2D';
[X_ftv, info_ftv] = IRhybrid_flsqr(A, bn, 1:ktot, optn_ftv);

% FLSQR utilising anisotropic TV penalisation
optn_fatv = options;
optn_fatv.SparsityTrans = 'atv2D';
[X_fatv, info_fatv] = IRhybrid_flsqr(A, bn, 1:ktot, optn_fatv);

% FLSQR utilising edge-enhancing diagonal weights
optn_fdiag = options;
optn_fdiag.SparsityTrans = 'diag';
[X_fdiag, info_fdiag] = IRhybrid_flsqr(A, bn, 1:ktot, optn_fdiag);

% IRN-LSQR utilising anisotropic TV penalisation
optn_temp = options;
optn_temp.RegMatrix = 'atv2D';
optn_temp.NoStop = 'off';
X_atv = zeros(n,ktot);
info_atv.Enrm = zeros(ktot,1);
info_atv.Rnrm= zeros(ktot,1);
info_atv.RegP= zeros(ktot,1);
countit = 0;
tic
for i = 1:kout
    [X_temp, info_temp] = IRhybrid_lsqr(A, bn, 1:K, optn_temp);
    optn_temp.weight0 = X_temp(:,end); % update weights
    its = info_temp.its;
    X_atv(:,countit+1:countit+its) = X_temp;
    info_atv.Enrm(countit+1:countit+its) = info_temp.Enrm;
    info_atv.Rnrm(countit+1:countit+its) = info_temp.Rnrm;
    info_atv.RegP(countit+1:countit+its) = info_temp.RegP;
    countit = countit+its;
end
info_atv.its = countit;
clear countit its info_temp X_temp optn_temp

% IRN-LSQR utilising isotropic TV penalisation
optn_temp = options;
optn_temp.RegMatrix = 'tv2D';
optn_temp.NoStop = 'off';
X_tv = zeros(n,ktot);
info_tv.Enrm = zeros(ktot,1);
info_tv.Rnrm= zeros(ktot,1);
info_tv.RegP= zeros(ktot,1);
countit = 0;
for i = 1:kout
    [X_temp, info_temp] = IRhybrid_lsqr(A, bn, 1:K, optn_temp);
    optn_temp.weight0 = X_temp(:,end); % update weights
    its = info_temp.its;
    X_tv(:,countit+1:countit+its) = X_temp;
    info_tv.Enrm(countit+1:countit+its) = info_temp.Enrm;
    info_tv.Rnrm(countit+1:countit+its) = info_temp.Rnrm;
    info_tv.RegP(countit+1:countit+its) = info_temp.RegP;
    countit = countit+its;
end
info_tv.its = countit;
clear countit its info_temp X_temp option_temp i

% IRN-LSQR utilising edge-enhancing diagonal weights
optn_temp = options;
optn_temp.RegMatrix = 'diag';
optn_temp.NoStop = 'off';
X_diag = zeros(n,ktot);
info_diag.Enrm = zeros(ktot,1);
info_diag.Rnrm= zeros(ktot,1);
info_diag.RegP= zeros(ktot,1);
countit = 0;
for i = 1:kout
    % tic
    [X_temp, info_temp] = IRhybrid_lsqr(A, bn, 1:K, optn_temp);
    % outloop = toc; fprintf('Time for outer loop %g: %g\n' ,i, outloop)
    optn_temp.weight0 = X_temp(:,end); %update weights
    its = info_temp.its;
    X_diag(:,countit+1:countit+its) = X_temp;
    info_diag.Enrm(countit+1:countit+its) = info_temp.Enrm;
    info_diag.Rnrm(countit+1:countit+its) = info_temp.Rnrm;
    info_diag.RegP(countit+1:countit+its) = info_temp.RegP;
    countit = countit+its;
end
info_diag.its = countit;
clear countit its info_temp X_temp optn_temp

%% SSIM of reconstructions
ssim_ftv = zeros(1,ktot);
ssim_fatv = zeros(1,ktot);
ssim_fdiag = zeros(1,ktot);
ssim_lsqr = zeros(1,ktot);
ssim_tv = zeros(1,ktot);
ssim_atv = zeros(1,ktot);
ssim_diag = zeros(1,ktot);

Xex = reshape(xexact,N,N);
for it = 1:ktot
    % flexible tv
    temp = reshape(X_ftv(:,it),N,N); % temp = temp/max(max(temp));
    ssim_ftv(it) = ssim(temp,Xex); 
    % flexible atv
    temp = reshape(X_fatv(:,it),N,N); % temp = temp/max(max(temp));
    ssim_fatv(it) = ssim(temp,Xex); 
    % flexible diag
    temp = reshape(X_fdiag(:,it),N,N); % temp = temp/max(max(temp));
    ssim_fdiag(it) = ssim(temp,Xex); 
    % LSQR-L
    temp = reshape(X_lsqr(:,it),N,N); % temp = temp/max(max(temp));
    ssim_lsqr(it) = ssim(temp,Xex); 
    % IRN TV
    temp = reshape(X_tv(:,it),N,N); % temp = temp/max(max(temp));
    ssim_tv(it) = ssim(temp,Xex); 
    % IRN aTV
    temp = reshape(X_atv(:,it),N,N); % temp = temp/max(max(temp));
    ssim_atv(it) = ssim(temp,Xex); 
    % IRN diag
    temp = reshape(X_diag(:,it),N,N); % temp = temp/max(max(temp));
    ssim_diag(it) = ssim(temp,Xex); 
end

%% Total variation of reconstructions
tv_ftv = zeros(1,ktot);
tv_fatv = zeros(1,ktot);
tv_fdiag = zeros(1,ktot);
tv_lsqr = zeros(1,ktot);
tv_tv = zeros(1,ktot);
tv_atv = zeros(1,ktot);
tv_diag = zeros(1,ktot);

temp = reshape(xexact,N,N); % temp = temp/max(max(temp));
Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
truth_tv =  sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
for it = 1:ktot
    % flexible tv
    temp = reshape(X_ftv(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_ftv(it) = sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % flexible atv
    temp = reshape(X_fatv(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_fatv(it) = sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % flexible diag
    temp = reshape(X_fdiag(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_fdiag(it) = sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % LSQR-L
    temp = reshape(X_lsqr(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_lsqr(it) =  sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % IRN TV
    temp = reshape(X_tv(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_tv(it) =  sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % IRN aTV
    temp = reshape(X_atv(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_atv(it) =  sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
    % IRN diag
    temp = reshape(X_diag(:,it),N,N); % temp = temp/max(max(temp));
    Dhx = temp(:,1:N-1)-temp(:,2:N);    Dvx = temp(1:N-1,:)-temp(2:N,:);
    tv_diag(it) =  sum(sqrt( Dhx(:).^2 + Dvx(:).^2));
end


%% PLOTS

% Setting font size, line width, and marker sizes for plots:
FS = 19;
LW = 2;
MS = 6;
colour = lines(7);

% Relative error - LSQR, Flexible, IRN
figure;  
semilogy(info_lsqr.Enrm,'--','linewidth',LW);
hold on;
semilogy(info_ftv.Enrm,'-x','MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_tv.Enrm,'-<','MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_lsqr.StopReg.It,info_lsqr.StopReg.Enrm, 'd','color',colour(1,:),'MarkerSize',10,'LineWidth',LW)
semilogy(info_ftv.StopReg.It,info_ftv.StopReg.Enrm, 'd','color',colour(2,:),'MarkerSize',10,'LineWidth',LW)
xlabel('Iteration, $k$', 'Interpreter', 'latex', 'FontSize',FS)
ylabel('$\|x_{true} - x^{(k)}\|_2/\|x_{true}\|_2$','Interpreter','latex', 'FontSize',FS)
title({'\textbf{Relative Errors}'},'fontsize',FS,'Interpreter', 'latex')
legend('LSQR-L','F-TV','IRN-TV','Interpreter', 'latex')
axis([0,ktot,-inf,inf])
set(gca,'FontSize',20)

% ssim
figure;  
semilogy(ssim_lsqr,'--','linewidth',LW);
hold on;
semilogy(ssim_ftv,'-x','MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(ssim_tv,'-<','MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_lsqr.StopReg.It,ssim_lsqr(info_lsqr.StopReg.It), 'd','color',colour(1,:),'MarkerSize',10,'LineWidth',LW)
semilogy(info_ftv.StopReg.It,ssim_ftv(info_ftv.StopReg.It), 'd','color',colour(2,:),'MarkerSize',10,'LineWidth',LW)
xlabel('Iteration, $k$', 'Interpreter', 'latex', 'FontSize',FS)
ylabel('SSIM($x^{(k)}$,$x_{true}$)','Interpreter','latex', 'FontSize',FS)
title({'\textbf{SSIM}'},'fontsize',FS,'Interpreter', 'latex')
legend('LSQR-L','F-TV','IRN-TV','Interpreter', 'latex')
axis([0,ktot,-inf,inf])
set(gca,'FontSize',20)

% Relative error - Flexible and IRN
figure;  
semilogy(info_ftv.Enrm,'-x','color',colour(2,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
hold on;
semilogy(info_tv.Enrm,'-<','color',colour(3,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_fatv.Enrm,'-o','color',colour(4,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_atv.Enrm,'-','color',colour(5,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_fdiag.Enrm,'-+','color',colour(6,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_diag.Enrm,'-*','color',colour(7,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(info_ftv.StopReg.It,info_ftv.StopReg.Enrm, 'd','color',colour(2,:),'MarkerSize',10,'LineWidth',LW)
semilogy(info_fatv.StopReg.It,info_fatv.StopReg.Enrm, 'd','color',colour(4,:),'MarkerSize',10,'LineWidth',LW)
semilogy(info_fdiag.StopReg.It,info_fdiag.StopReg.Enrm, 'd','color',colour(6,:),'MarkerSize',10,'LineWidth',LW)
xlabel('Iteration, $k$', 'Interpreter', 'latex', 'FontSize',FS)
ylabel('$\|x_{true} - x^{(k)}\|_2/\|x_{true}\|_2$','Interpreter','latex', 'FontSize',FS)
title({'\textbf{Relative Errors}'},'fontsize',FS,'Interpreter', 'latex')
legend('F-TV','IRN-TV','F-aTV','IRN-aTV','F-diag','IRN-diag','Interpreter', 'latex')
axis([0,ktot,-inf,inf])
set(gca,'FontSize',20)

% ssim
figure;  
semilogy(ssim_ftv,'-x','color',colour(2,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
hold on;
semilogy(ssim_tv,'-<','color',colour(3,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(ssim_fatv,'-o','color',colour(4,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(ssim_atv,'-','color',colour(5,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(ssim_fdiag,'-+','color',colour(6,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
semilogy(ssim_diag,'-*','color',colour(7,:),'MarkerIndices',1:ktot*0.1:ktot,'MarkerSize',MS,'linewidth',LW);
ylabel('SSIM($x^{(k)}$,$x_{true}$)','Interpreter','latex', 'FontSize',FS)
title({'\textbf{SSIM}'},'fontsize',FS,'Interpreter', 'latex')
legend('F-TV','IRN-TV','F-aTV','IRN-aTV','F-diag','IRN-diag','Interpreter', 'latex')
axis([0,ktot,-inf,inf])
set(gca,'FontSize',20)
 
% Reconstructions
figure; PRshowx(xexact,ProbInfo);
title({'\textbf{Original}'},'Interpreter', 'latex' , 'FontSize',FS);
figure;
try 
    PRshowb(bn,ProbInfo);
    title({'\textbf{Observed Data}'},'Interpreter', 'latex' , 'FontSize',FS);
catch
	PRshowb(bdisp,ProbInfo);% account for in-painting bn being the wrong size
    title({'\textbf{Observed Data}'},'Interpreter', 'latex' , 'FontSize',FS);
    figure; PRshowb(Ablur*xexact,ProbInfo)
    title({'\textbf{Blurred Data}'},'Interpreter', 'latex' , 'FontSize',FS);
end

FS = 20;
figure; imagesc(reshape(info_ftv.StopReg.X, N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{F-TV}' ; ['(',num2str(info_ftv.StopReg.Enrm),' \#',num2str(info_ftv.StopReg.It),')'] },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(info_fatv.StopReg.X,N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{F-aTV}' ; ['(',num2str(info_fatv.StopReg.Enrm),' \#',num2str(info_fatv.StopReg.It),')'] },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(info_fdiag.StopReg.X, N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{F-diag}' ; ['(',num2str(info_fdiag.StopReg.Enrm),' \#',num2str(info_fdiag.StopReg.It),')'] },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(info_lsqr.StopReg.X, N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{LSQR-L}' ; ['(',num2str(info_lsqr.StopReg.Enrm),' \#',num2str(info_lsqr.StopReg.It),')'] },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(X_tv(:,info_tv.its), N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{IRN-TV}' ; ['(',num2str(info_tv.Enrm(info_tv.its)),' \#',num2str(info_tv.its),')']  },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(X_atv(:,info_atv.its), N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{IRN-aTV}' ; ['(',num2str(info_atv.Enrm(info_atv.its)),' \#',num2str(info_atv.its),')']  },'Interpreter', 'latex' , 'FontSize',FS);
figure; imagesc(reshape(X_diag(:,info_diag.its), N, N)), c = colorbar; c.FontSize = 16; axis image, axis off;
title({'\textbf{IRN-diag}' ; ['(',num2str(info_diag.Enrm(info_diag.its)),' \#',num2str(info_diag.its),')'] },'Interpreter', 'latex' , 'FontSize',FS);
