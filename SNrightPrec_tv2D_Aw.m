function w = SNrightPrec_tv2D_Aw(xx, invW_2d, u, v, Q, pinvDiag, sizeP, p1, p2, tflag)
 
% w = P * xx;
% where P corresponds to the SN preconditioner coming from the 2D derivative 
% operator if build as: 
% [~,u,S,v] = buildD_svd(nn);
%   - u and v are matrices from the SVD of D_1d (nn-by-nn)
%   - S = [kron(s,I);kron(I,s)], s the singular values of D_1d
% [Q,Diag] = Givens_eff(S,nn);
% xx=spdiags(Diag(1:(n-1),:));
% pinvDiag = sparse(1:n-1,1:n-1,1./xx,n,nn*(nn-1)*2);


n = sizeP(1);   % dimension of vector x, in Ax=b
nn = sqrt(n);   % dimesion of (square) array X in AX=B


if strcmpi(tflag,'size')
    w = [n,2*nn*(nn-1)];
elseif strcmp(tflag, 'notransp')
% Use approximation     pinv(Lbar) = pinv(D_2d)*inv(W_2d)

%%%%%% ACTUAL PINV %%%%%%%%%%%%
% d=ones(nn, 1);
% D_2d = spdiags([d -d], 0:1 , nn-1, nn);
% D_2d=[kron(D_2d,speye(nn));kron(speye(nn),D_2d)];
% Lpinv = pinv(full(W_2d*D_2d));
% w = A_times_vec(Lpinv,xx);
% return
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     % Calculate preliminary value y = inv(W_2d)*x 
    y = A_times_vec(invW_2d,xx);

     % Calculate w = pinv(D_2d)*y
    w = AWpinv_grad2D(y, u, v, Q, pinvDiag, sizeP, p1, p2, tflag);
    
    
elseif strcmp(tflag, 'transp')
% Use approximation     pinv(Lbar)' = inv(W_2D)'*pinv(D_2d)'

%%%%%% ACTUAL PINV %%%%%%%%%%%%
% d=ones(nn, 1);
% D_2d = spdiags([d -d], 0:1 , nn-1, nn);
% D_2d=[kron(D_2d,speye(nn));kron(speye(nn),D_2d)];
% Lpinv = pinv(full(W_2d*D_2d));
% w = A_times_vec(Lpinv',xx);
% return
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Calculate preliminary value y = pinv(D_2d)'*xx    
    y = AWpinv_grad2D(xx, u, v, Q, pinvDiag, sizeP, p1, p2, tflag);
    
    % Calculate w = inv(W_2d)'*y
    w = A_times_vec(invW_2d,y);
    
elseif strcmp(tflag, 'weights')
    % Returns the weights from inv_W2d
    w = diag(invW_2d);
    
end
