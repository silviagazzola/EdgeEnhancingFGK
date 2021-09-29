function w = AWpinv_grad2D(xx, u, v, Q, pinvDiag, sizeP, p1, p2, tflag)
 
% w = P * xx;
% where P corresponds to the SN preconditioner coming from the 2D derivative 
% operator if build as: 
% [~,u,S,v] = buildD_svd(nn);
% [Q,Diag] = Givens_eff(S,nn);
% xx=spdiags(Diag(1:(n-1),:));
% pinvDiag = sparse(1:n-1,1:n-1,1./xx,n,nn*(nn-1)*2);

n = sizeP(1);
nn = sqrt(n);

if strcmpi(tflag,'size')
    w = [n,2*nn*(nn-1)];
elseif strcmp(tflag, 'notransp')
    w1 = SNrightPrec_grad2D(xx, u, v, Q, pinvDiag, sizeP, tflag);
    w2 = p2'*w1; w2 = p1*w2;
    w = w1 - w2;
elseif strcmp(tflag, 'transp')
    xxbis = p1'*xx; xxbis = p2*xxbis; 
    xx = xx - xxbis;
    w = SNrightPrec_grad2D(xx, u, v, Q, pinvDiag, sizeP, tflag);
end
