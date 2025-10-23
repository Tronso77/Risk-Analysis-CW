function shrinkage = optimalShrinkage(Y,Ymkt,T,N,sample,target)

covmkt=(Y'*Ymkt)./T; % covariance of original variables with common factor
varmkt=(Ymkt'*Ymkt)./T;

% estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
Y2 = Y.^2;
sample2 = (Y2'*Y2)./T; % sample covariance matrix of squared returns
piMat = sample2-sample.^2;
pihat = sum(sum(piMat));

% estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
gammahat = norm(sample-target,'fro')^2;

% diagonal part of the parameter that we call rho 
rho_diag = sum(diag(piMat));

% off-diagonal part of the parameter that we call rho 
temp = Y .* Ymkt(:,ones(1,N));
v1 = (1/T) * Y2' * temp-covmkt(:,ones(1,N)) .* sample;
roff1 = sum(sum(v1.*covmkt(:,ones(1,N))'))/varmkt...
   -sum(diag(v1).*covmkt)/varmkt;
v3 = (1/T)*temp'*temp-varmkt*sample;
roff3 = sum(sum(v3.*(covmkt*covmkt')))/varmkt^2 ...
   -sum(diag(v3).*covmkt.^2)/varmkt^2;
rho_off = 2*roff1-roff3;

% compute shrinkage intensity
rhohat = rho_diag + rho_off;
kappahat = (pihat-rhohat)/gammahat;
shrinkage = max(0,min(1,kappahat/T));

end

