% function [y, cost, n_iter] = sig_manif(Fopt, FRF, FBB)
% [Nt, NRF] = size(FRF);
% 
% manifold = complexcirclefactory(Nt*NRF);
% problem.M = manifold;
% 
% % problem.cost  = @(x) norm( Fopt - reshape(x,Nt,NRF) * FBB,'fro')^2;
% % problem.egrad = @(x) -2 * kron(conj(FBB), eye(Nt)) * (Fopt(:) - kron(FBB.', eye(Nt)) * x);
% f = Fopt(:);
% A = kron(pagetranspose(FBB), eye(Nt));
% 
% problem.cost  = @(x) (f-A*x)'*(f-A*x);
% problem.egrad = @(x) -2*A'*(f-A*x); %(14)
% % checkgradient(problem);
% warning('off', 'manopt:getHessian:approx');
% 
% [x, cost, n_iter] = conjugategradient(problem,FRF(:));
% % [x,cost,info,options] = trustregions(problem, FRF(:));
% % info.iter
% y = reshape(x,Nt,NRF);
% 
% end
function [y, cost, comp] = sig_manif(Fopt, FRF, FBB)
[Nt, Nrf] = size(FRF);
K = size(FBB,3);
Ns = Nrf;

manifold = complexcirclefactory(Nt*Nrf);
problem.M = manifold;

for k = 1:K
    temp = Fopt(:,:,k);
    A = kron(FBB(:,:,k).', eye(Nt));
    
    C1(:,:,k) = temp(:)'*A;
    C2(:,k) = A'*temp(:);
    C3(:,:,k) = A'*A;
    C4(k) = norm(temp,'fro')^2;
end
B1 = sum(C1,3);
B2 = sum(C2,2);
B3 = sum(C3,3);
B4 = sum(C4);

problem.cost = @(x) -B1*x - x'*B2 + trace(B3*x*x') + B4;
problem.egrad = @(x) -2*B2 + 2*B3*x;

% checkgradient(problem);
warning('off', 'manopt:getHessian:approx');

[x,cost,info,options, n_iter] = conjugategradient(problem,FRF(:));
% [x,cost,info,options] = trustregions(problem, FRF(:));
y = reshape(x,Nt,Nrf);
% comp = n_iter*(8*K*Nrf*Ns + 6*Nt*Nrf);
end