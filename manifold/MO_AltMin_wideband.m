function [FRF,FBB, count] = MO_AltMin_wideband(Fopt)
[Nt, Ns, K] = size(Fopt);
Nrf = 2;
FBB = zeros(Nrf, Ns, K);
count = 0;
if Nt > Nrf % HBF
    y = [];
    FRF = exp(1i*unifrnd(0,2*pi,Nt,Nrf));
    while (isempty(y) || abs(y(1)-y(2)) > 1e-3)
        count = count + 1;
        y = [0,0];
        for k = 1:K
            FBB(:,:,k) = pinv(FRF) * Fopt(:,:,k);
            y(1) = y(1) + norm(Fopt(:,:,k) - FRF * FBB(:,:,k),'fro')^2;
        end
        [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
    end
else
    FRF = eye(Nrf);
    FBB = Fopt;
end

for k = 1:K
    FBB(:,:,k) = sqrt(Ns) * FBB(:,:,k) / norm(FRF * FBB(:,:,k),'fro');
    if abs(norm(FRF * FBB(:,:,k),'fro')^2 - Ns) > 1e-4
        error('check power constraint !!!!!!!!!!!!')
    end
end

end