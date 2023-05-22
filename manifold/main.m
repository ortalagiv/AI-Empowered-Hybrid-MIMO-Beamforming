clear all;
addpath(pwd);
cd manopt;
addpath(genpath(pwd));
cd ..;

Nt = 12; Ns = 4; K = 16; Nrf = 2;
S = load('channel.mat'); % of size [Ns, Nt, K]

for i = 1:K
    [u,s,v] = svd(ch(:,:,i)); 
    f_opt(:,:,i) = v(:,1:Ns);
end

[f_rf, f_bb, count] = MO_AltMin_wideband(f_opt, Nrf);
