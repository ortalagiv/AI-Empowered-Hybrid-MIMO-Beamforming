%function [Rate_Weiyu]=hybrid_weiyu_P2P(Nt,Nrf_t,Nr,Nrf_r,Ns,Pow,noisevar,H,K)
clear all; clc;

Nt = 12; Nrf_t = 2; Nr = 4; Nrf_r = 2; Ns = 4; Pow = Ns; K = 16;
H1 = load('ch5_te.mat'); H = H1.H;

for snr = linspace(-5,10,16)
    noisevar = 10^(-0.1*snr);
    Rate_100 = 0;
    avg_iter = 0;
    for n = 1:100
        Fopt = zeros(Nt, Ns, K);
        Frf0 = zeros(Nt,Nrf_t);
        for i = 1:K
            [Us,Ss,Vs] = svd(squeeze(H(n,:,:,i)));
            Fopt(:,:,i) = Vs(:,1:Ns);
            Frf0 = Frf0 + (1/K)*Vs(:,1:Nrf_t);
        end
        BB_scheme = 1;
        [FRF, FBB, count] = AO(Pow,noisevar,Nrf_t,squeeze(H(n,:,:,:)),Fopt,BB_scheme,Frf0);
        avg_iter = avg_iter + count/100;
        Rate = 0;
        for i = 1:K
            H_r = squeeze(H(n,:,:,i));
            FBB_r = squeeze(FBB(:,:,i));
            Rate_k = log2(det(eye(Nr)+(1/(Ns*noisevar))*H_r*FRF*FBB_r*FBB_r'*FRF'*H_r'));
            Rate = Rate + abs(Rate_k)/K;
        end
        Rate_100 = Rate_100 + Rate/100;    
    end
    disp(snr);
    iter(snr+6) = avg_iter;
    ao_rates(snr+6) = Rate_100;
end