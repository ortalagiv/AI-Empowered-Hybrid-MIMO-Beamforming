# ---- Learn to Rapidly and Robustly Optimize for Hybrid Precoding ----
# --- Ortal Agiv (agivo@post.bgu.ac.il) and Nir Shlezinger (nirshl@bgu.ac.il) ---

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class ProjGA(nn.Module):

    def __init__(self, hyp):
        super().__init__()
        self.hyp = nn.Parameter(hyp)  # parameters = (mu_a, mu_(d,1), ..., mu_(d,B))

    def forward(self, h, n, l1, b, num_of_iter):
        # ------- Projection Gradient Ascent execution ---------

        # --- inputs:
        # h - channel realization
        # n - num of users
        # l1 - num of RF chains
        # b - num of frequency bands
        # num_of_iter - num of iters of the PGA algorithm

        # ---- initializing variables
        # svd for H_avg --> H = u*smat*vh
        _, _, vh = np.linalg.svd(sum(h) / b, full_matrices=True)
        vh = torch.from_numpy(vh)
        # initializing Wa as vh
        wa = vh[:, :, :l1]
        wa = torch.cat(((wa[None, :, :, :],) * b), 0)
        # randomizing Wd,b
        wd = torch.randn(b, len(h[0]), l1, n)
        # projecting Wd,b onto the constraint
        wd = (torch.sqrt(n * b / (sum(torch.linalg.matrix_norm(wa @ wd, ord='fro')**2)))).reshape(len(h[0]), 1, 1) * wd

        # defining an array which holds the values of the rate of each iteration
        obj = torch.zeros(num_of_iter, len(h[0]))

        # update equations
        for x in range(num_of_iter):
            # ---------- Wa ---------------
            # gradient ascent
            wa_t = wa + self.hyp[x][0] * self.grad_wa(h, wa, wd, n, b)
            # projection
            wa = (torch.sqrt(n * b / (sum(torch.linalg.matrix_norm(wa_t @ wd, ord='fro') ** 2)))).reshape(len(h[0]), 1,
                                                                                                          1) * wa_t

            # ---------- Wd,b ---------------
            wd_t = wd.clone().detach()
            for i in range(b):
                # gradient ascent
                wd_t[i] = wd[i].clone().detach() + self.hyp[x][i + 1] * self.grad_wd(h[i], wa[0], wd[i].clone().detach(), n, b)
                # projection
                wd = (torch.sqrt(n * b / (sum(torch.linalg.matrix_norm(wa @ wd_t, ord='fro') ** 2)))).reshape(len(h[0]),
                                                                                                              1,
                                                                                                              1) * wd_t

            # update the rate
            obj[x] = self.objec(h, wa, wd, n, b)

        return torch.transpose(obj, 0, 1), wa, wd

    def objec(self, h, wa, wd, n, b):
        # calculates the rate for a given channel (h) and precoders (wa, wd)
        return sum(torch.log((torch.eye(n).reshape((1, 1, n, n)) +
                       h @ wa @ wd @ torch.transpose(wd, 2, 3).conj() @
                       torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()).det())) / b

    def grad_wa(self, h, wa, wd, n, b):
        # calculates the gradient with respect to wa for a given channel (h) and precoders (wa, wd)
        f2 = sum(torch.transpose(h, 2, 3) @ torch.transpose(torch.linalg.inv(torch.eye(n).reshape((1, 1, n, n))
                                                                             + h @ wa @ wd @
                                                                             torch.transpose(wd, 2, 3).conj() @
                                                                             torch.transpose(wa, 2, 3).conj() @
                                                                             torch.transpose(h, 2, 3).conj()), 2, 3)
                                                                             @ h.conj() @ wa.conj() @ wd.conj() @
                                                                             torch.transpose(wd, 2, 3)) / b
        return torch.cat(((f2[None, :, :, :],) * b), 0)

    def grad_wd(self, h, wa, wd, n, b):
        # calculates the gradient with respect to wd,b for a given channel (h) and precoders (wa, wd)
        return (torch.transpose(wa, 1, 2) @ torch.transpose(h, 1, 2) @
                torch.transpose(torch.linalg.inv(torch.eye(n).reshape((1, n, n)).repeat(len(h), 1, 1) +
                h @ wa @ wd @ torch.transpose(wd, 1, 2).conj() @
                torch.transpose(wa, 1, 2).conj() @ torch.transpose(h, 1, 2).conj()), 1, 2) @
                h.conj() @ wa.conj() @ wd.conj()) / b


def sum_loss(wa, wd, h, n, b, batch_size):
    a1 = torch.transpose(wa, 2, 3).conj() @ torch.transpose(h, 2, 3).conj()
    a2 = torch.transpose(wd, 2, 3).conj() @ a1
    a3 = h @ wa @ wd @ a2
    g = torch.eye(n).reshape((1, 1, n, n)) + a3  # g = Ik + H*Wa*Wd*Wd^(H)*Wa^(H)*H^(H)
    s = torch.log(g.det())  # s = log(det(g))
    ra = sum(s) / b
    loss = sum(ra) / batch_size
    return -loss


# ---- MAIN ----
# ---- the systems features
B = 8      # Num of frequency bands
N = 6      # Num of users
L = 10      # RF chains
M = 12     # Tx antennas

# ---- generating data set
train_size = 1000
valid_size = 100
test_size = 100

# train data set
H_train = torch.randn(B, train_size, N, M)
# validation data set
H_valid = torch.randn(B, valid_size, N, M)
# test data set
H_test = torch.randn(B, test_size, N, M)

# ---- Classical PGA ----
# parameters defining
num_of_iter_pga = 30
mu = torch.tensor([[50 * 1e-2] * (B+1)] * num_of_iter_pga, requires_grad=True)

# Object defining
classical_model = ProjGA(mu)

# executing classical PGA on the test set
sum_rate_class, __, __ = classical_model.forward(H_test, N, L, B, num_of_iter_pga)
# ploting the results
plt.figure()
y = [r.detach().numpy() for r in (sum(sum_rate_class)/test_size)]
x = np.array(list(range(num_of_iter_pga))) +1
plt.plot(x, y, 'o')
plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the Classical PGA')
plt.xlabel('Number of Iteration')
plt.ylabel('Achievable Rate')
plt.grid()
plt.show()

# ---- Unfolded PGA ----
# parameters defining
num_of_iter_pga_unf = 5
mu_unf = torch.tensor([[50 * 1e-2] * (B+1)] * num_of_iter_pga_unf, requires_grad=True)

# Object defining
unfolded_model = ProjGA(mu_unf)

# training procedure
optimizer = torch.optim.Adam(unfolded_model.parameters(), lr=0.4)

epochs = 110
batch_size = 100  # batch size
train_losses, valid_losses = [], []

for i in range(epochs):
    H_shuffeld = torch.transpose(H_train, 0, 1)[np.random.permutation(len(H_train[1]))]
    for b in range(0, len(H_train), batch_size):
        H = torch.transpose(H_shuffeld[b:b+batch_size], 0, 1)
        __, wa, wd = unfolded_model.forward(H, N, L, B, num_of_iter_pga_unf)
        loss = sum_loss(wa, wd, H, N, B, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # train loss
    __, wa, wd = unfolded_model.forward(H_train, N, L, B, num_of_iter_pga_unf)
    train_losses.append(sum_loss(wa, wd, H_train, N, B, train_size))

    # validation loss
    __, wa, wd = unfolded_model.forward(H_valid, N, L, B, num_of_iter_pga_unf)
    valid_losses.append(sum_loss(wa, wd, H_valid, N, B, valid_size))

# plotting learning curve
y_t = [r.detach().numpy() for r in train_losses]
x_t = np.array(list(range(len(train_losses))))
y_v = [r.detach().numpy() for r in valid_losses]
x_v = np.array(list(range(len(valid_losses))))
plt.figure()
plt.plot(x_t, y_t, 'o', label='Train')
plt.plot(x_v, y_v, '*', label='Valid')
plt.grid()
plt.title(f'Loss Curve, Num Epochs = {epochs}, Batch Size = {batch_size} \n Num of Iterations of PGA = {num_of_iter_pga_unf}')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()

# executing unfolded PGA on the test set
sum_rate_unf, __, __ = unfolded_model.forward(H_test, N, L, B, num_of_iter_pga_unf)
# ploting the results
plt.figure()
y = [r.detach().numpy() for r in (sum(sum_rate_unf)/test_size)]
x = np.array(list(range(num_of_iter_pga_unf))) +1
plt.plot(x, y, 'o')
plt.title(f'The Average Achievable Sum-Rate of the Test Set \n in Each Iteration of the unfolded PGA')
plt.xlabel('Number of Iteration')
plt.ylabel('Achievable Rate')
plt.grid()
plt.show()
