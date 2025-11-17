#Solving 1D Heat Equation using Data-Free Physics-Informed Kolmogorov-Arnold Network (DF-PIKAN)
#Author: Mohammad E. Heravifard
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
# 1. Small univariate network φ(x): 1D -> 1D
# ============================================================

class UnaryNet(nn.Module):
    """
    A tiny 1D neural network: R -> R
    Used to approximate φ_ij in the Kolmogorov–Arnold style.
    """
    def __init__(self, hidden_dim=8):
        super(UnaryNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (N,1)
        return self.net(x)


# ============================================================
# 2. KAN layer: y_j = sum_i φ_ij(x_i)
# ============================================================

class KANLayer(nn.Module):
    """
    Kolmogorov–Arnold inspired layer:
    Input:  x in R^{d_in}
    Output: y in R^{d_out}

    y_j = sum_{i=1}^{d_in} φ_ij( x_i )

    where each φ_ij is a small UnaryNet.
    """
    def __init__(self, d_in, d_out, hidden_dim_unary=8):
        super(KANLayer, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        # Create φ_ij nets as a d_out x d_in table of UnaryNets
        self.phi = nn.ModuleList([
            nn.ModuleList([
                UnaryNet(hidden_dim_unary) for _ in range(d_in)
            ]) for _ in range(d_out)
        ])

    def forward(self, x):
        """
        x: (N, d_in)
        returns: (N, d_out)
        """
        N = x.shape[0]
        y = x.new_zeros((N, self.d_out))

        # For each output dimension j
        for j in range(self.d_out):
            # Sum over input dims i of φ_ij(x_i)
            s = 0.0
            for i in range(self.d_in):
                # Take the i-th coordinate: shape (N,) -> (N,1)
                xi = x[:, i:i+1]
                s = s + self.phi[j][i](xi).squeeze(-1)  # (N,)
            y[:, j] = s

        return y


# ============================================================
# 3. KAN-based PINN model
# ============================================================

class KAN_PINN(nn.Module):
    """
    KAN-based Physics-Informed Network:
    Input: (x,t) -> R^2
    Output: u(x,t) -> R^1
    """
    def __init__(self, hidden_dim=8, hidden_layers=2, unary_hidden_dim=8):
        super(KAN_PINN, self).__init__()

        layers = []
        d_in = 2  # x, t
        d_hidden = hidden_dim

        # First KAN layer
        layers.append(KANLayer(d_in, d_hidden, hidden_dim_unary=unary_hidden_dim))

        # Middle KAN layers
        for _ in range(hidden_layers - 1):
            layers.append(KANLayer(d_hidden, d_hidden, hidden_dim_unary=unary_hidden_dim))

        # Final linear layer to output u
        self.kan_layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(d_hidden, 1)

    def forward(self, x, t):
        """
        x, t: (N,1)
        returns: u(x,t): (N,1)
        """
        inp = torch.cat([x, t], dim=1)  # (N,2)
        h = inp
        for layer in self.kan_layers:
            h = layer(h)
            h = torch.tanh(h)
        u = self.out_layer(h)
        return u


# ============================================================
# 4. Utility: gradients for PDE
# ============================================================

def gradients(u, x, order=1):
    """
    Compute du/dx or higher-order derivatives using autograd.
    u: (N,1)
    x: (N,1) with requires_grad=True
    order: 1 or 2
    """
    if order == 1:
        grads = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return grads
    elif order == 2:
        u_x = gradients(u, x, order=1)
        u_xx = gradients(u_x, x, order=1)
        return u_xx
    else:
        raise ValueError("order must be 1 or 2.")


# ============================================================
# 5. Create training data (collocation)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# Domain: x in [0,1], t in [0,1]
N_f = 8000    # collocation points for PDE residual
N_b = 200     # boundary points
N_i = 200     # initial condition points

# Interior points (for PDE residual)
x_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1)

# Boundary x=0 and x=1
t_b = torch.rand(N_b, 1)
x_b0 = torch.zeros(N_b, 1)
x_b1 = torch.ones(N_b, 1)

# Initial line t=0
x_i = torch.rand(N_i, 1)
t_i = torch.zeros(N_i, 1)

# Exact initial condition: u(x,0) = sin(pi x)
u_i_true = torch.sin(torch.pi * x_i)

x_f = x_f.to(device).requires_grad_(True)
t_f = t_f.to(device).requires_grad_(True)
x_b0 = x_b0.to(device).requires_grad_(True)
x_b1 = x_b1.to(device).requires_grad_(True)
t_b = t_b.to(device).requires_grad_(True)
x_i = x_i.to(device).requires_grad_(True)
t_i = t_i.to(device).requires_grad_(True)
u_i_true = u_i_true.to(device)


# ============================================================
# 6. Instantiate model and optimizer
# ============================================================

model = KAN_PINN(hidden_dim=8, hidden_layers=2, unary_hidden_dim=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mse_loss = nn.MSELoss()


# ============================================================
# 7. PDE residual for heat equation u_t = u_xx
# ============================================================

def pde_residual(x, t):
    """
    Heat equation residual: u_t - u_xx = 0
    """
    u = model(x, t)          # (N,1)
    u_t = gradients(u, t, order=1)
    u_xx = gradients(u, x, order=2)
    return u_t - u_xx        # should be ~0


# ============================================================
# 8. Training loop
# ============================================================

num_epochs = 4000  # adjust as needed

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 1) PDE residual loss
    r_f = pde_residual(x_f, t_f)
    loss_f = mse_loss(r_f, torch.zeros_like(r_f))

    # 2) Boundary loss: u(0,t) = 0, u(1,t) = 0
    u_b0 = model(x_b0, t_b)
    u_b1 = model(x_b1, t_b)
    loss_b = mse_loss(u_b0, torch.zeros_like(u_b0)) + \
             mse_loss(u_b1, torch.zeros_like(u_b1))

    # 3) Initial condition: u(x,0) = sin(pi x)
    u_i = model(x_i, t_i)
    loss_i = mse_loss(u_i, u_i_true)

    loss = loss_f + loss_b + loss_i

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: total_loss={loss.item():.4e}, "
              f"loss_f={loss_f.item():.4e}, loss_b={loss_b.item():.4e}, loss_i={loss_i.item():.4e}")


# ============================================================
# 9. Visualization
#     - 1D slices u(x, t0) vs exact
#     - 2D heatmaps of solution + error
# ============================================================

model.eval()

# Exact analytical solution for comparison:
# u(x,t) = exp(-pi^2 t) * sin(pi x)
def exact_solution(x, t):
    """
    x, t: torch tensors on any device, shape (N,1)
    returns u(x,t) of same shape.
    """
    return torch.exp(- (torch.pi**2) * t) * torch.sin(torch.pi * x)


# 9.1 Plot 1D slices at different times
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 200).view(-1, 1).to(device)

    times_to_plot = [0.0, 0.25, 0.5, 0.75]
    plt.figure(figsize=(10, 6))

    for t0 in times_to_plot:
        t_plot = torch.full_like(x_plot, t0)
        u_pred = model(x_plot, t_plot)
        u_ex = exact_solution(x_plot, t_plot)

        x_cpu = x_plot.cpu().numpy().flatten()
        u_pred_cpu = u_pred.cpu().numpy().flatten()
        u_ex_cpu = u_ex.cpu().numpy().flatten()

        plt.plot(x_cpu, u_ex_cpu, 'k--', linewidth=1.5, label=f"Exact t={t0}" if t0 == times_to_plot[0] else None)
        plt.plot(x_cpu, u_pred_cpu, linewidth=1.5, label=f"KAN-PINN t={t0}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("KAN-PINN vs exact solution for 1D heat equation")
    plt.legend()
    plt.grid(True)
    plt.show()


# 9.2 2D heatmaps: solution and error on (x,t) grid
with torch.no_grad():
    Nx = 100
    Nt = 100
    x_lin = torch.linspace(0, 1, Nx).to(device)
    t_lin = torch.linspace(0, 1, Nt).to(device)
    X, T = torch.meshgrid(x_lin, t_lin, indexing='ij')  # X, T: (Nx, Nt)

    x_grid = X.reshape(-1, 1)
    t_grid = T.reshape(-1, 1)

    u_pred_grid = model(x_grid, t_grid).reshape(Nx, Nt)
    u_ex_grid = exact_solution(x_grid, t_grid).reshape(Nx, Nt)
    err_grid = (u_pred_grid - u_ex_grid).abs()

    X_cpu = X.cpu().numpy()
    T_cpu = T.cpu().numpy()
    u_pred_cpu = u_pred_grid.cpu().numpy()
    u_ex_cpu = u_ex_grid.cpu().numpy()
    err_cpu = err_grid.cpu().numpy()

    # Predicted solution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(u_pred_cpu.T, extent=[0,1,0,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("KAN-PINN u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")

    # Exact solution
    plt.subplot(1, 3, 2)
    plt.imshow(u_ex_cpu.T, extent=[0,1,0,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("Exact u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")

    # Absolute error
    plt.subplot(1, 3, 3)
    plt.imshow(err_cpu.T, extent=[0,1,0,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title("|Error|")
    plt.xlabel("x")
    plt.ylabel("t")

    plt.tight_layout()
    plt.show()
