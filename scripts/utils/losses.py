import torch
from torch import ones, zeros, linspace, exp, clone, sum

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DMPIntegrationMSE:
    def __init__(self, train_param):
        self.train_param = train_param
        self.model_param = self.train_param.model_param
        self.dmp_param = self.model_param.dmp_param
        self.dof = self.dmp_param.dof
        self.dt = self.dmp_param.dt
        self.n_bf = self.dmp_param.n_bf
        self.ay = self.dmp_param.ay
        self.by = self.dmp_param.by
        self.cs_runtime = self.dmp_param.cs_runtime
        self.cs_ax = self.dmp_param.cs_ax
        self.tau = self.dmp_param.tau
        self.scale = self.dmp_param.scale
        self.timesteps = self.dmp_param.timesteps

    def __call__(self, dmp_params, dmp_traj):
        self.integrateDMP(dmp_params)
        mse = torch.nn.MSELoss()
        loss = mse(self.y_track, dmp_traj)
        return loss

    def integrateDMP(self, dmp_params):
        self.splitDMPparameters(dmp_params)
        self.initializeDMP()
        for t in range(self.timesteps):
            self.step(t)

    def splitDMPparameters(self, dmp_params):
        self.batch_s = dmp_params.shape[0]
        if self.scale != None: dmp_params = self.scale.denormalize_torch(dmp_params)
        self.y0 = dmp_params[:, :2].reshape(self.batch_s, self.dof, 1)
        self.goal = dmp_params[:, 2:4].reshape(self.batch_s, self.dof, 1)
        self.w = dmp_params[:, 4:].reshape(self.batch_s, self.dof, self.n_bf)

    def initializeDMP(self):
        self.x = ones(self.batch_s, 1, 1).to(DEVICE)
        self.c = exp(-self.cs_ax * linspace(0, self.cs_runtime, self.n_bf).reshape(-1, 1)).to(DEVICE)
        self.h = ones(self.n_bf, 1).to(DEVICE) * self.n_bf**1.5 / self.c / self.cs_ax
        self.y = clone(self.y0)
        self.dy = zeros(self.batch_s, self.dof, 1).to(DEVICE)
        self.ddy = zeros(self.batch_s, self.dof, 1).to(DEVICE)
        self.y_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.dy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.ddy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)

    def step(self, t):
        self.canonicalStep()
        psi = (exp(-self.h * (self.x - self.c)**2))
        f = self.frontTerm() * (self.w @ psi) / sum(psi, dim = 1).reshape(self.batch_s, 1, 1)

        self.ddy = (self.ay * (self.by * (self.goal - self.y) - self.dy / self.tau) + f) * self.tau
        self.dy = self.dy + (self.ddy * self.tau * self.dt)
        self.y = self.y + (self.dy * self.dt)
    
        self.y_track[:, t] = self.y.reshape(self.batch_s, self.dof)
        self.dy_track[:, t] = self.dy.reshape(self.batch_s, self.dof)
        self.ddy_track[:, t] = self.ddy.reshape(self.batch_s, self.dof)

    def canonicalStep(self):
        self.x = self.x + (-self.cs_ax * self.x * self.tau * self.dt)

    def frontTerm(self):
        self.term = self.x * (self.goal - self.y0)
        return self.term