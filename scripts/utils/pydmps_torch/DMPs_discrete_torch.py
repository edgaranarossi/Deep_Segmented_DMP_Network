import torch
from torch import ones, zeros, linspace, exp, clone, sum, cos, sin, tensor, cat, from_numpy

if __name__ == '__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Based on studywolf's pydmps
https://github.com/studywolf/pydmps
https://pypi.org/project/pydmps/
"""

class DMPs_discrete_torch:
    def __init__(self, n_dmps, n_bfs, ay, dt, by = None):
        """
        Initialize the DMPs_discrete_torch class.

        Parameters:
        n_dmps (int): Number of dynamic movement primitives.
        n_bfs (int): Number of basis functions.
        ay (float): Gain term for the dynamical system.
        dt (float): Time step.
        by (float, optional): Gain term for the dynamical system. Defaults to ay/4.
        """
        self.dof = n_dmps
        self.dt = dt
        self.n_bf = n_bfs
        self.ay = ay
        self.by = by if by != None else self.ay/4
        self.cs_runtime = 1.0
        self.cs_ax = 1.0
        self.tau = 1
        self.timesteps = int(self.cs_runtime / self.dt)

        # To be defined
        self.w = None       # shape: (batch_size, dof, n_bf)
        self.y0 = None      # shape: (batch_size, dof, 1)
        self.goal = None    # shape: (batch_size, dof, 1)

    def rollout(self, prev_dy = None, prev_ddy = None, tau = None, rot_deg = None):
        """
        Perform a rollout of the DMP.

        Parameters:
        prev_dy (tensor, optional): Previous velocity. Defaults to None.
        prev_ddy (tensor, optional): Previous acceleration. Defaults to None.
        tau (float, optional): Temporal scaling factor. Defaults to None.
        rot_deg (tensor, optional): Rotation degree. Defaults to None.

        Returns:
        tuple: y_track, dy_track, ddy_track
        """
        assert self.w != None and self.y0 != None and self.goal != None
        self.batch_s = self.w.shape[0]

        self.reset_state(dy = prev_dy, ddy = prev_ddy)
        if tau != None:
            self.tau = tau
            self.timesteps = int(self.timesteps / self.tau)
        if rot_deg == None:
            rot_deg = tensor(0).to(DEVICE)

        for t in range(self.timesteps):
            self.step(t, rot_deg)

        return self.y_track, self.dy_track, self.ddy_track

    def reset_state(self, dy = None, ddy = None):
        """
        Reset the state of the DMP.

        Parameters:
        dy (tensor, optional): Initial velocity. Defaults to None.
        ddy (tensor, optional): Initial acceleration. Defaults to None.
        """
        self.x = ones(self.batch_s, 1, 1).to(DEVICE)
        self.c = exp(-self.cs_ax * linspace(0, self.cs_runtime, self.n_bf).reshape(-1, 1)).to(DEVICE)
        self.h = ones(self.n_bf, 1).to(DEVICE) * self.n_bf**1.5 / self.c / self.cs_ax
        self.y = clone(self.y0).to(DEVICE)
        self.dy = (zeros(self.batch_s, self.dof, 1).to(DEVICE)) if dy == None else dy.reshape(self.batch_s, -1, 1)
        self.ddy = (zeros(self.batch_s, self.dof, 1).to(DEVICE)) if ddy == None else ddy.reshape(self.batch_s, -1, 1)
        self.y_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.dy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.ddy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)

    def step(self, t, rot_deg):
        """
        Perform a single step of the DMP.

        Parameters:
        t (int): Current timestep.
        rot_deg (tensor): Rotation degree.
        """
        self.canonicalStep()
        psi = (exp(-self.h * (self.x - self.c)**2))
        f = self.frontTerm() * (self.w @ psi) / sum(psi, dim = 1).reshape(self.batch_s, 1, 1)

        # self.ddy = (self.ay * (self.by * (self.goal - self.rotateCoord(self.y, self.y0, -rot_deg)) - self.dy / self.tau) + f) * self.tau
        # self.dy = self.dy + (self.ddy * self.tau * self.dt)
        # self.y = self.rotateCoord(self.rotateCoord(self.y, self.y0, -rot_deg) + (self.dy * self.dt), self.y0, rot_deg)

        self.ddy = (self.ay * (self.by * (self.goal - self.y) - self.dy / self.tau) + f) * self.tau
        self.dy = self.dy + (self.ddy * self.tau * self.dt)
        self.y = self.y + (self.dy * self.dt)

        self.y_track[:, t] = self.y.reshape(self.batch_s, self.dof)
        self.dy_track[:, t] = self.dy.reshape(self.batch_s, self.dof)
        self.ddy_track[:, t] = self.ddy.reshape(self.batch_s, self.dof)

    def canonicalStep(self):
        """
        Update the canonical system state.
        """
        self.x = self.x + (-self.cs_ax * self.x * self.tau * self.dt)

    def frontTerm(self):
        """
        Compute the front term of the DMP.

        Returns:
        tensor: The front term.
        """
        self.term = self.x * (self.goal - self.y0)
        return self.term

    def rotateCoord(self, y, y0, rot_deg):
        """
        Rotate the coordinates.

        Parameters:
        y (tensor): Current position.
        y0 (tensor): Initial position.
        rot_deg (tensor): Rotation degree.

        Returns:
        tensor: Rotated coordinates.
        """
        px = y[:, 0]
        py = y[:, 1]
        
        cx = y0[:, 0]
        cy = y0[:, 1]
        
        new_x = cos(rot_deg).to(DEVICE) * (px-cx) - sin(rot_deg).to(DEVICE) * (py-cy) + cx
        new_y = sin(rot_deg).to(DEVICE) * (px-cx) + cos(rot_deg).to(DEVICE) * (py-cy) + cy
        y_rot = cat([new_x, new_y], dim = 1).reshape(y.shape[0], self.dof, 1)
        return y_rot
