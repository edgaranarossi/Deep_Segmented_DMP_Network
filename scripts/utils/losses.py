import torch
from torch.nn import MSELoss
from torch import ones, zeros, zeros_like, linspace, exp, clone, sum, cos, sin, tensor, cat, cdist, diff, clamp, floor, tile, sign
from .soft_dtw_cuda import SoftDTW

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

    def __call__(self, dmp_params, dmp_traj = None, rot_deg = None):
        if rot_deg == None:
            rot_deg = tensor(0).to(DEVICE)
        self.integrateDMP(dmp_params, rot_deg)
        mse = torch.nn.MSELoss()
        if dmp_traj != None:
            loss = mse(self.y_track, dmp_traj)
            return loss

    def integrateDMP(self, dmp_params, rot_deg):
        self.splitDMPparameters(dmp_params)
        self.initializeDMP()
        for t in range(self.timesteps):
            self.step(t, rot_deg)

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
        self.y = clone(self.y0).to(DEVICE)
        self.dy = zeros(self.batch_s, self.dof, 1).to(DEVICE)
        self.ddy = zeros(self.batch_s, self.dof, 1).to(DEVICE)
        self.y_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.dy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)
        self.ddy_track = zeros(self.batch_s, self.timesteps, self.dof).to(DEVICE)

    def step(self, t, rot_deg):
        self.canonicalStep()
        psi = (exp(-self.h * (self.x - self.c)**2))
        f = self.frontTerm() * (self.w @ psi) / sum(psi, dim = 1).reshape(self.batch_s, 1, 1)

        self.ddy = (self.ay * (self.by * (self.goal - self.rotateCoord(self.y, self.y0, -rot_deg)) - self.dy / self.tau) + f) * self.tau
        self.dy = self.dy + (self.ddy * self.tau * self.dt)
        self.y = self.rotateCoord(self.rotateCoord(self.y, self.y0, -rot_deg) + (self.dy * self.dt), self.y0, rot_deg)

        self.y_track[:, t] = self.y.reshape(self.batch_s, self.dof)
        self.dy_track[:, t] = self.dy.reshape(self.batch_s, self.dof)
        self.ddy_track[:, t] = self.ddy.reshape(self.batch_s, self.dof)

    def canonicalStep(self):
        self.x = self.x + (-self.cs_ax * self.x * self.tau * self.dt)

    def frontTerm(self):
        self.term = self.x * (self.goal - self.y0)
        return self.term

    def rotateCoord(self, y, y0, rot_deg):
        px = y[:, 0]
        py = y[:, 1]
        
        cx = y0[:, 0]
        cy = y0[:, 1]
        
        new_x = cos(rot_deg).to(DEVICE) * (px-cx) - sin(rot_deg).to(DEVICE) * (py-cy) + cx
        new_y = sin(rot_deg).to(DEVICE) * (px-cx) + cos(rot_deg).to(DEVICE) * (py-cy) + cy
        
        y_rot = cat([new_x, new_y], dim = 1).reshape(y.shape[0], self.dof, 1)
        return y_rot

class SegmentVelocityLoss:
    def __init__(self, train_param):
        self.train_param    = train_param
        self.model_param    = self.train_param.model_param
        self.dmp_param      = self.model_param.dmp_param

    def __call__(self, segment_start_end, sampling_ratio, original_traj):
        self.mse_loss           = MSELoss()
        self.batch_s            = segment_start_end.shape[0]
        self.pred_start_points  = segment_start_end[:, :-1, :]
        self.pred_end_points    = segment_start_end[:,  1:, :]
        self.label_start_points = zeros_like(self.pred_start_points).to(DEVICE)
        self.label_end_points   = zeros_like(self.pred_end_points).to(DEVICE)
        self.original_traj      = original_traj
        self.original_vel       = cat([zeros(self.batch_s, 1, 2).to(DEVICE), diff(original_traj, dim = 1)], dim = 1)
        self.max_segments       = self.model_param.max_segments
        self.traj_length        = self.model_param.traj_length
        
        closest_start_points_indices                = cdist(self.pred_start_points, original_traj, p = 2).min(dim = 2).indices.reshape(1, self.batch_s, -1)
        closest_end_points_indices                  = cdist(self.pred_end_points, original_traj, p = 2).min(dim = 2).indices.reshape(1, self.batch_s, -1)
        same_indices                                = closest_start_points_indices == closest_end_points_indices
        closest_end_points_indices[same_indices]    = clamp(closest_end_points_indices[same_indices] + 2, max = self.traj_length - 1)
        same_indices                                = closest_start_points_indices == closest_end_points_indices
        closest_start_points_indices[same_indices]  = closest_start_points_indices[same_indices] - 2
        closest_points_indices                      = cat([closest_start_points_indices, closest_end_points_indices], dim = 0)
        self.closest_start_points_indices           = closest_points_indices.min(dim = 0).values
        self.closest_end_points_indices             = closest_points_indices.max(dim = 0).values
        self.N                                      = (self.closest_end_points_indices - self.closest_start_points_indices).reshape(self.batch_s, -1, 1)
        one_N                                       = self.N == 1
        self.N[one_N]                               = 2

        self.segment_points_sum = zeros(self.batch_s, self.max_segments).to(DEVICE)
        self.segment_vel_sum = zeros(self.batch_s, self.max_segments, self.dmp_param.dof).to(DEVICE)
        for i in range(self.traj_length):
            cur_segment_idx = floor(tensor(i / self.traj_length).to(DEVICE)).int()

            start_check = i >= self.closest_start_points_indices
            end_check = i <= self.closest_end_points_indices
            in_segment_check = (start_check * end_check).int()[:, cur_segment_idx]            
            self.segment_points_sum[:, cur_segment_idx] = self.segment_points_sum[:, cur_segment_idx] + (1 * in_segment_check)

            end_check = i < self.closest_end_points_indices
            in_segment_check = (start_check * end_check).int()[:, cur_segment_idx].reshape(self.batch_s, 1)
            
            # print(self.segment_vel_sum[:, cur_segment_idx].shape, self.original_vel[:, i, :].shape, in_segment_check.shape)
            self.segment_vel_sum[:, cur_segment_idx] = self.segment_vel_sum[:, cur_segment_idx] + (self.original_vel[:, i, :] * in_segment_check)
        self.segment_vel_mean = self.segment_vel_sum / self.N

        self.segment_vel_variance = zeros(self.batch_s, self.max_segments, self.dmp_param.dof).to(DEVICE)
        for i in range(self.traj_length):
            cur_segment_idx = floor(tensor(i / self.traj_length).to(DEVICE)).int()

            start_check = i >= self.closest_start_points_indices
            end_check = i < self.closest_end_points_indices
            in_segment_check = (start_check * end_check).int()[:, cur_segment_idx].reshape(self.batch_s, 1)

            self.segment_vel_variance[:, cur_segment_idx] = self.segment_vel_variance[:, cur_segment_idx] + ((self.original_vel[:, i, :] - self.segment_vel_mean[:, cur_segment_idx])**2 * in_segment_check)
        self.segment_vel_variance = self.segment_vel_variance / (self.N - 1)

        self.segment_points_distribution = self.segment_points_sum / self.traj_length
        # print(self.segment_points_distribution, sampling_ratio)
        self.velocity_loss = self.mse_loss(self.segment_points_distribution, sampling_ratio)
        self.variance_loss = self.segment_vel_variance.sum()
        # print(self.velocity_loss, self.variance_loss)

        return (self.velocity_loss + self.variance_loss)

class SegmentLimitedMSE:
    def __init__(self, train_param):
        self.train_param    = train_param
        self.model_param    = self.train_param.model_param
        self.dmp_param      = self.model_param.dmp_param
        self.max_segments   = self.model_param.max_segments
        self.mse_loss       = MSELoss(reduction = 'none')

    def __call__(self, X, Y, num_segments_label):
        self.batch_s = Y.shape[0]

        total_loss = zeros(self.batch_s, 1).to(DEVICE)
        for i in range(self.max_segments):
            # multiplier = (abs(sign(clamp(num_segments_label - i, min = 0)) - 1) * 99) + 1
            multiplier = sign(clamp(num_segments_label - i, min = 0))
            total_loss = total_loss + self.mse_loss(X[:, i, :], Y[:, i, :]).mean(dim = 1).reshape(-1, 1) * multiplier
        loss = total_loss.mean() / self.max_segments
        return loss