import numpy as np

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max

class DMPParamScale:
    def __init__(self, y_old, w_old, y_new = [-1.0, 1.0], w_new = [-1.0, 1.0]):
        self.y_old = MinMax(y_old[0], y_old[1])
        self.y_new = MinMax(y_new[0], y_new[1])
        self.w_old = MinMax(w_old[0], w_old[1])
        self.w_new = MinMax(w_new[0], w_new[1])

    def normalize(self, X):
        X_np = np.array(X)
        X_normalized = np.zeros_like(X_np)
        X_normalized[:, :4] = (self.y_new.max - self.y_new.min) * \
                              (X_np[:, :4] - self.y_old.min) / \
                              (self.y_old.max - self.y_old.min) + \
                              self.y_new.min
        X_normalized[:, 4:] = (self.w_new.max - self.w_new.min) * \
                              (X_np[:, 4:] - self.w_old.min) / \
                              (self.w_old.max - self.w_old.min) + \
                              self.w_new.min
        return X_normalized

    def denormalize_np(self, X):
        X_np = np.array(X)
        X_denormalized = np.zeros_like(X_np)
        X_denormalized[:, :4] = (X_np[:, :4] - self.y_new.min) / \
                                (self.y_new.max - self.y_new.min) * \
                                (self.y_old.max - self.y_old.min) + \
                                self.y_old.min
        X_denormalized[:, 4:] = (X_np[:, 4:] - self.w_new.min) / \
                                (self.w_new.max - self.w_new.min) * \
                                (self.w_old.max - self.w_old.min) + \
                                self.w_old.min
        return X_denormalized

    def denormalize_torch(self, X):
        X_denormalized = torch.zeros_like(X)
        X_denormalized[:, :4] = (X[:, :4] - self.y_new.min) / \
                                (self.y_new.max - self.y_new.min) * \
                                (self.y_old.max - self.y_old.min) + \
                                self.y_old.min
        X_denormalized[:, 4:] = (X[:, 4:] - self.w_new.min) / \
                                (self.w_new.max - self.w_new.min) * \
                                (self.w_old.max - self.w_old.min) + \
                                self.w_old.min
        return X_denormalized