import torch


class RunningNormalizer:
    def __init__(self, shape, device, epsilon=1e-8):
        self.mean = torch.zeros(shape, device=device)
        self.std = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.device = device

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_size = x.shape[0]
        batch_var = torch.var(x, dim=0, unbiased=False)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_size

        new_mean = self.mean + delta * batch_size / tot_count
        m_a = self.std ** 2 * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta ** 2 * self.count * batch_size / tot_count
        new_std = torch.sqrt(M2 / tot_count)

        self.mean = new_mean
        self.std = new_std
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.epsilon)

    def state_dict(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'count': self.count
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean'].to(self.device)
        self.std = state_dict['std'].to(self.device)
        self.count = state_dict['count']
