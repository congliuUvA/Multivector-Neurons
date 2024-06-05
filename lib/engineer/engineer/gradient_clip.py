import torch
import math
import collections


class Clipper:

    def clip_gradient(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.clip_gradient(*args, **kwargs)


class ValueClipper(Clipper):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def clip_gradient(self, model):
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.max_norm, norm_type=2.0
        )


class AdativeStdClipper(Clipper):
    def __init__(self, max_length=50, n_std=2, n_mean=1.5, verbose=True):
        self.max_length = max_length
        self.n_std = n_std
        self.n_mean = n_mean
        self.verbose = verbose

        self.gradnorm_queue = collections.deque(maxlen=self.max_length)

    def mean_std(self):
        length = len(self.gradnorm_queue) 
        mu = sum(self.gradnorm_queue) / length if length != 0 else 1e5
        std = sum(math.pow(x - mu, 2) / length for x in self.gradnorm_queue) ** 0.5
        return mu, std

    def clip_gradient(self, model):
        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        mu, std = self.mean_std()
        max_grad_norm = self.n_mean * mu + self.n_std * std

        # Clips gradient and returns the norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )
        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.append(max_grad_norm)
        else:
            self.gradnorm_queue.append(grad_norm.item())

        # if self.verbose and float(grad_norm) > max_grad_norm:
        #     print(
        #         f"Clipped gradient with value {grad_norm:.1f} "
        #         f"while allowed {max_grad_norm:.1f}"
        #     )
        return grad_norm


def setup_gradclip(name):
    return {
        "value": ValueClipper,
        "adaptive": AdativeStdClipper,
        None: lambda: ValueClipper(float("inf")),
    }[name]()
