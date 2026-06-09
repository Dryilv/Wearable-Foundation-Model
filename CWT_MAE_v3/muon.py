import torch
import torch.optim as optim


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration for matrix orthogonalization.
    Forces bfloat16 internally to avoid overflow in float16.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)

    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(optim.Optimizer):
    """
    Muon optimizer for 2D matrix weights (nn.Linear).
    Implements Newton-Schulz orthogonalization with adjusted LR scaling.
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                update = zeropower_via_newtonschulz5(buf)
                update = update.to(p.dtype)

                scale = 0.2 * max(p.size(0), p.size(1)) ** 0.5

                p.data.mul_(1.0 - lr * weight_decay)
                p.data.add_(update, alpha=-lr * scale)

        return loss
