import math


def recu_tau(epoch, epochs, tau_min=0.85, tau_max=0.99):
    """Official ReCU exponential tau schedule."""
    e = math.e
    A = (tau_max - tau_min) / (e - 1.0)
    B = tau_min - A
    return A * math.exp(epoch / epochs) + B


def official_warmup_lr(base_lr, epoch):
    """Official code uses 5 warm-up epochs: lr * (epoch+1)/5."""
    return base_lr * (epoch + 1) / 5.0
