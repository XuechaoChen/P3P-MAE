
import math
import numpy as np

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.scheduler.kwargs.initial_epochs:
        lr = args.optimizer.kwargs.lr * epoch / args.scheduler.kwargs.initial_epochs 
    else:
        lr = 1e-6 + (args.optimizer.kwargs.lr - 1e-6) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.scheduler.kwargs.initial_epochs) / (args.scheduler.kwargs.epochs - args.scheduler.kwargs.initial_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule