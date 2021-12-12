"""Implement some lr schedule"""
import math
from mindspore import ops
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class CosineDecayLRWithWarmup(LearningRateSchedule):
    def __init__(self,
                max_lr,
                min_lr,
                total_steps,
                warmup_steps,
                by_epoch=False):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps-warmup_steps
        self.by_epoch = by_epoch
        self.math_pi = math.pi
        self.delta = max_lr - min_lr
        self.cos = ops.Cos()
        self.min = ops.Minimum()
        self.cast = ops.Cast()
    
    def construct(self, global_step):
        if global_step <= self.warmup_steps:
            ratio = global_step / self.warmup_steps
            return ratio * self.delta + self.min_lr
        else:
            ratio = (global_step - self.warmup_steps) / self.decay_steps
            degrees = ratio* self.math_pi
            cosine_value = 1/2 * (self.cos(degrees) + 1.)
            return self.delta * cosine_value + self.min_lr