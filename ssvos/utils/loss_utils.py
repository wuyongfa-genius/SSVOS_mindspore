"""Implement some frequently used loss in MindSpore."""
from mindspore import numpy as np, dtype as mstype
from mindspore import nn, ops


class CrossEntropyLoss(nn.Cell):
    """Computes softmax cross entropy between logits and labels."""
    def __init__(self, reduction="mean", is_int_label=True):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=is_int_label)
        if reduction == "sum":
            self.reduction = ops.ReduceSum()
        if reduction == "mean":
            self.reduction = ops.ReduceMean()

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)[0]
        loss = self.reduction(loss, (-1,))
        return loss


class CosineSimilarityLoss(nn.Cell):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.normalize = ops.L2Normalize(axis=-1)
        self.reduction = ops.ReduceMean()
        self.reduce_sum_op = ops.ReduceSum()

    def construct(self, a, b):
        a = self.normalize(a)
        b = self.normalize(b)
        loss = self.reduce_sum_op(a * b, (-1,))

        return 2 - 2 * self.reduction(loss, (-1,))


class InfoNCELoss(nn.Cell):
    def __init__(self, T=0.3, rank=0):
        super().__init__()
        self.T = T
        self.rank = rank
        self.normalize = ops.L2Normalize(axis=1)
        self.cross_entropy = CrossEntropyLoss()
        # init some ops to be used
        self.all_gather = ops.AllGather()
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, q, k):
        N = q.shape[0]
        q = self.normalize(q)
        k = self.normalize(k)
        # gather all targets
        k = self.all_gather(k)
        logits = self.matmul(q, k) / self.T
        # create labels
        labels = np.arange(N, dtype=mstype.int64) + N*self.rank
        cross_entropy = self.cross_entropy(logits, labels)

        return 2*self.T*cross_entropy