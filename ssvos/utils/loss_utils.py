"""Implement some frequently used loss in MindSpore."""
from mindspore import nn, ops, Tensor
from mindspore.common import dtype as mstype


class CrossEntropyLoss(nn.Cell):
    """Computes softmax cross entropy between logits and labels."""
    def __init__(self, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        if reduction == "sum":
            self.reduction = ops.ReduceSum()
        if reduction == "mean":
            self.reduction = ops.ReduceMean()
        self.one_hot = ops.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)[0]
        loss = self.reduction(loss, (-1,))
        return loss