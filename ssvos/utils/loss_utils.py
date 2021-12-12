"""Implement some frequently used loss in MindSpore."""
from mindspore import nn, ops


class CrossEntropyLoss(nn.Cell):
    """Computes softmax cross entropy between logits and labels."""
    def __init__(self, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        if reduction == "sum":
            self.reduction = ops.ReduceSum()
        if reduction == "mean":
            self.reduction = ops.ReduceMean()

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)[0]
        loss = self.reduction(loss, (-1,))
        return loss