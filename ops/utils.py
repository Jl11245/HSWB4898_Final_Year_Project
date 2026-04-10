import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def multi_label_accuracy(output, target, threshold=0):
    # Apply threshold to output tensor
    output = (output > threshold).float()

    # Check if the tensors are on the same device and are of the same dtype
    if output.device != target.device:
        target = target.to(output.device)

    if output.dtype != target.dtype:
        target = target.float()

    # Compute accuracy
    correct = (output == target).all(dim=1).float() # 1's where all labels match, 0's where they don't
    accuracy = correct.mean() # Average to get final accuracy

    return accuracy.item() # Return as standard python number

def per_label_accuracy(output, target, threshold=0):
    # Apply threshold to output tensor
    output = (output > threshold).float()

    # Check if the tensors are on the same device and are of the same dtype
    if output.device != target.device:
        target = target.to(output.device)

    if output.dtype != target.dtype:
        target = target.float()

    # Compute per-label accuracy
    correct = (output == target).float() # 1's where the label is correct, 0 where it's not
    accuracy_per_label = correct.mean(dim=0) # Average over the batch dimension

    return accuracy_per_label.cpu().numpy() # Return as a PyTorch tensor


def false_nagetive(output, target, thredHold):
    batchSize = output.size(0)
    values = output.cpu().numpy()
    targets = target.cpu().numpy()
    falseNegativeCounter = 0
    
    for index, value in enumerate(values):
        # Find the indices of the two largest numbers
        twoLargestIndices = np.argpartition(value, -2)[-2:]
        # Sort the indices by the values in the array
        sortedIndices = twoLargestIndices[np.argsort(value[twoLargestIndices])][::-1]
        # Get the first maximum value and its index
        firstMaxIndex = sortedIndices[0]
        firstMaxValue = value[firstMaxIndex]
        # Get the second maximum value and its index
        secondMaxIndex = sortedIndices[1]
        secondMaxValue = value[secondMaxIndex]
        # Get text
        if firstMaxIndex == 0:
            if secondMaxValue / firstMaxValue < thredHold:
                result = 0
            else:
                result = secondMaxIndex
        else:
            result = firstMaxIndex

        if result != targets[index]:
            if result != 0 and targets[index] == 0:
                falseNegativeCounter += 1
    
    return falseNegativeCounter / batchSize


def false_nagetive_multilabel(output, target, threshold):
    output = (output > threshold).float()
    outputs = output.cpu().numpy()
    targets = target.cpu().numpy()

    mcm = multilabel_confusion_matrix(targets, outputs)

    # Calculate false negatives for each class
    fn = [mcm[i][1][0] for i in range(mcm.shape[0])]

    return fn


