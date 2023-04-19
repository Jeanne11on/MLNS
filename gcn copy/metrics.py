import torch
import torch.nn.functional as F

"""
The PyTorch equivalent of TensorFlow's tf.nn.softmax_cross_entropy_with_logits is torch.nn.functional.cross_entropy.
The reduction argument is set to 'none' to return per-element losses. The tf.cast function is replaced with the .float() method.
The tf.equal function is replaced with the torch.eq function. Finally, the tf.reduce_mean function is replaced with torch.mean.
"""

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = F.cross_entropy(input=preds, target=torch.argmax(labels, dim=1), reduction='none')
    mask = mask.float()
    mask /= torch.mean(mask)
    loss *= mask
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = torch.eq(torch.argmax(preds, dim=1), torch.argmax(labels, dim=1))
    accuracy_all = correct_prediction.float()
    mask = mask.float()
    mask /= torch.mean(mask)
    accuracy_all *= mask
    return torch.mean(accuracy_all)