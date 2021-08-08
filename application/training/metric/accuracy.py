from torch import nn


class Accuracy(nn.Module):
    def __init__(self, topk=(1,)):
        self.topk = topk

    def forward(self, preds, targets):
        """Computes the precision@k for the specified values of k"""
        batch_size = targets.shape[0]

        # get all indices of top k of the largets values in preds based on dim=1
        _, topk_indices = preds.topk(k=max(self.topk), dim=1, largest=True, sorted=True)  # B x k

        topk_preds = topk_indices.t()  # k x B
        topk_targets = targets.view(1, -1).expand_as(topk_preds)  # B -> 1 x B -> k x B

        topk_corrects = topk_preds == topk_targets  # k x B

        topk_results = []
        for k in self.topk:
            k_corrects = topk_corrects[:k].reshape(-1, 1).float().sum(dim=0)
            topk_results.append(k_corrects * 100.0 / batch_size)

        return topk_results
