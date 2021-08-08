import torch
from torch import nn


class TrainEpoch(nn.Module):
    def __init__(self):
        super(TrainEpoch, self).__init__()

    def forward(self, dataloader, model, loss_fn, accuracy_fn, optimizer, device='cpu'):
        model.to(device)
        model.train()  # set training mode

        top1_accs, top5_accs, losses = [], [], []

        for samples, targets in dataloader:
            # set device for samples and targets
            samples = samples.to(device)
            targets = targets.to(device)

            # compute predictions and loss value
            preds = model(samples)
            loss = loss_fn(preds, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()  # dw = 0
            loss.backward()  # get dw
            optimizer.step()  # w = w - lr * dw

            # compute topk accuracy
            top1_acc, top5_acc = accuracy_fn(preds, targets, topk=(1, 1))

            # append
            top1_accs.append(top1_acc.item())
            top5_accs.append(top5_acc.item())
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses) if len(losses) else 0.
        avg_top1_acc = sum(top1_accs) / len(top1_accs) if len(top1_accs) else 0.
        avg_top5_acc = sum(top5_accs) / len(top5_accs) if len(top5_accs) else 0.

        return avg_top1_acc, avg_top5_acc, avg_loss


class EvalEpoch(nn.Module):
    def __init__(self):
        super(EvalEpoch, self).__init__()

    def forward(self, dataloader, model, loss_fn, accuracy_fn, device='cpu'):
        model.to(device)
        model.eval()    # set valuating mode

        top1_accs, top5_accs, losses = [], [], []

        with torch.no_grad():
            for samples, targets in dataloader:
                samples = samples.to(device)
                targets = targets.to(device)

                preds = model(samples)
                loss = loss_fn(preds, targets)

                # compute topk accuracy
                top1_acc, top5_acc = accuracy_fn(preds, targets, topk=(1, 1))

                # append
                top1_accs.append(top1_acc.item())
                top5_accs.append(top5_acc.item())
                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses) if len(losses) else 0.
            avg_top1_acc = sum(top1_accs) / len(top1_accs) if len(top1_accs) else 0.
            avg_top5_acc = sum(top5_accs) / len(top5_accs) if len(top5_accs) else 0.

            return avg_top1_acc, avg_top5_acc, avg_loss
