import copy
import utils
import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    config = utils.load_yaml('config/hymenoptera_training.yaml')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train dataloader
    train_dataset = utils.create_instance(config['train_dataset'])
    train_loader = utils.create_instance(config['train_loader'], **{'data': train_dataset})

    # train eval dataloader
    train_eval_dataset = utils.create_instance(config['train_eval_dataset'])
    train_eval_loader = utils.create_instance(config['train_eval_loader'], **{'data': train_eval_dataset})

    # valid dataloader
    valid_dataset = utils.create_instance(config['valid_dataset'])
    valid_loader = utils.create_instance(config['valid_loader'], **{'data': valid_dataset})

    # model
    model = utils.create_instance(config['model'])

    # loss function
    loss_fn = utils.create_instance(config['loss'])

    # accuracy function
    accuracy_fn = utils.create_instance(config['accuracy'])

    optimizer = utils.create_instance(config['optimizer'], **{'params': model.parameters()})
    lr_scheduler = utils.create_instance(config['lr_scheduler'], **{'optimizer': optimizer})

    train_epoch = utils.create_instance(config['trainer'])
    eval_epoch = utils.create_instance(config['evaluator'])

    train_loss_history = []
    train_acc_history = []
    train_top5_acc_history = []

    valid_loss_history = []
    valid_acc_history = []
    valid_top5_acc_history = []

    best_valid_acc = 0.
    best_model_state_dict = dict()
    best_optim_state_dict = dict()

    num_epochs = 1000

    for epoch in range(num_epochs):
        train_acc, train_loss = train_epoch(dataloader=train_loader,
                                            model=model,
                                            loss_fn=loss_fn,
                                            accuracy_fn=accuracy_fn,
                                            optimizer=optimizer,
                                            device=device)
        train_eval_acc, train_eval_loss = eval_epoch(dataloader=train_eval_loader,
                                                     model=model,
                                                     loss_fn=loss_fn,
                                                     accuracy_fn=accuracy_fn,
                                                     device=device)
        valid_acc, valid_loss = eval_epoch(dataloader=valid_loader,
                                           model=model,
                                           loss_fn=loss_fn,
                                           accuracy_fn=accuracy_fn,
                                           device=device)

        print(f'#Epoch: {epoch + 1} - train_loss: {train_loss} - train_accuracy: {train_acc}')
        print(f'#Epoch: {epoch + 1} - train_eval_loss: {train_eval_loss} - train_eval_accuracy: {train_eval_acc} \n')
        print(f'#Epoch: {epoch + 1} - valid_loss: {valid_loss} - valid_accuracy: {valid_acc} \n')

        lr_scheduler.step(valid_loss)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optim_state_dict = copy.deepcopy(optimizer.state_dict())
            best_valid_acc = valid_acc

        # early_stopping(valid_loss, model)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)

    print(f'Best Validation Accuracy: {best_valid_acc:4f}')

    plt.plot(train_loss_history, color='b', label='train_loss')
    plt.plot(valid_loss_history, color='r', label='valid_loss')
    plt.legend()

    plt.plot(train_acc_history, color='b', label='train_acc')
    plt.plot(valid_acc_history, color='r', label='valid_acc')
    plt.legend()

    plt.show()
