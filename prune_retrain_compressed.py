"""
This script loads in a model and iterively removes weights and retrains the remaining healthy ones
Accuracy and representational dissimilarity metrics for each injury steps get saved
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import main
import registry

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path_orig = "run/cifar10/pretrain/cifar10_vgg19.pth"
path_compressed = "run/cifar10/prune/cifar10-global-l1-vgg19/cifar10_vgg19_l1_3.0.pth"

num_epochs = 3
batch_size = 40
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
num_injury_steps = 15
percent_to_injure = .2

model = registry.get_model('vgg19', num_classes=10)

# load compressed model, follow model compression techniques from https://github.com/VainF/Torch-Pruning
model = torch.load(path_compressed)
# model.load_state_dict(torch.load(path_orig))
model = model.to(device)

# a dict to store the activations
activation = {}

def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

num_classes, train_dst, val_dst, input_size = registry.get_dataset('cifar10', data_root="data")

test_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, num_workers=4)
train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,)


# sort indices of test images into lists corresponding to same class for RDMs
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

gt_class = [val_dst[i][1] for i in range(len(val_dst))]

lists = [[] for i in range(10)]
for i in range(len(gt_class)):
    for j in range(10):
        if gt_class[i] == j:
            lists[j].append(i)


def train_model(
        model,
        train_loader,
        test_loader,
        epochs,
        lr,

        # For pruning
        weight_decay=5e-4,
        regularizer=None,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if regularizer is None else 0.0,
    )

    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if regularizer is not None:
                regularizer(model)  # for sparsity learning
            optimizer.step()
            if i % 100 == 0:
                print(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )

        model.eval()
        acc, val_loss = main.eval(model, test_loader, device=device)
        print(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )


# accuracy of healthy model
print(main.eval(model, test_loader, device))

# register hooks on penultimate layer to extract activations for healthy model
model.pool2.register_forward_hook(getActivation('penultimate_layer'))
model.eval()

activation_layers_healthy = dict()
activation_layers_healthy[model.pool2] = {}
layer_name = model.pool2

# get activations sorted by each class and save them into a dataframe
for i in range(len(lists)):
    for j in range(100):
        # print(lists[i][j])
        img, label = val_dst[lists[i][j]]
        img = img.to(device)

        out = model(img.unsqueeze(0))

        get_layer_activation = activation['penultimate_layer']
        get_layer_activation = get_layer_activation.cpu()
        activ = np.asarray(get_layer_activation).squeeze()
        activation_layers_healthy[layer_name]['img_' + str(i) + str(j)] = activ.flatten()

df_healthy = pd.DataFrame(activation_layers_healthy[layer_name])

# create dissimilarity matrix
rdm_healthy = (1 - df_healthy.corr())

# flatten RDM to compare to 'injured' RDMs
flat_healthy = rdm_healthy.values.flatten()

pruned_accuracy = []
retrained_accuracy = []
tau_pruned = []
tau_retrained = []
injury_level = []

# loop through progressive 'injury' to the model
for i in range(num_injury_steps):
    x = 1-(1 - percent_to_injure) ** i
    injury_level.append(x)

    for name, module in model.named_modules():
        # iterively prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.random_unstructured(module, name='weight', amount=0.2)
        # iterively prune 20% of connections in all linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.random_unstructured(module, name='weight', amount=0.2)

#    pruned_path = "injured_model_paths/full_model/full_prune_step_" + str(i) + ".pth"
#    torch.save(model.state_dict(), pruned_path)

    # test injured model accuracy
    print('pruned')
    accuracy = main.eval(model, test_loader, device)
    pruned_accuracy.append(accuracy)

    # activations for all images in test set for pruned model
    activation = {}
    activation_layers_pruned = dict()
    activation_layers_pruned[model.pool2] = {}
    layer_name = model.pool2

    for k in range(len(lists)):
        for j in range(100):
            # print(lists[k][j])
            img, label = val_dst[lists[k][j]]
            img = img.to(device)

            out = model(img.unsqueeze(0))

            get_layer_activation = activation['penultimate_layer']
            get_layer_activation = get_layer_activation.cpu()
            activ = np.asarray(get_layer_activation).squeeze()
            activation_layers_pruned[layer_name]['img_' + str(k) + str(j)] = activ.flatten()

    # Kendall's tau correlation to healthy activations
    df_injured = pd.DataFrame(activation_layers_pruned[layer_name])
    rdm_injured = 1 - df_injured.corr()
    flat_injured = rdm_injured.values.flatten()

    combine = np.column_stack((flat_healthy, flat_injured))
    combine = pd.DataFrame(combine)

    corr_kendall = combine.corr(method='kendall')
    tau_value = corr_kendall.iat[1, 0]
    tau_pruned.append(tau_value)

    print('injured tau complete')

    # retrain model
    train_model(model, train_loader, test_loader, 3, learning_rate)

    # test retrained model accuracy
    print('retrained')
  
#    retrained_path = "injured_model_paths/full_model/full_retrained_step_" + str(i) + ".pth"
#    torch.save(model.state_dict(), retrained_path)

    accuracy_ret = main.eval(model, test_loader, device)
    retrained_accuracy.append(accuracy_ret)

    # activations for all images in test set for retrained model
    activation = {}
    activation_layers_retrained = dict()
    activation_layers_retrained[model.pool2] = {}
    layer_name = model.pool2

    for i in range(len(lists)):
        for j in range(100):
            img, label = val_dst[lists[i][j]]
            img = img.to(device)

            out = model(img.unsqueeze(0))

            get_layer_activation = activation['penultimate_layer']
            get_layer_activation = get_layer_activation.cpu()
            activ = np.asarray(get_layer_activation).squeeze()
            activation_layers_retrained[layer_name]['img_' + str(i) + str(j)] = activ.flatten()
    print('retrained tau complete')

    # Kendall's tau correlation to healthy activations
    df_retrained = pd.DataFrame(activation_layers_retrained[layer_name])
    rdm_retrained = 1 - df_retrained.corr()
    flat_retrained = rdm_retrained.values.flatten()

    combine_r = np.column_stack((flat_healthy, flat_retrained))
    combine_r = pd.DataFrame(combine_r)

    corr_kendall_r = combine_r.corr(method='kendall')
    tau_value_r = corr_kendall_r.iat[1, 0]
    tau_retrained.append(tau_value_r)

# save metrics for analysis
np.savetxt('compressed_pruned_accuracy.csv', pruned_accuracy)
np.savetxt('compressed_retrained_accuracy.csv', retrained_accuracy)
np.savetxt('compressed_tau_pruned.csv', tau_pruned)
np.savetxt('compressed_tau_retrained.csv', tau_retrained)
