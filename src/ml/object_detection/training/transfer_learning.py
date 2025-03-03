import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import multiprocessing

#early stop class to prevent model from overfitting on training using epoch accuracy as a metric
class EarlyStopping:
    def __init__(self, patience=4, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.early_stop = False
        self.best_model_params = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_params = model.state_dict()
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_params = model.state_dict()
            self.counter = 0

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5, save_path='best.pt'):
    since = time.time()

    #create instance of early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase] #changed to float for mps use
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model weights
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), save_path)
                
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop: #if we need to do an early stop
                    print("Stopping Training Early!")
                    time_elapsed = time.time() - since
                    print(f"Training stopped at epoch {epoch} after {time_elapsed// 60:.0f}m {time_elapsed % 60:.0f}s")
                    model.load_state_dict(torch.load(save_path, weights_only=True))#later on if we want to just train the model and save weights then remove this line to ensure we do not load the weights
                    return model
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load the best model weights
    model.load_state_dict(torch.load(save_path, weights_only=True))#later on if we want to just train the model and save weights then remove this line to ensure we do not load the weights
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    cudnn.benchmark = True
    plt.ion()   # interactive mode

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    """
    the following lines of code may give the most trouble on AWS if we are pulling from buckets of images
    current_dir gets the directory the code is running in
    data_dir then finds the directory where the folder 'dataset is', within 'dataset' will be 'train/val' folders, may need to split images later when getting the user's images
    once we are able to get the images into train/val folders then this should work
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), '../dataset')

    global image_datasets, dataloaders, dataset_sizes, class_names, device

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                 batch_size=4,
                                                 shuffle=True, 
                                                 num_workers=4)  # number of subprocesses to fetch data at a time
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #edited this to use mps instead of cpu, change later.
    print(device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names)) #edited this to contain and go to target of length of classes instead of 2 or 3
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    #decrease our learning rate by 0.1 after each 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    #train our model
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                           exp_lr_scheduler, num_epochs=25)

    #visualize the model on some data, removed this to just train and save the weights
    # visualize_model(model_conv)
    # plt.ioff()
    # plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

"""
Notes:
- Decaying the learning rate allows for the training to take large steps at first but 
    lowering after a couple of epochs allows the model for finer adjustments and converge
    to a better local minimum without overshooting the optimal solution
- By not decaying the learning rate it will likely overshoot the optimal solution and lead to 
    oscillations. Leads to instability and slower convergence
-Cross Entropy Loss: measures the difference between the predicted probability distribution 
    and the true distribution.
-for scaling, think about saving each weight .pt file for each user and replace based on if they want to keep
    or replace their old saved configurations in a bucket

TODO:
- When training on 1 class, accuracy goes to 1, need negative data/class to set a threshold
"""



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import torch.backends.cudnn as cudnn
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# from PIL import Image
# import multiprocessing

# #early stop class to prevent model from overfitting on training using epoch accuracy as a metric
# class EarlyStopping:
#     def __init__(self, patience=4, verbose=False):
#         self.patience = patience
#         self.verbose = verbose
#         self.best_loss = None
#         self.early_stop = False
#         self.best_model_params = None

#     def __call__(self, val_loss, model):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.best_model_params = model.state_dict()
#         elif val_loss > self.best_loss:
#             self.counter += 1
#             if self.verbose:
#                 print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.best_model_params = model.state_dict()
#             self.counter = 0

# def imshow(inp, title=None):
#     """Display image for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=5, save_path='best.pt'):
#     since = time.time()

#     # Create instance of early stopping
#     early_stopping = EarlyStopping(patience=patience, verbose=True)
#     best_acc = 0.0

#     # Lists to store loss and accuracy values for plotting
#     train_loss_history = []
#     train_acc_history = []
#     val_loss_history = []
#     val_acc_history = []

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # Save metrics for plotting
#             if phase == 'train':
#                 train_loss_history.append(epoch_loss)
#                 train_acc_history.append(epoch_acc)
#             else:
#                 val_loss_history.append(epoch_loss)
#                 val_acc_history.append(epoch_acc)

#             # Save best model weights
#             if phase == 'val':
#                 if epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     torch.save(model.state_dict(), save_path)

#                 early_stopping(epoch_loss, model)
#                 if early_stopping.early_stop:  # If we need to do an early stop
#                     print("Stopping Training Early!")
#                     time_elapsed = time.time() - since
#                     print(f"Training stopped at epoch {epoch} after {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
#                     model.load_state_dict(torch.load(save_path, weights_only=True))
#                     # Plot the training and validation metrics
#                     plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
#                     return model
#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # Load the best model weights
#     model.load_state_dict(torch.load(save_path, weights_only=True))

#     # Plot the training and validation metrics
#     plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

#     return model


# def plot_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
#     plt.figure(figsize=(12, 5))

#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss_history, label='Train Loss')
#     plt.plot(val_loss_history, label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Plot Accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc_history, label='Train Accuracy')
#     plt.plot(val_acc_history, label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)

# def main():
#     cudnn.benchmark = True
#     plt.ion()   # interactive mode

#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }

#     """
#     the following lines of code may give the most trouble on AWS if we are pulling from buckets of images
#     current_dir gets the directory the code is running in
#     data_dir then finds the directory where the folder 'dataset is', within 'dataset' will be 'train/val' folders, may need to split images later when getting the user's images
#     once we are able to get the images into train/val folders then this should work
#     """
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(os.path.dirname(current_dir), '../dataset')#edited this line since we are using curr Dir but needed dir before to access dataset folder

#     global image_datasets, dataloaders, dataset_sizes, class_names, device

#     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                             data_transforms[x])
#                     for x in ['train', 'val']}
#     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
#                                                  batch_size=4,
#                                                  shuffle=True, 
#                                                  num_workers=4)  # number of subprocesses to fetch data at a time
#                   for x in ['train', 'val']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#     class_names = image_datasets['train'].classes

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Get a batch of training data
#     inputs, classes = next(iter(dataloaders['train']))

#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out, title=[class_names[x] for x in classes])

#     model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
#     for param in model_conv.parameters():
#         param.requires_grad = False

#     num_ftrs = model_conv.fc.in_features
#     model_conv.fc = nn.Linear(num_ftrs, len(class_names)) #edited this to contain and go to target of length of classes instead of 2 or 3
#     model_conv = model_conv.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
#     #decrease our learning rate by 0.1 after each 7 epochs
#     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#     #train our model
#     model_conv = train_model(model_conv, criterion, optimizer_conv,
#                            exp_lr_scheduler, num_epochs=25)

#     #visualize the model on some data, removed this to just train and save the weights
#     # visualize_model(model_conv)
#     # plt.ioff()
#     # plt.show()

# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     main()