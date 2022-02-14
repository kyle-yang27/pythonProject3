from unet_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
import copy


batch_size = 3 #changed batch size to 3
lr = 0.05 #changed learning rate to 0.05
momentum = 1    #changed momentum to 1
decay = 0.03 #changed decay to 0.03

train_dataset = TASDataset('tas500v1.1',aug_flag=1) 
val_dataset = TASDataset('tas500v1.1', eval=True, mode='val',aug_flag=1)
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test',aug_flag=1)


train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)
def dice_loss(pred, target):
    # p(pred,'preshape')
    # p(target.shape,'targetshape')
    ret = []
    
    pred = torch.softmax(pred,dim=1)
    # Ignore IoU for undefined class ("9")
    for cls in range(9):  # last class is ignored
        current_pred = pred[:,cls,:,:]
        # showimg(current_pred,'currepred')
        target_inds = target == cls
        # showimg(target_inds,'target_inds')
        intersection = torch.sum(current_pred*target_inds)
        denom = torch.sum(current_pred + target_inds)
        # p(intersection,'intersection')
        # p(denom,'denom')
        ret.append(1 - 2*intersection/denom)
        
    # print(ious)
    return np.sum(ret)/10
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases

epochs = 100
criterion = torch.nn.CrossEntropyLoss(reduction='none') # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
n_class = 10
unet_model = UNET(n_class=n_class)
unet_model.apply(init_weights)

optimizer =  torch.optim.Adam(unet_model.parameters(),lr,weight_decay=decay) # Use Adam to optimize

#choose an optimizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

unet_model = unet_model.to(device) #transfer the model to the device

# Returns list of normalized loss, pixel accuracy, and iou for training and validation data.
# TODO: Check if trianing pixel and iou is needed.
def train():
    best_iou_score = 0.0
    
    # For early stopping.
    patience_counter = 0
    patience_len = 5
    last_valid_loss = 1e10
    
    # Keep track of results.
    train_loss_list = []
    train_acc_list = []
    train_iou_list = []
    
    valid_loss_list = []
    valid_acc_list = []
    valid_iou_list = []
    
    for epoch in range(epochs):    
        print('epoch',epoch)
        ts = time.time()
        
        # To keep track of training values.
        losses = []
        mean_iou_scores = []
        accuracy = []
        
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            # p(inputs.shape,'inputshape')
            optimizer.zero_grad()
            labels = labels.long()
            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the model to the device #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the model to the device #transfer the labels to the same device as the model's

            outputs = unet_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = dice_loss(outputs, labels) #calculate loss
            
            # TODO: Shape of labels.
            numSamples = loss.flatten().shape[0]
            
            loss = torch.sum(loss)
            
            # Append losses.
            losses.append(loss.item() / numSamples)
            
            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch {}, iter {}, loss: {}".format(epoch, iter, loss.item() / numSamples))
                
            # For early stopping.
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        
        current_loss, current_acc, current_miou_score = val(epoch)
        
        # Append to result to list.
        valid_loss_list.append(current_loss)
        valid_acc_list.append(current_acc)
        valid_iou_list.append(current_miou_score)
        
        # TODO: implement patience.
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            torch.save(unet_model,'bestmodel_unet.pt')
            
        train_loss_list.append(np.mean(losses))
        # Early stopping
        if (current_loss > last_valid_loss):
            patience_counter += 1
            if(patience_counter >= patience_len):
                break
        else:
            patience_counter = 0
            
        last_valid_loss = current_loss
        
        
    return train_loss_list, train_acc_list, train_iou_list, valid_loss_list, valid_acc_list, valid_iou_list


# Runs model on validation data and returns loss, accuracy, iou score.
def val(epoch):
    unet_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            print('iter',iter)
            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the model to the device
            label = label.to(device) #transfer the model to the device

            output = unet_model(input)##batchszie * width * height * 10
            label = label.long()
            loss = dice_loss(output, label) #calculate loss
            
            # TODO: Shape of labels.
            numSamples = loss.flatten().shape[0]
            
            loss = torch.sum(loss)
            
            losses.append(loss.item() / numSamples) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 

            pred = torch.argmax(output,dim=1)
            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class))) # Complete this function in the util, notice the use of np.nanmean() here

            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    unet_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(accuracy), np.mean(mean_iou_scores)

def test():
    #TODO: load the best model and complete the rest of the function for testing
    pass

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    # test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()