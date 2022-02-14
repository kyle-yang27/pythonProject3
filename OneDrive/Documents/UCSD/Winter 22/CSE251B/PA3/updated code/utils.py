import numpy as np
import torch

import matplotlib.pyplot as plt

def iou(pred, target, n_classes = 10):
  ious = []
  # p(pred.shape,'pred.shape')
  # p(target.shape,'target.shape')
  pred = pred.view(-1)
  target = target.view(-1)
  
  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = pred_inds & target_inds
    union = pred_inds|target_inds
    # showimg(union,'union')
    # showimg(target_inds,'target')
    
    if union.any():
      ious.append((torch.sum(intersection)/torch.sum(union)).item())
    else:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    
  # print(ious)
  return np.array(ious)

def pixel_acc(pred, target):
  correct = pred == target
  sample = target != 9
  correct = correct & sample
  acc = (torch.sum(correct)/torch.sum(sample)).item()
  print(acc)
  return np.array(acc)
    
def showimg(img,title):
  img = img.long()
  img.cpu()

  img = img.reshape(-1,384,768)
  print(img.shape)
  for i in img:
    i = i.cpu()
    a = plt.figure()
    plt.title(title)
    plt.imshow(i,cmap='Blues', interpolation='none')
    plt.colorbar()
    plt.show()

def p(content,name='',debug=True):
    if(debug==True):
        print(name,end='\t')
        print(content,end='\n')
        
def plot(x1,x2,name1,name2,ylabel,title):
    n = len(x1)
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 22})
    plt.plot(np.arange(n),x1,label=name1 ,color = "blue")
    plt.plot(np.arange(n),x2,label=name2,color = "red")
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def my_save(name,obj):
    name = './repData/' + name
    import pickle
    data = obj
    f = open(name, 'wb')
    pickle.dump(data, f)

def my_load(name):
    import pickle
    name = './repData/' + name
    f = open(name, 'rb')
    r = pickle.load(f)
    return r