import torch
from torch import tensor
from data_preparation import *



model = torch.load("/content/drive/My Drive/IMAGE_RECOGNITION/checkpoint_path")
model.eval()
test_acc_count=0

for k, (test_images, test_labels) in enumerate(dataloaders['test']):
  test_outputs = model(test_images)
  alpha, prediction = torch.max(test_outputs.data, 1)
  #print(test_outputs,'\n',alpha, prediction,'\n',"aaaaaaaaa", test_labels.data)
  print(test_labels.data)
  if test_labels.data==tensor([0]) and prediction.data==tensor([0]):
    print("the image belongs to correct folder (others) ")
  elif test_labels.data==tensor([1]) and prediction.data==tensor([1]):
    print("the image belongs to correct folder (rice)")
  elif test_labels.data==tensor([0]) and prediction.data==tensor([1]):
    print("the image belongs to other folder but predited as rice")
  elif test_labels.data==tensor([1]) and prediction.data==tensor([0]):
    print("the image belongs to rice folder but predited as other")
  test_acc_count += torch.sum(prediction == test_labels.data).item()
  print(test_acc_count)
print (test_acc_count/len(dataloaders['test']))
