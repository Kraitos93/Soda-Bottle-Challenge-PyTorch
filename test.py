import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models 
from cnn_architecture import CNN
from dataloader import bottle_test,data_split
import sys

def get_accuracy(model,dataloader):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in dataloader:
			images, labels = data
			images,labels = images.to(device), labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return (100. * correct / total)

def define_loader(dataset_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    data_test_transform = transforms.Compose([transforms.Resize(size = (224,224)), transforms.ToTensor(), normalize])
    soda_bottles_test = bottle_test('samples.csv', data_test_transform, path=dataset_path)
    test_loader = torch.utils.data.DataLoader(dataset=soda_bottles_test, batch_size=32, shuffle=True, num_workers=4)
    return test_loader


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Testing on ",device)
    args = sys.argv
    loader = define_loader(args[1])
    mode = args[2]
    soda_model = CNN()
    soda_model.load_state_dict(torch.load('SGD_soda_model_%s.pkl' % (mode)))
    soda_model.eval()
    soda_model.to(device)
    test_acc = get_accuracy(soda_model,loader)
    print("Test Accuracy: ",test_acc)
