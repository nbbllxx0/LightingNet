from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os

class dehaze_val_dataset(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort()
        self.root_hazy = test_dir
        self.file_len = len(self.list_test)


    def __getitem__(self, index, is_train=True):
        LOW = Image.open(self.root_hazy +'/'+ self.list_test[index])
        LOW = self.transform(LOW)

#
        return LOW

    def __len__(self):
        return self.file_len

