import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from sal_model import BASNet

from data_loader import RescaleT, ToTensorLab, SalObjDataset

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def load_model(model_dir):
  net = BASNet(3, 1)
  net.load_state_dict(torch.load(model_dir))
  if torch.cuda.is_available():
    net.cuda()
  net.eval()
  return net

def get_output(image_path, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_path)
    image = image[:, :, :3]
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    return imo

def test_single(model, img_path):

    img_name = os.path.basename(img_path)
    img_name_list = list({img_path})
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                      transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)
    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d1, d2, d3, d4, d5, d6, d7, d8 = model(inputs_test)
        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        imo = get_output(img_path, pred)
        del d1, d2, d3, d4, d5, d6, d7, d8
        return imo



