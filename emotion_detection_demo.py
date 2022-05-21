import cv2
import numpy as np
from random import choice

import math
from time import sleep
import torch
import matplotlib.pyplot as plt

PATH = "C:\\Users\\veerb\\Downloads\\deep_learning\\project\\full_final_model.pt" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# creating audio device interface to control volume 
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = cast(interface, POINTER(IAudioEndpointVolume))

# defining classes to search for
CLASS_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral"
}

# creating model class in order to load saved model
class Model_base(torch.nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                 
        loss = torch.nn.functional.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                    
        loss = torch.nn.functional.cross_entropy(out, labels)   
        acc = accuracy(out, labels)         
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()     
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
def conv_block(in_chnl, out_chnl, padding=1):
    layers = [
        torch.nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding, stride=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_chnl),
        torch.nn.MaxPool2d(2),
        torch.nn.Dropout(0.4)]
    return torch.nn.Sequential(*layers)

class Model(Model_base):
    def __init__(self, in_chnls, num_cls):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_chnls, 256, kernel_size=3, padding=1)
        self.block1 = conv_block(256, 512)
        self.block2 = conv_block(512, 384)     
        self.block3 = conv_block(384, 192)
        self.block4 = conv_block(192, 384)

        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                        torch.nn.Linear(3456, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(256),
                                        torch.nn.Dropout(0.3),
                                        torch.nn.Linear(256, num_cls))    
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.block1(out)
        out = self.block2(out)       
        out = self.block3(out)
        out = self.block4(out)

        return self.classifier(out)

def get_class(value):
    return CLASS_MAP[value]


def main():
    img_shape = (48, 48)

    model = torch.load(PATH).to(device)


    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error opening video")

    while True:
        retreived, frame = cap.read()
        if not retreived:
            continue

        # displaying rectangle over captured area
        cv2.rectangle(frame, (75, 75), (300, 300), (0, 0, 255), 2)

        # capturing sub-frame
        capture_region = frame[75:300, 75:300]
        img = cv2.cvtColor(capture_region, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img = cv2.resize(img, img_shape)
        print(img.shape)
        img = torch.Tensor(np.array(img)/255.0)
        print(img.shape)

        
        
        img = img[None, None, :].to(device)
        
        # predicting Emotion
        print("img size: ", img.shape)
        pred = model(img)
        print(pred)
        move_code = get_class(np.argmax(pred.cpu().detach().numpy()[0]))
        
        # displaying recognized Emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Detected Emotion: " + move_code, (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Emotion Detected", frame)
        
        # defining command to quit (q)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()