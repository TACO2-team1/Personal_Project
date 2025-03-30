from pronets import PNet,RNet,ONet
from trainer import Trainer
import os

EPOCH_NUM=50

def train_pnet(epoches = EPOCH_NUM):
    print("PNet training")
    print("PNet training")
    print("PNet training")
    print("PNet training")
    print("PNet training")
    pnet = PNet()
    if not os.path.exists("./param/pnet"):
        os.makedirs("./param/pnet")

    if not os.path.exists("./logs/pnet"):
        os.makedirs("./logs/pnet")


    trainer=Trainer(pnet,"./Dataset/Train/12")
    trainer.test("./Dataset/Test/12","./logs/pnet/test_after_epoch0.txt")

    for epoch in range(epoches):
        trainer.train(epoch+1, "./param/pnet/epoch{}.pth".format(epoch+1),
            "./logs/pnet/epoch{}.txt".format(epoch+1))
        trainer.test("./Dataset/Test/12","./logs/pnet/test_after_epoch{}.txt".format(epoch+1))

def train_rnet(epoches = EPOCH_NUM):
    print("RNet training")
    print("RNet training")
    print("RNet training")
    print("RNet training")
    print("RNet training")
    rnet = RNet()
    if not os.path.exists("./param/rnet"):
        os.makedirs("./param/rnet")

    if not os.path.exists("./logs/rnet"):
        os.makedirs("./logs/rnet")


    trainer=Trainer(rnet,"./Dataset/Train/24")
    trainer.test("./Dataset/Test/24","./logs/rnet/test_after_epoch0.txt")

    for epoch in range(epoches):
        trainer.train(epoch+1, "./param/rnet/epoch{}.pth".format(epoch+1),
            "./logs/rnet/epoch{}.txt".format(epoch+1))
        trainer.test("./Dataset/Test/24","./logs/rnet/test_after_epoch{}.txt".format(epoch+1))

def train_onet(epoches = EPOCH_NUM):
    print("ONet training")
    print("ONet training")
    print("ONet training")
    print("ONet training")
    print("ONet training")
    onet = ONet()
    if not os.path.exists("./param/onet"):
        os.makedirs("./param/onet")

    if not os.path.exists("./logs/onet"):
        os.makedirs("./logs/onet")


    trainer=Trainer(onet,"./Dataset/Train/48")
    trainer.test("./Dataset/Test/48","./logs/onet/test_after_epoch0.txt")

    for epoch in range(epoches):
        trainer.train(epoch+1, "./param/onet/epoch{}.pth".format(epoch+1),
            "./logs/onet/epoch{}.txt".format(epoch+1))
        trainer.test("./Dataset/Test/48","./logs/onet/test_after_epoch{}.txt".format(epoch+1))


if __name__=='__main__':
    train_pnet()
    train_rnet()
    train_onet()