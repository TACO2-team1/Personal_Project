import torch
import torch.nn as nn
import torch.optim as optim
import time

import utils
from face_dataset import FaceDataset


class Trainer:
    def __init__(self, net, dataset_path, batch_size=64):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.device = utils.try_gpu()
        self.net = net.to(self.device)
        self.cls_loss_func = nn.BCELoss()
        self.offset_loss_func = nn.MSELoss()
        self.landmark_loss_func = nn.MSELoss()
        
        self.optimizer = optim.Adam(self.net.parameters())
    
    def train(self, epoch, save_pth_path, log_path):
        self.net.train()
        log_file = open(log_path, "w")

        begin_time=time.time()

        faceDataset = FaceDataset(self.dataset_path)
        dataloader = torch.utils.data.DataLoader(faceDataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        for batch_idx,(img_data, cls, offset, landmark) in enumerate(dataloader):
            img_data = img_data.to(self.device)
            cls = cls.to(self.device)
            offset = offset.to(self.device)
            landmark = landmark.to(self.device)

            out_cls,out_offset,out_landmark = self.net(img_data)

            out_cls = out_cls.view(-1,1)
            out_offset = out_offset.view(-1,4)
            out_landmark = out_landmark.view(-1,10)

            # calc loss
            # eq:==,lt:<,gt:>,le:<=,ge:>=
            mask = torch.lt(cls, 2)
            masked_cls = torch.masked_select(cls, mask)
            masked_out_cls = torch.masked_select(out_cls, mask)
            cls_loss = self.cls_loss_func(masked_out_cls, masked_cls)

            mask = torch.min(torch.gt(cls, 0), torch.lt(cls, 3))
            masked_offset = torch.masked_select(offset, mask)
            masked_out_offset = torch.masked_select(out_offset, mask)
            offset_loss = self.offset_loss_func(masked_out_offset, masked_offset)

            mask = torch.gt(cls, 2)
            masked_landmark = torch.masked_select(landmark, mask)
            masked_out_landmark = torch.masked_select(out_landmark, mask)
            landmark_loss = self.landmark_loss_func(masked_out_landmark, masked_landmark)

            loss = cls_loss + offset_loss + landmark_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx%100==0:
                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                landmark_loss = landmark_loss.cpu().item()
                loss = loss.cpu().item()
                
                log_str='Train Epoch: {} [{}/{} ({:.0f}%)]    Loss: {:.6f} + {:.6f} + {:.6f} = {:.6f}    Time: {:.2f}s'.format(
                epoch, batch_idx * len(img_data), len(dataloader.dataset), 100.0*batch_idx * len(img_data) / len(dataloader.dataset),
                cls_loss, offset_loss, landmark_loss, loss, time.time() - begin_time)

                print(log_str)
                log_file.write(log_str)
                log_file.write("\n")
                log_file.flush()

        torch.save(self.net.state_dict(), save_pth_path)
        print("param saved")
    
    def test(self, dataset_path, log_path):
        self.net.eval()
        log_file = open(log_path, "w")

        faceDataset = FaceDataset(dataset_path)
        dataloader = torch.utils.data.DataLoader(faceDataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        test_cls_loss = 0
        test_cls_correct = 0
        test_cls_counter = 0
        test_offset_loss = 0
        test_landmark_loss = 0
        test_loss = 0
        counter = 0

        with torch.no_grad():
            for batch_idx,(img_data, cls, offset, landmark) in enumerate(dataloader):
                img_data = img_data.to(self.device)
                cls = cls.to(self.device)
                offset = offset.to(self.device)
                landmark = landmark.to(self.device)

                out_cls,out_offset,out_landmark = self.net(img_data)

                out_cls = out_cls.view(-1,1)
                out_offset = out_offset.view(-1,4)
                out_landmark = out_landmark.view(-1,10)

                # calc loss
                # eq:==,lt:<,gt:>,le:<=,ge:>=
                mask = torch.lt(cls, 2)
                masked_cls = torch.masked_select(cls, mask)
                masked_out_cls = torch.masked_select(out_cls, mask)
                cls_loss = self.cls_loss_func(masked_out_cls, masked_cls)

                cls_minus_abs = (masked_cls-masked_out_cls).abs().lt(0.5)
                test_cls_correct += cls_minus_abs.sum().cpu().item()
                test_cls_counter += cls_minus_abs.shape[0]

                mask = torch.min(torch.gt(cls, 0), torch.lt(cls, 3))
                masked_offset = torch.masked_select(offset, mask)
                masked_out_offset = torch.masked_select(out_offset, mask)
                offset_loss = self.offset_loss_func(masked_out_offset, masked_offset)

                mask = torch.gt(cls, 2)
                masked_landmark = torch.masked_select(landmark, mask)
                masked_out_landmark = torch.masked_select(out_landmark, mask)
                landmark_loss = self.landmark_loss_func(masked_out_landmark, masked_landmark)

                loss = cls_loss + offset_loss + landmark_loss

                test_cls_loss += cls_loss.cpu().item()
                test_offset_loss += offset_loss.cpu().item()
                test_landmark_loss += landmark_loss.cpu().item()
                test_loss += loss.cpu().item()
                counter += 1

        test_cls_loss /= counter
        test_offset_loss /= counter
        test_landmark_loss /= counter
        test_loss /= counter
        log_str="Acc: ({:4f}%)    Loss: {:.6f} + {:.6f} + {:.6f} = {:.6f}".format(100.0*test_cls_correct/test_cls_counter,
            test_cls_loss,test_offset_loss,test_landmark_loss,test_loss)
        
        print(log_str)
        log_file.write(log_str)
        log_file.write("\n")
        log_file.flush()
