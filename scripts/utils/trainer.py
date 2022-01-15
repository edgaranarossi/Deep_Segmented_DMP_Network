import numpy as np
# from numpy.core.fromnumeric import mean
import torch
import numpy as np
from torch import round, mean, squeeze, empty, tensor, clamp
from typing import List
import copy
from os.path import join
from .soft_dtw_cuda import SoftDTW
from matplotlib import pyplot as plt
from datetime import datetime

torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROUND = 7

class Trainer:
    def __init__(self, model : torch.nn.Module, train_param, save_path = None, log_writer_path = None, writer = None):
        self.model = model
        self.train_param = train_param
        self.save_path = save_path
        self.LOG_WRITER_PATH = log_writer_path
        self.writer = writer
        self.epoch = 0
        
        if self.train_param.loss_type == 'MSE':
            self.loss_fn = torch.nn.MSELoss()
        elif self.train_param.loss_type == 'SDTW':
            self.loss_fn = SoftDTW(use_cuda=True, gamma=0.1)

        if self.train_param.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr = self.train_param.learning_rate, 
                                              weight_decay = self.train_param.weight_decay if self.train_param.weight_decay != None else 0, 
                                              eps = self.train_param.eps if self.train_param.eps != None else 1e-08)

    def train(self, data_loaders : List[torch.utils.data.DataLoader]):
        train_loaders = data_loaders[0]
        val_loaders = data_loaders[1]

        self.writeLog('Loss Function : ' + str(self.loss_fn))
        self.writeLog('Optimizer : ' + str(self.optimizer))

        _, val_losses = self.getLosses(val_loaders, train = False)
        best_val_loss = mean(val_losses)

        self.writeLog('Initial Validation Loss : ' + str(best_val_loss) + '\n')

        self.train = True
        self.val_fail_count = 0
        self.epoch = 0
        self.mean_train_losses = []
        self.mean_val_losses = []

        while self.train:
            try:
                # Train
                _, train_losses = self.getLosses(train_loaders)
                # print(train_losses)
                # print(train_losses)
                self.mean_train_losses.append(mean(train_losses))

                # Validate
                if self.epoch % self.train_param.validation_interval == 0: 
                    _, val_losses = self.getLosses(val_loaders, train = False)
                    self.mean_val_losses.append(mean(val_losses))

                # Check Validation Loss
                if self.mean_val_losses[-1] > best_val_loss:
                    self.val_fail_count += 1
                else:
                    best_val_loss = self.mean_val_losses[-1]
                    self.val_fail_count = 0
                    if self.save_path != None: torch.save(self.model.state_dict(), join(self.save_path, 'best_net_parameters'))

                # Write Training logs
                if self.epoch % self.train_param.log_interval == 0:
                    self.writeTensorboardLogs()

                if self.epoch % self.train_param.validation_interval == 0: 
                    # self.writeLog('Epoch : ' + str(self.epoch) + ' | Training Loss : ' + str(round(self.mean_train_losses[-1], ROUND)) + ' | Validation Loss : ' + str(round(self.mean_val_losses[-1], ROUND)))
                    log_str = 'Epoch : {} | Train Loss : {:.'+str(ROUND)+'f} | Val. Loss : {:.'+str(ROUND)+'f} | Val. Fail : {}'
                    self.writeLog(log_str.format(self.epoch, 
                                                 self.mean_train_losses[-1], 
                                                 self.mean_val_losses[-1],
                                                 self.val_fail_count))
                    #  + ' | Validation Fail Count : ' + str(self.val_fail_count)
                else:
                    log_str = 'Epoch : {} | Train Loss : {:.'+str(ROUND)+'f}'
                    self.writeLog(log_str.format(self.epoch, 
                                                 self.mean_train_losses[-1]))
                    # self.writeLog('Epoch : ' + str(self.epoch) + ' | Training Loss : ' + str(round(self.mean_train_losses[-1], ROUND)))

                self.checkStoppingCondition()

            except KeyboardInterrupt:
                self.train = False
                self.writeLog('\nStopping Reason : Manual stop (Ctrl+C)')

        # Write Final logs
        self.writeTensorboardLogs()

        self.writeLog('Final Epoch = ' + str(self.epoch))
        self.writeLog('Final Validation Loss : ' + str(self.mean_val_losses[-1].item()))
        self.writeLog('Best Validation Loss : ' + str(best_val_loss.item()))
        if self.save_path != None: torch.save(self.model.state_dict(), join(self.save_path, 'final_net_parameters'))

        print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), "Training finished")

    def test(self, data_loaders : List[torch.utils.data.DataLoader], plot_comparison_idx = None):
        test_loaders = data_loaders[2]
        _, test_losses = self.getLosses(test_loaders, train = False, plot_comparison_idx = plot_comparison_idx)
        print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), 'Test Loss :', mean(test_losses).item())

    def getLosses(self, data_loader, train = True, plot_comparison_idx = None):
        predictions = []
        losses = []

        first_pred = None
        first_label = None
        # epoch_loss = torch.Tensor([0]).to(DEVICE)
        if train:
            self.model.train()
            for _, (data, outputs) in enumerate(data_loader):
                self.optimizer.zero_grad()
                outputs = outputs.squeeze(axis=1)
                preds = self.model(data).squeeze()
                # preds = clamp(preds, min=0, max = 250)
                loss = self.loss_fn(preds, outputs).mean()
                batch_loss = mean(loss).to(DEVICE)
                # print('Epoch', self.epoch, 'batch loss :',loss)
                loss.backward()
                self.optimizer.step()
                losses.append(batch_loss)
                # epoch_loss = epoch_loss + batch_loss
                predictions.append(preds)
            self.epoch += 1
            losses = tensor(losses).to(DEVICE)
        else:
            self.model.eval()
            with torch.no_grad():
                for _, (data, outputs) in enumerate(data_loader):
                    self.optimizer.zero_grad()
                    outputs = outputs.squeeze(axis=1)
                    preds = self.model(data).squeeze()
                    if plot_comparison_idx != None:
                        if first_pred == None: first_pred = preds[plot_comparison_idx]
                        if first_label == None: first_label = outputs[plot_comparison_idx]
                    # preds = clamp(preds, min=-50, max = 50)
                    # print(preds.shape, outputs.shape)
                    loss = self.loss_fn(preds, outputs)
                    batch_loss = mean(loss).to(DEVICE)
                    losses.append(batch_loss)
                    # epoch_loss = epoch_loss + batch_loss
                    predictions.append(preds)
            losses = tensor(losses).to(DEVICE)

            if self.train_param.plot_interval != None and self.epoch % self.train_param.plot_interval == 0:
                self.plotTrajectory(outputs[0], preds[0])
            # print('Epoch', self.epoch, 'validation loss :',losses.mean())

            if plot_comparison_idx != None:
                self.plotTrajectory(first_label, first_pred)

        return predictions, losses

    def checkStoppingCondition(self):
        if self.train_param.max_epoch != None and self.epoch > self.train_param.max_epoch:
            self.train = False
            self.writeLog('\nStopping Reason : Maximum epoch reached')
        if self.train_param.max_val_fail != None and self.val_fail_count > self.train_param.max_val_fail:
            self.train = False
            self.writeLog('\nStopping Reason : Validation fail limit reached')

    def writeLog(self, log):
        if log[0] == '\n':
            print('\n', datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), log[1:])
        else:
            print(datetime.now().strftime("%Y-%m-%d_%H-%M-%S ::"), log)
        if self.LOG_WRITER_PATH != None:
            LOG_FILE = open(self.LOG_WRITER_PATH, 'a')
            LOG_FILE.write('\n' + log)
            LOG_FILE.close()

    def writeTensorboardLogs(self):
        if self.writer != None: 
            self.writer.add_scalar('data/train_loss', self.mean_train_losses[-1], self.epoch)
            self.writer.add_scalar('data/val_loss', self.mean_val_losses[-1], self.epoch)
            if self.epoch == 1:
                self.old_train_loss = self.mean_train_losses[-1]
                self.old_val_loss = self.mean_val_losses[-1]
            else:
                self.old_train_loss = self.mean_train_losses[-2]
                self.old_val_loss = self.mean_val_losses[-2]

            self.writer.add_scalar('data/train_loss_grad', (self.mean_train_losses[-1] - self.old_train_loss) / self.train_param.log_interval, self.epoch)
            self.writer.add_scalar('data/val_loss_grad', (self.mean_val_losses[-1] - self.old_val_loss) / self.train_param.log_interval, self.epoch)
            self.writer.add_scalar('data/val_fail_count', self.val_fail_count, self.epoch)

    def plotTrajectory(self, original_traj, pred_traj):
        original_traj = original_traj.cpu().numpy().reshape(-1, 2)
        pred_traj = pred_traj.cpu().numpy().reshape(-1, 2)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.title("Trajectory Reconstruction - Epoch " + str(self.epoch))
        plt.figure(1, figsize=(6, 6))
        plt.axis("equal")
        plt.plot(original_traj[:, 0], original_traj[:, 1], color = 'green')
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], color = 'red', linestyle = '--')
        plt.draw()
        plt.show(block=False)