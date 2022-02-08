import numpy as np
# from numpy.core.fromnumeric import mean
import torch
import numpy as np
from torch import mean, tensor
from typing import List
import copy
from os.path import join
from .soft_dtw_cuda import SoftDTW
from .losses import DMPIntegrationMSE
from matplotlib import pyplot as plt
from datetime import datetime

torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROUND = 7

class Trainer:
    def __init__(self, model : torch.nn.Module, train_param, save_path = None, log_writer_path = None, writer = None):
        self.model = model
        self.train_param = train_param
        self.model_param = self.train_param.model_param
        self.output_mode = self.model_param.output_mode
        self.save_path = save_path
        self.LOG_WRITER_PATH = log_writer_path
        self.writer = writer
        self.epoch = 0
        
        self.loss_fns = []
        for loss_type in self.model_param.loss_type:
            if loss_type == 'MSE':
                self.loss_fns.append(torch.nn.MSELoss())
            elif loss_type == 'SDTW':
                self.loss_fns.append(SoftDTW(use_cuda=True, gamma=train_param.sdtw_gamma))
            elif loss_type == 'DMPIntegrationMSE':
                self.loss_fns.append(DMPIntegrationMSE(train_param = self.train_param))

        if self.train_param.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr = self.train_param.learning_rate, 
                                              weight_decay = self.train_param.weight_decay if self.train_param.weight_decay != None else 0, 
                                              eps = self.train_param.eps if self.train_param.eps != None else 1e-08,
                                              amsgrad = 0)

    def train(self, data_loaders : List[torch.utils.data.DataLoader]):
        train_loaders = data_loaders[0]
        val_loaders = data_loaders[1]

        self.train_param.writeLog('Loss Function : ' + str(self.loss_fns))
        self.train_param.writeLog('Optimizer : ' + str(self.optimizer))

        _, val_losses = self.getLosses(val_loaders, train = False)
        best_val_loss = mean(val_losses)

        self.train_param.writeLog('Initial Validation Loss : ' + str(best_val_loss) + '\n')

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
                    self.train_param.writeLog(log_str.format(self.epoch, 
                                                 self.mean_train_losses[-1], 
                                                 self.mean_val_losses[-1],
                                                 self.val_fail_count))
                    #  + ' | Validation Fail Count : ' + str(self.val_fail_count)
                else:
                    log_str = 'Epoch : {} | Train Loss : {:.'+str(ROUND)+'f}'
                    self.train_param.writeLog(log_str.format(self.epoch, 
                                                 self.mean_train_losses[-1]))
                    # self.writeLog('Epoch : ' + str(self.epoch) + ' | Training Loss : ' + str(round(self.mean_train_losses[-1], ROUND)))

                self.checkStoppingCondition()

            except KeyboardInterrupt:
                self.train = False
                self.train_param.writeLog('\nStopping Reason : Manual stop (Ctrl+C)')

        # Write Final logs
        self.writeTensorboardLogs()

        self.train_param.writeLog('Final Epoch = ' + str(self.epoch))
        self.train_param.writeLog('Final Validation Loss : ' + str(self.mean_val_losses[-1].item()))
        self.train_param.writeLog('Best Validation Loss : ' + str(best_val_loss.item()))
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

                preds = self.model(data)
                
                total_loss = torch.tensor(0.).to(DEVICE)
                for i in range(len(self.output_mode)):
                    loss_fn = self.loss_fns[i]
                    loss = loss_fn(preds[i], outputs[self.output_mode[i]])
                    if len(loss.shape) > 0:
                        loss = loss.mean()
                    total_loss = total_loss + loss

                total_loss.backward()
                self.optimizer.step()
                losses.append(total_loss)
                predictions.append(preds)
            self.epoch += 1
            losses = tensor(losses).to(DEVICE)
        else:
            self.model.eval()
            with torch.no_grad():
                for _, (data, outputs) in enumerate(data_loader):
                    self.optimizer.zero_grad()

                    preds = self.model(data)

                    if plot_comparison_idx != None:
                        if first_pred == None: first_pred = preds[0][plot_comparison_idx]
                        if first_label == None: first_label = outputs[self.output_mode[0]][plot_comparison_idx]
                        
                    total_loss = torch.tensor(0.).to(DEVICE)
                    for i in range(len(self.output_mode)):
                        loss_fn = self.loss_fns[i]
                        # print(preds[i].shape, outputs[self.output_mode[i]].shape)
                        loss = loss_fn(preds[i], outputs[self.output_mode[i]])
                        if len(loss.shape) > 0:
                            loss = loss.mean()
                        total_loss = total_loss + loss
                        
                    losses.append(total_loss)
                    predictions.append(preds)

                # plt.imshow(data['image'][0].detach().cpu().numpy().reshape(150, 150, 3))
                # plt.show()
                # print(torch.round(torch.clamp(preds[0], min = 1, max = 6)), outputs['num_segments'])
                # input('continue')

            losses = tensor(losses).to(DEVICE)

            if self.train_param.plot_interval != None and self.epoch % self.train_param.plot_interval == 0:
                self.plotTrajectory(outputs[self.output_mode[0]][:self.train_param.plot_num], preds[0][:self.train_param.plot_num])
            # print('Epoch', self.epoch, 'validation loss :',losses.mean())

            if plot_comparison_idx != None:
                self.plotTrajectory(first_label, first_pred)

        return predictions, losses

    def checkStoppingCondition(self):
        if self.train_param.max_epoch != None and self.epoch >= self.train_param.max_epoch:
            self.train = False
            self.train_param.writeLog('\nStopping Reason : Maximum epoch reached')
        if self.train_param.max_val_fail != None and self.val_fail_count >= self.train_param.max_val_fail:
            self.train = False
            self.train_param.writeLog('\nStopping Reason : Validation fail limit reached')

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
        base_size = 6
        fig, axs = plt.subplots(1, len(original_traj), figsize=(base_size*len(original_traj), base_size))
        if self.epoch != 0: title = 'Trajectory Reconstruction - Epoch ' + str(self.epoch)
        else: title = ''
        fig.suptitle(title)

        for i in range(len(original_traj)):
            output_np = original_traj[i].detach().cpu().numpy().reshape(-1, 2)
            pred_np = pred_traj[i].detach().cpu().numpy().reshape(-1, 2)
            # print(axs[0], axs[1], axs[2])
            if len(original_traj) > 1:
                axs[i].plot(output_np[:, 0], output_np[:, 1], color = 'blue')
                axs[i].scatter(output_np[:, 0], output_np[:, 1], color = 'blue')
                axs[i].plot(pred_np[:, 0], pred_np[:, 1], color = 'r', ls=':')
                axs[i].scatter(pred_np[:, 0], pred_np[:, 1], color = 'r')
                axs[i].scatter(pred_np[0, 0], pred_np[0, 1], color = 'g')
                # plt.title('Epoch ' + str(epoch) + ' | Loss = ' + str(loss))
                axs[i].set_xlim(-2, 8)
                axs[i].set_ylim(-2, 8)
                axs[i].legend(['original', 'dmp'])
            else:
                axs.plot(output_np[:, 0], output_np[:, 1], color = 'blue')
                axs.scatter(output_np[:, 0], output_np[:, 1], color = 'blue')
                axs.plot(pred_np[:, 0], pred_np[:, 1], color = 'r', ls=':')
                axs.scatter(pred_np[:, 0], pred_np[:, 1], color = 'r')
                axs.scatter(pred_np[0, 0], pred_np[0, 1], color = 'g')
                # plt.title('Epoch ' + str(epoch) + ' | Loss = ' + str(loss))
                axs.set_xlim(-2, 8)
                axs.set_ylim(-2, 8)
                axs.legend(['original', 'dmp'])
            # plt.axis('equal')
        plt.draw()
        plt.show(block=False)

        # original_traj = original_traj.cpu().numpy().reshape(-1, 2)
        # pred_traj = pred_traj.cpu().numpy().reshape(-1, 2)
        # plt.cla()
        # plt.clf()
        # plt.close('all')
        # plt.title("Trajectory Reconstruction - Epoch " + str(self.epoch))
        # plt.figure(1, figsize=(6, 6))
        # plt.axis("equal")
        # plt.plot(original_traj[:, 0], original_traj[:, 1], color = 'green')
        # plt.plot(pred_traj[:, 0], pred_traj[:, 1], color = 'red', linestyle = '--')
        # plt.draw()
        # plt.show(block=False)