# from numpy.core.fromnumeric import mean
import torch
import numpy as np
from torch import mean, tensor, clone, zeros, ones, cat, clamp
from typing import List
import copy
from copy import copy, deepcopy
from os.path import join
from .soft_dtw_cuda import SoftDTW
from .losses import DMPIntegrationMSE
from matplotlib import pyplot as plt
from datetime import datetime
import psutil
from .pydmps_torch import DMPs_discrete_torch
from PIL import Image, ImageOps
from multiprocessing import Process
from bagpy import create_fig

torch.autograd.set_detect_anomaly(True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROUND = 8

class Trainer:
    def __init__(self, model : torch.nn.Module, train_param, save_path = None, log_writer_path = None, writer = None):
        self.model = model
        self.train_param = train_param
        self.memory_limit = self.train_param.memory_percentage_limit
        self.model_param = self.train_param.model_param
        self.loss_name = self.model_param.loss_name
        self.dmp_param   = self.model_param.dmp_param
        self.input_mode = self.model_param.input_mode
        self.output_mode = self.model_param.output_mode
        self.save_path = save_path
        self.LOG_WRITER_PATH = log_writer_path
        self.writer = writer
        self.epoch = 0
        self.best_state = None
        self.best_val_loss_separate = None
        self.best_val_loss_epoch = None
        
        self.loss_fns = []
        for loss_type in self.model_param.loss_type:
            if loss_type == 'MSE':
                self.loss_fns.append(torch.nn.MSELoss())
            elif loss_type == 'SDTW':
                self.loss_fns.append(SoftDTW(use_cuda=True, gamma=train_param.sdtw_gamma))
            elif loss_type == 'DMPIntegrationMSE':
                self.loss_fns.append(DMPIntegrationMSE(train_param = self.train_param))

        self.scaler = self.train_param.scaler
            

        if self.train_param.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                              lr = self.train_param.learning_rate, 
                                              weight_decay = self.train_param.weight_decay if self.train_param.weight_decay != None else 0, 
                                              eps = self.train_param.eps if self.train_param.eps != None else 1e-08,
                                              amsgrad = 0)
        # input('init complete')

    def train(self, data_loaders : List[torch.utils.data.DataLoader], show_tau_error = True):
        self.show_tau_error = show_tau_error
        train_loaders = data_loaders[0]
        val_loaders = data_loaders[1]
        test_loaders = data_loaders[2]
        test_dataset = test_loaders.dataset

        self.train_param.writeLog('Loss Function   : ' + str(self.loss_fns))
        self.train_param.writeLog('Optimizer       : ' + str(self.optimizer))

        _, val_losses = self.getLosses(val_loaders, train = False)
        self.best_val_loss = mean(val_losses)
        self.best_val_loss_epoch = copy(self.epoch)
        self.best_val_loss_separate = self.loss_separate

        self.train_param.writeLog('Initial Validation Loss : ' + str(np.round(self.best_val_loss.item(), 5)) + '\n')

        self.training = True
        self.val_fail_count = 0
        self.epoch = 0
        self.step = 0
        self.mean_train_losses = []
        self.mean_val_losses = []

        while self.training:
            try:
                # Train
                # input('train #3')
                _, train_losses = self.getLosses(train_loaders)
                # print(train_losses)
                # print(train_losses)
                if not self.show_tau_error: train_losses = train_losses[:-1]
                self.mean_train_losses.append(mean(train_losses))

                # Validate
                if self.epoch % self.train_param.validation_interval == 0: 
                    _, val_losses = self.getLosses(val_loaders, train = False)
                    if not self.show_tau_error: val_losses = val_losses[:-1]
                    self.mean_val_losses.append(mean(val_losses))

                # Check Validation Loss
                if self.mean_val_losses[-1] > self.best_val_loss:
                    self.val_fail_count += 1
                else:
                    self.best_val_loss = self.mean_val_losses[-1]
                    self.best_val_loss_epoch = copy(self.epoch)
                    self.best_val_loss_separate = copy(self.loss_separate)
                    self.val_fail_count = 0
                    self.best_state = deepcopy(self.model.state_dict())
                    
                if self.epoch % self.train_param.model_save_interval == 0: 
                    torch.save(self.best_state, join(self.save_path, 'best_net_parameters'))

                # Write Training logs
                if self.epoch % self.train_param.log_interval == 0:
                    self.writeTensorboardLogs()

                if self.epoch % self.train_param.validation_interval == 0: 
                    # self.writeLog('Epoch : ' + str(self.epoch) + ' | Training Loss : ' + str(round(self.mean_train_losses[-1], ROUND)) + ' | Validation Loss : ' + str(round(self.mean_val_losses[-1], ROUND)))
                    log_str = 'Epoch : {} | Step : {} | Train Loss : {:.'+str(ROUND)+'f} | Val. Loss : {:.'+str(ROUND)+'f} | Val. Fail : {}'
                    self.train_param.writeLog(log_str.format(self.epoch,
                                                             self.step,
                                                             self.mean_train_losses[-1], 
                                                             self.mean_val_losses[-1],
                                                             self.val_fail_count))

                    self.train_param.writeLog(self.train_log)
                    self.train_param.writeLog(self.val_log)

                    log_str = 'Best Val. Loss Separate'
                    for i in range(len(self.best_val_loss_separate)):
                        if (self.loss_name[i] == 'tau' and self.show_tau_error) or self.loss_name[i] != 'tau':
                            log_str += ' | ' + self.loss_name[i] + ': ' + str(np.round(self.best_val_loss_separate[i], ROUND))
                    self.train_param.writeLog(log_str)

                    log_str = 'Best Validation Loss    | {:.'+str(ROUND)+'f}\n'
                    self.train_param.writeLog(log_str.format(self.best_val_loss))
                    #  + ' | Validation Fail Count : ' + str(self.val_fail_count)
                else:
                    log_str = 'Epoch : {} | Step : {} | Train Loss : {:.'+str(ROUND)+'f}\n'
                    self.train_param.writeLog(log_str.format(self.epoch, 
                                                             self.step,
                                                             self.mean_train_losses[-1]))
                    # self.writeLog('Epoch : ' + str(self.epoch) + ' | Training Loss : ' + str(round(self.mean_train_losses[-1], ROUND)))

                self.checkStoppingCondition()

            except KeyboardInterrupt:
                self.training = False
                self.train_param.writeLog('\nStopping Reason : Manual stop (Ctrl+C)')

        # Write Final logs
        self.writeTensorboardLogs()

        self.train_param.writeLog('Final Epoch = ' + str(self.epoch))
        self.train_param.writeLog('Final Validation Loss : ' + str(self.mean_val_losses[-1].item()))
        self.train_param.writeLog('Best Validation Loss : ' + str(self.best_val_loss.item()))
        if self.save_path != None: 
            print('Saving final and best model')
            torch.save(self.model.state_dict(), join(self.save_path, 'final_net_parameters'))
            torch.save(self.best_state, join(self.save_path, 'best_net_parameters'))

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
            # print('training')
            self.model.train()
            self.loss_separate = [[] for i in self.output_mode]
            for _, (data, outputs) in enumerate(data_loader):
                self.optimizer.zero_grad()

                preds = self.model(data)                
                total_loss = torch.tensor(0.).to(DEVICE)
                # loss_str = 'Train Loss'
                for i in range(len(self.output_mode)):
                    loss_fn = self.loss_fns[i]
                    # print(preds[i], outputs[self.output_mode[i]])
                    loss = loss_fn(preds[i], outputs[self.output_mode[i]])
                    if len(loss.shape) > 0:
                        loss = loss.mean()
                    self.loss_separate[i].append(loss.item())
                    # loss_str += ' | L' + str(i+1) + ': ' + str(np.round(loss.item(), 5).item())
                    total_loss = total_loss + loss
                # print(loss_str)
                total_loss.backward()
                # print('backwarded')
                self.optimizer.step()
                self.step += 1
                # print('stepped')
                losses.append(total_loss)
                predictions.append(preds)
            self.loss_separate = np.array(self.loss_separate).mean(axis = 1)
            self.train_log = 'Train Loss             '
            for i in range(len(self.loss_separate)):
                if (self.loss_name[i] == 'tau' and self.show_tau_error) or self.loss_name[i] != 'tau':
                    self.train_log += ' | ' + self.loss_name[i] + ': ' + str(np.round(self.loss_separate[i], ROUND))
            self.epoch += 1
            losses = tensor(losses).to(DEVICE)
        else:
            # print('pre-eval')
            self.model.eval()
            self.loss_separate = [[] for i in self.output_mode]
            # print('eval-mode')
            with torch.no_grad():
                for _, (data, outputs) in enumerate(data_loader):
                    self.optimizer.zero_grad()

                    preds = self.model(data)
                    total_loss = torch.tensor(0.).to(DEVICE)
                    # loss_str = 'Val Loss'
                    for i in range(len(self.output_mode)):
                        loss_fn = self.loss_fns[i]
                        # print(preds[i], outputs[self.output_mode[i]])
                        loss = loss_fn(preds[i], outputs[self.output_mode[i]])
                        if len(loss.shape) > 0:
                            loss = loss.mean()
                        self.loss_separate[i].append(loss.item())
                        # loss_str += ' | L' + str(i+1) + ': ' + str(np.round(loss.item(), 5))
                        total_loss = total_loss + loss
                    # print(loss_str)

                    if plot_comparison_idx != None:
                        if first_pred == None: 
                            first_pred = preds[0][plot_comparison_idx]
                        if first_label == None: 
                            first_label = outputs[self.output_mode[0]][plot_comparison_idx]
                        
                    losses.append(total_loss)
                    predictions.append(preds)
                self.loss_separate = np.array(self.loss_separate).mean(axis = 1)
                self.val_log = 'Val Loss               '
                for i in range(len(self.loss_separate)):
                    if (self.loss_name[i] == 'tau' and self.show_tau_error) or self.loss_name[i] != 'tau':
                        self.val_log += ' | ' + self.loss_name[i] + ': ' + str(np.round(self.loss_separate[i], ROUND))
                # plt.imshow(data['image'][0].detach().cpu().numpy().reshape(150, 150, 3))
                # plt.show()
                # print(torch.round(torch.clamp(preds[0], min = 1, max = 6)), outputs['num_segments'])
                # input('continue')

            losses = tensor(losses).to(DEVICE)
            
            # print('Plotting')
            if self.train_param.plot_interval != None and \
               self.epoch % self.train_param.plot_interval == 0:
                if self.model_param.model_type in ['DSDNetV1']:
                    if self.dmp_param.dof == 2:
                        img = data['image'][0].detach().cpu().numpy().reshape(self.model_param.image_dim[1],
                                                                    self.model_param.image_dim[2],
                                                                    self.model_param.image_dim[0])
                        img = np.flipud(img)
                        rescaled_pred = []
                        rescaled_label = []
                        for idx, key in enumerate(self.model_param.output_mode):
                            if key in self.model_param.keys_to_normalize:
                                rescaled_pred.append(self.scaler[key].denormalize(preds[idx][0]))
                                rescaled_label.append(self.scaler[key].denormalize(outputs[self.output_mode[idx]][0]))
                            else:
                                rescaled_pred.append(preds[idx][0])
                                rescaled_label.append(outputs[self.output_mode[idx]][0])

                        num_segments_pred = int(clamp(torch.round(rescaled_pred[0]).reshape(1), max = self.model_param.max_segments).item())
                        num_segments_label = int(clamp(torch.round(rescaled_label[0]).reshape(1), max = self.model_param.max_segments).item())
                        y_label = zeros(num_segments_label, int(1 / self.dmp_param.dt), self.dmp_param.dof).to(DEVICE)
                        y_pred = zeros(num_segments_pred, int(1 / self.dmp_param.dt), self.dmp_param.dof).to(DEVICE)

                        all_pos_pred = cat([rescaled_pred[1].reshape(1, self.dmp_param.dof, 1), rescaled_pred[2].reshape(-1, self.dmp_param.dof, 1)], dim = 0)
                        all_pos_label = cat([rescaled_label[1].reshape(1, self.dmp_param.dof, 1), rescaled_label[2].reshape(-1, self.dmp_param.dof, 1)], dim = 0)

                        y0s_label = all_pos_label[:-1]
                        y0s_pred = all_pos_pred[:-1]
                        goals_label = all_pos_label[1:]
                        goals_pred = all_pos_pred[1:]

                        dmp_label = DMPs_discrete_torch(n_dmps = self.dmp_param.dof, 
                                                        n_bfs = self.dmp_param.n_bf, 
                                                        ay = self.dmp_param.ay, 
                                                        dt = self.dmp_param.dt)
                        # dmp_label.y0 = rescaled_label[1].reshape(1, self.dmp_param.dof, 1)
                        dmp_label.y0        = y0s_label[:num_segments_label]
                        dmp_label.goal      = goals_label[:num_segments_label]
                        dmp_label.w         = rescaled_label[3][:num_segments_label].reshape(num_segments_label, self.dmp_param.dof, self.dmp_param.n_bf)
                        y_track_label, _, _ = dmp_label.rollout()

                        dmp_pred = DMPs_discrete_torch(n_dmps = self.dmp_param.dof, 
                                                    n_bfs = self.dmp_param.n_bf, 
                                                    ay = self.dmp_param.ay, 
                                                    dt = self.dmp_param.dt)
                        # dmp_pred.y0 = rescaled_pred[1].reshape(1, self.dmp_param.dof, 1)
                        dmp_pred.y0         = y0s_pred[:num_segments_pred]
                        dmp_pred.goal       = goals_pred[:num_segments_pred]
                        dmp_pred.w          = rescaled_pred[3][:num_segments_pred].reshape(num_segments_pred, self.dmp_param.dof, self.dmp_param.n_bf)
                        y_track_pred, _, _  = dmp_pred.rollout()

                        # for i in range(num_segments_label):
                        #     dmp_label.goal      = rescaled_label[2][i].reshape(1, self.dmp_param.dof, 1)
                        #     dmp_label.w         = rescaled_label[3][i].reshape(1, self.dmp_param.dof, self.dmp_param.n_bf)
                        #     y_track_label, _, _ = dmp_label.rollout()
                        #     y_label[i]          = y_track_label.reshape(-1, self.dmp_param.dof)
                        #     dmp_label.y0        = y_label[i, -1].reshape(1, self.dmp_param.dof, 1)

                        #     if i < num_segments_pred:
                        #         dmp_pred.goal       = rescaled_pred[2][i].reshape(1, self.dmp_param.dof, 1)
                        #         dmp_pred.w          = rescaled_pred[3][i].reshape(1, self.dmp_param.dof, self.dmp_param.n_bf)
                        #         y_track_pred, _, _  = dmp_pred.rollout()
                        #         y_pred[i]           = y_track_pred.reshape(-1, self.dmp_param.dof)
                        #         dmp_pred.y0         = y_pred[i, -1].reshape(1, self.dmp_param.dof, 1)

                        y_label = y_track_label.reshape(-1, self.dmp_param.dof)
                        y_pred = y_track_pred.reshape(-1, self.dmp_param.dof)

                        padding = 6
                        multiplier = 50
                        y_label = ((y_label.detach().cpu().numpy() * multiplier) + padding).reshape(-1, self.dmp_param.dof)
                        y_pred = ((y_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, self.dmp_param.dof)
                        all_pos_pred_np = ((all_pos_pred.detach().cpu().numpy() * multiplier) + padding).reshape(-1, self.dmp_param.dof)

                        plt.scatter(all_pos_pred_np[:num_segments_pred + 1, 0], all_pos_pred_np[:num_segments_pred + 1, 1], c = 'c', zorder = 6)
                        # print(img.shape)
                        self.plot(y_pred, y_label, img = img)

                else:
                    # self.plotTrajectory(outputs[self.output_mode[0]][:self.train_param.plot_num], preds[0][:self.train_param.plot_num])
                    pass
            # print('Epoch', self.epoch, 'validation loss :',losses.mean())

            if plot_comparison_idx != None:
                self.plotTrajectory(first_label, first_pred)

        return predictions, losses

    def plot(self, y_pred, y_label, img = None):
        if img.any() != None: plt.imshow(img, origin = 'lower', cmap='gray')
        plt.plot(y_label[:, 0], y_label[:, 1], lw = 5, c = 'g')
        plt.scatter(y_pred[:, 0], y_pred[:, 1], c = 'r', zorder = 5)
        plt.title('Epoch ' + str(self.epoch))
        plt.show()

    def checkStoppingCondition(self):
        if psutil.virtual_memory().percent > self.memory_limit:
            self.training = False
            self.train_param.writeLog('\nStopping Reason : Out of Memory (>{}%)'.format(self.memory_limit))
        if self.train_param.max_epoch != None and self.epoch >= self.train_param.max_epoch:
            self.training = False
            self.train_param.writeLog('\nStopping Reason : Maximum epoch reached')
        if self.train_param.max_val_fail != None and self.val_fail_count >= self.train_param.max_val_fail:
            self.training = False
            self.train_param.writeLog('\nStopping Reason : Validation fail limit reached')
        if self.mean_val_losses[-1] < self.train_param.loss_threshold:
            self.training = False
            self.train_param.writeLog('\nStopping Reason : Loss threshold exceeded')

    def writeTensorboardLogs(self):
        if self.writer != None: 
            # print('tensorboard logging')
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
        fig, axs = plt.subplots(2, len(original_traj), figsize=(base_size*len(original_traj), 2 * base_size))
        if self.epoch != 0: title = 'Trajectory Reconstruction - Epoch ' + str(self.epoch)
        else: title = ''
        fig.suptitle(title)

        for i in range(len(original_traj)):
            output_np = original_traj[i].detach().cpu().numpy().reshape(-1, 2)
            pred_np = pred_traj[i].detach().cpu().numpy().reshape(-1, 2)
            for j in range(2):
                # print(axs[0], axs[1], axs[2])
                if len(original_traj) > 1:
                    axs[j][i].plot(output_np[:, 0], output_np[:, 1], color = 'blue')
                    axs[j][i].scatter(output_np[:, 0], output_np[:, 1], color = 'blue')
                    axs[j][i].plot(pred_np[:, 0], pred_np[:, 1], color = 'r', ls=':')
                    axs[j][i].scatter(pred_np[:, 0], pred_np[:, 1], color = 'r')
                    axs[j][i].scatter(pred_np[0, 0], pred_np[0, 1], color = 'g')
                    axs[j][i].scatter(pred_np[-1, 0], pred_np[-1, 1], color = 'b')
                    # plt.title('Epoch ' + str(epoch) + ' | Loss = ' + str(loss))
                    if j == 0:
                        axs[j][i].set_xlim(-2, 8)
                        axs[j][i].set_ylim(-2, 8)
                    axs[j][i].legend(['original', 'dmp'])
                else:
                    axs[j].plot(output_np[:, 0], output_np[:, 1], color = 'blue')
                    axs[j].scatter(output_np[:, 0], output_np[:, 1], color = 'blue')
                    axs[j].plot(pred_np[:, 0], pred_np[:, 1], color = 'r', ls=':')
                    axs[j].scatter(pred_np[:, 0], pred_np[:, 1], color = 'r')
                    axs[j].scatter(pred_np[0, 0], pred_np[0, 1], color = 'g')
                    axs[j].scatter(pred_np[-1, 0], pred_np[-1, 1], color = 'b')
                    # plt.title('Epoch ' + str(epoch) + ' | Loss = ' + str(loss))
                    if j == 0:
                        axs[j].set_xlim(-2, 8)
                        axs[j].set_ylim(-2, 8)
                    axs[j].legend(['original', 'dmp'])
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