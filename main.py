#This is a binary classification task.

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import logging
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import sklearn
from config import get_config
from data import load_dataset
from model import AlexNet, GoogleNet, ResNet50, VGG16
from torch.optim.lr_scheduler import StepLR

INDEX = 0

class ddl:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.train_batch_size = args.train_batch_size
        self.workers = args.workers

        # Initialize the model
        self.Mymodel = self._initialize_model(args.model_name)
        self.Mymodel.to(args.device)

        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _initialize_model(self, model_name):
        models = {
            'AlexNet': AlexNet,
            'GoogleNet': GoogleNet,
            'ResNet50': ResNet50,
            'VGG16': VGG16,
        }
        if model_name not in models:
            raise ValueError('Unknown model name: {}'.format(model_name))
        model = models[model_name]()
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)  # Xavier
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        return model

    def _print_args(self):
        self.logger.info('> arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer):
        self.args.index += 1
        total_loss, total_correct, samples = 0, 0, 0

        self.Mymodel.train()

        for inputs, targets, _ in tqdm(dataloader, disable=self.args.backend, ascii='>='):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            optimizer.zero_grad()

            outputs = self.Mymodel(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            samples += targets.size(0)

        avg_loss = total_loss / samples
        accuracy = total_correct / samples

        return avg_loss, accuracy
    
    def _test(self, dataloader, criterion):
        test_loss, total_correct, samples = 0, 0, 0
        image_paths = []
        all_targets, all_predicts, all_scores = [], [], []

        self.Mymodel.eval()

        with torch.no_grad():
            for inputs, targets, paths in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                predicts = self.Mymodel(inputs)
                loss = criterion(predicts, targets)

                test_loss += loss.item() * targets.size(0)
                total_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                samples += targets.size(0)

                all_targets.extend(targets.cpu().numpy())
                all_predicts.extend(torch.argmax(predicts, dim=1).cpu().numpy())
                all_scores.extend(predicts.cpu().numpy())
                image_paths.extend(paths)

        test_acc = total_correct / samples
        metrics = {
            'roc_auc': roc_auc_score(all_targets, all_predicts, multi_class='ovr'),
            'recall': recall_score(all_targets, all_predicts, average='macro'),
            'precision': precision_score(all_targets, all_predicts, average='macro', zero_division=0),
            'f1': f1_score(all_targets, all_predicts, average='macro')
        }

        with open('./result/' + self.args.model_name + 'predictions.txt', 'w') as f:
            for path, target, predict in zip(image_paths, all_targets, all_predicts):
                f.write(f'{path}, target: {target}, predict: {predict}\n')

        return test_loss / samples, test_acc, metrics['recall'], metrics['precision'], metrics['f1'], all_targets, all_scores

    def run(self):
        # Print the parameters of model
        for name, layer in self.Mymodel.named_parameters(recurse=True):
            print(name, layer.shape, sep=" ")

        train_dataloader, test_dataloader = load_dataset(self)
        params = filter(lambda x: x.requires_grad, self.Mymodel.parameters())
        
        # CrossEntropyLoss is chosen as the loss function
        criterion = nn.CrossEntropyLoss() 

        # AdamW is chosen as the optimizer
        optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        index = 0
        l_trloss,l_acc,l_precision,l_recall,l_f1,l_epo = [],[],[],[],[],[]

        # Calculate the parameters of model
        best_loss, best_acc, best_precision, best_recall, best_f1 = 0, 0, 0, 0, 0
        for epoch in range(self.args.epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_precision, test_recall, test_f1, all_targets, all_scores = self._test(test_dataloader, criterion)
            l_epo.append(epoch)
            l_acc.append(test_acc)
            l_trloss.append(train_loss)
            l_precision.append(test_precision)
            l_recall.append(test_recall)
            l_f1.append(test_f1)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss, best_precision, best_recall, best_f1 = test_acc, test_loss, test_precision, test_recall, test_f1
                index = epoch
            self.logger.info('{}/{} - {:.2f}%'.format(epoch + 1, self.args.epoch, 100 * (epoch + 1) / self.args.epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, F1-score: {:.2f}'.format(test_loss, test_acc * 100, test_precision * 100, test_recall * 100, test_f1 * 100))

            scheduler.step()

        self.logger.info('best loss: {:.4f}, best acc: {:.2f}, best precision: {:.2f}, best recall: {:.2f}, best F1-score: {:.2f},best index: {:d}'.format(best_loss, best_acc * 100, best_precision*100, best_recall * 100, best_f1 * 100, index))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        
        # Draw the training process
        plt.figure(1)
        plt.plot(l_epo, l_trloss, color = 'blue')
        plt.xlabel('epoch')
        plt.ylabel('train-loss')
        plt.title('train-loss ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'trloss.jpg')
       
        plt.figure(2)
        plt.plot(l_epo, l_acc, color = 'red')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('accuracy ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'acc.jpg')

        plt.figure(3)
        plt.plot(l_epo, l_precision , color = 'green')
        plt.xlabel('epoch')
        plt.ylabel('precision')
        plt.title('precision ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'precision.jpg')

        plt.figure(4)
        plt.plot(l_epo, l_recall, color = 'yellow')
        plt.xlabel('epoch')
        plt.ylabel('recall')
        plt.title('recall ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'recall.jpg')

        plt.figure(5)
        plt.plot(l_epo, l_f1, color = 'purple')
        plt.xlabel('epoch')
        plt.ylabel('F1-score')
        plt.title('F1-score ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'F1-score.jpg')

        plt.figure(6)
        all_targets = np.array(all_targets)
        all_scores = np.array(all_scores)
        n_classes = all_scores.shape[1]

        for i in range(n_classes):
            if np.sum(all_targets == i) == 0:
                continue
            fpr, tpr, _ = roc_curve(all_targets == i, all_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color = 'orange')
        
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve ' + self.args.model_name)
        plt.savefig('./result/' + self.args.model_name + 'roc.jpg')

if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    detect = ddl(args, logger)
    detect.run()
