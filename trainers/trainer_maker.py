import importlib
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from loss.ND_Crossentropy import CrossentropyND, TopKLoss
from loss.focalloss import *
from loss.label_smoothing import LabelSmoothingCrossEntropy

class TrainerMaker:
    def __init__(self, args, model, data, feature_extractor=None):
        print("[Make Trainer]")
        if args.mode == 'train':
            criterion = self.__set_criterion(args.criterion)
            optimizer = self.__set_optimizer(args, model)
            scheduler = self.__set_scheduler(args, optimizer)
            # history = self.__make_history(args.metrics)
            history = defaultdict(list)
            self.feature_extractor = feature_extractor
            self.trainer = self.__make_trainer(args=args,
                                               model=model,
                                               data=data,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               scheduler=scheduler,
                                               history=history,
                                               feature_extractor=feature_extractor)
        else:
            criterion = self.__set_criterion(args.criterion)
            history = defaultdict(list)
            self.feature_extractor = feature_extractor
            self.trainer = self.__make_trainer(args=args,
                                               model=model,
                                               data=data,
                                               criterion=criterion,
                                               history=history,
                                               feature_extractor=feature_extractor)
        print("")

    def __set_criterion(self, criterion):
        if criterion == "MSE":
            criterion = nn.MSELoss()
        elif criterion == "CEE":
            criterion = nn.CrossEntropyLoss()
        elif criterion == "Focal":
            criterion = FocalLoss(gamma=2)
        elif criterion == "ND":
            criterion = CrossentropyND()
        elif criterion == "TopK":
            criterion = TopKLoss()
        elif criterion == "LS":
            criterion = LabelSmoothingCrossEntropy()
        return criterion

    def __set_optimizer(self, args, model):
        if args.opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.opt == "Adam":
            optimizer = optim.Adam(list(model.parameters()), lr=args.lr, weight_decay=args.wd)
        elif args.opt == 'AdamW':
            optimizer = optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=args.wd)
        else:
            raise ValueError(f"Not supported {args.opt}.")
        return optimizer

    def __set_scheduler(self, args, optimizer):
        if args.scheduler is None:
            return None
        elif args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             threshold=0.1, threshold_mode='abs', verbose=True)
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=args.T_max if args.T_max else args.epochs,
                                                             eta_min=args.eta_min if args.eta_min else 0)
        else:
            raise ValueError(f"Not supported {args.scheduler}.")
        return scheduler

    def __make_trainer(self, **kwargs):
        if self.feature_extractor == None:
            module = importlib.import_module(f"trainers.{kwargs['args'].model}_trainer")
        else: 
            module = importlib.import_module(f"trainers.classifier_trainer")    

        trainer = getattr(module, 'Trainer')(**kwargs)
        return trainer
