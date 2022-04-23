import os
import numpy as np
from utils.get_args import Args
from utils.utils import fix_random_seed, timeit, write_csv
from data_loader.data_generator import DataGenerator
from models.model_builder import ModelBuilder
import models.SimSiam_model
from trainers.trainer_maker import TrainerMaker
import scipy.io
from sklearn.metrics import accuracy_score
import torch
torch.backends.cudnn.enable = False

@timeit
def main():
    args_class = Args()
    args = args_class.args

    for args.subject in args.target_subject:
        args_class.preprocess()
        args_class.print_info()

        # Fix random seed
        if args.seed:
            fix_random_seed(args)

        # Load data
        data = DataGenerator(args)
        
        # Build model
        model = ModelBuilder(args).model

        # Make Trainer
        if args.mode == 'train':
            cls = models.SimSiam_model.classifier().to(model.device)
            trainer = TrainerMaker(args, model, data).trainer
            classifier_trainer = TrainerMaker(args, cls, data, feature_extractor=model).trainer
            
            trainer.train()
            classifier_trainer.train(downstream=True)
       
        elif args.get_prediction:
            classifier_trainer = TrainerMaker(args, model[1], data, feature_extractor=model[0]).trainer
            classifier_trainer.get_prediction()

        elif args.evaluation:
            trainer.evaluation()
        else:
            trainer.test()


    if args.evaluation:
        write_csv(os.path.join(args.base_load_path, "prediction.csv"), args.acc_list)

if __name__ == '__main__':
    main()
