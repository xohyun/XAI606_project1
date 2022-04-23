import os
import torch
import models.SimSiam_model as simsiam
import models.ShallowConvNet_model as conv
from torchinfo import summary

from utils.utils import import_model, pretrained_model, write_pickle


class ModelBuilder:
    def __init__(self, args):
        print("[Build Model]")
        self.model = self.__build_model(args)
        self.__set_device(self.model, args.device)
        self.model_summary(args, self.model)

    def __build_model(self, args):
        if args.mode == 'train':
            # model = import_model(args.model, args.cfg)
            base_enc = conv.ShallowConvNet(F1=40, T1=25, F2=40, P1_T=75, P1_S=15, drop_out=0.5, pool_mode='mean')
            model = simsiam.SimSiam(base_enc, 27, 27)
            write_pickle(os.path.join(args.save_path, "model.pk"), model)
            return model
        else:
            feature_extractor = pretrained_model(args.load_path, s_dict='feature_ext_state_dict')
            model = pretrained_model(args.load_path, classifier=True)
            return feature_extractor, model

    def __set_device(self, model, device):
        if device == 'cpu':
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise ValueError("Check GPU")
            device = torch.device(f'cuda:{device}')
            torch.cuda.set_device(device)  # If you want to check device, use torch.cuda.current_device().
            try:
                model.cuda()
            except:
                model[0].cuda()
                model[1].cuda()    
        # Print device
        try:
            model.device = device
        except:
            model[0].device = device
            model[1].device = device
        print(f"device: {device}")
        print("")

    def model_summary(self, args, model):
        if args.summary:
            # results = summary(model, args.cfg.input_shape, col_names=["kernel_size", "output_size", "num_params"],
            #                   device=model.device if not model.device == 'multi' else torch.device("cuda:0"))
            # args.trainable_params = results.trainable_params
            print("")