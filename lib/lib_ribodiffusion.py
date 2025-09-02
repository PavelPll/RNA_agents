# Implemented from here: https://github.com/ml4bio/RiboDiffusion

import os
import sys
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../RiboDiffusion'))

import torch
import numpy as np
from models import *
from utils import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets import utils as du
import functools
import tree
from configs.inference_ribodiffusion import get_config
import yaml

class Pdb2Fasta:
    def __init__(self):
        with open("../configs/rna_rag.yaml", 'r') as f:
            self.global_config = yaml.safe_load(f)
        self.config = get_config()
        checkpoint_path = '../RiboDiffusion/ckpts/exp_inf_large.pth'
        #self.config.eval.sampling_steps = 50
        self.config.eval.sampling_steps = self.config.eval.sampling_steps  
        # Initialize tools for model inference.
        self.NUM_TO_LETTER = np.array(['A', 'G', 'C', 'U'])  

        # Initialize model
        self.model = create_model(self.config)
        ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config.model.ema_decay)
        params = self.model.parameters()
        optimizer = self.get_optimizer(self.config, self.model.parameters())
        state = dict(optimizer=optimizer, model=self.model, ema=ema, step=0)

        model_size = sum(p.numel() for p in self.model.parameters()) * 4 / 2 ** 20
        print('model size: {:.1f}MB'.format(model_size))

        # Load checkpoint
        state = restore_checkpoint(checkpoint_path, state, device=self.config.device)
        ema.copy_to(self.model.parameters())

        # Initialize noise scheduler
        self.noise_scheduler = NoiseScheduleVP(self.config.sde.schedule, continuous_beta_0=self.config.sde.continuous_beta_0,
                                        continuous_beta_1=self.config.sde.continuous_beta_1)
        # Obtain data scalar and inverse scalar
        self.inverse_scaler = get_data_inverse_scaler(self.config)

        # Setup sampling function
        ### test_sampling_fn = get_sampling_fn(self.config, self.noise_scheduler, self.config.eval.sampling_steps, self.inverse_scaler)
        self.pdb2data = functools.partial(du.PDBtoData, num_posenc=self.config.data.num_posenc, 
                                          num_rbf=self.config.data.num_rbf, 
                                          knn_num=self.config.data.knn_num)


    def get_optimizer(self, config, params):
        """Return a flax optimizer object based on `config`."""
        if config.optim.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
        else:
            raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')
        return optimizer



    def pdb2fasta(self, source_path = "results/exper_1/fasta50A.pdb", cond_scale=5):
        config = self.config

        pdb_file=source_path
        pdb_id = pdb_file.replace('.pdb', '')
        if '/' in pdb_id:
            pdb_id = pdb_id.split('/')[-1]    

        # You can adjust the conditional scaling weight by setting config.eval.dynamic_threshold 
        # and config.eval.cond_scale to obtain more diverse sequences.
        config.eval.dynamic_threshold=False
        #print("config.eval.cond_scale:", config.eval.cond_scale)
        #print("config.eval.dynamic_threshold=True:", config.eval.dynamic_threshold)
        config.eval.cond_scale=cond_scale
        # config.eval.n_samples=100
        config.eval.n_samples = self.global_config["ribodiffusion"]["n_samples"]
        test_sampling_fn = get_sampling_fn(config, self.noise_scheduler, config.eval.sampling_steps, self.inverse_scaler)
        struct_data = self.pdb2data(pdb_file)
        struct_data = tree.map_structure(lambda x:x.unsqueeze(0).repeat_interleave(config.eval.n_samples, dim=0).to(config.device), struct_data)
        samples = test_sampling_fn(self.model, struct_data)
        # print(f'PDB ID: {pdb_id}')
        native_seq = ''.join(list(self.NUM_TO_LETTER[struct_data['seq'][0].cpu().numpy()]))
        # print(f'Native sequence: {native_seq}')
        output = []
        for i in range(len(samples)):
            # native_seq = ''.join(list(NUM_TO_LETTER[struct_data['seq'].squeeze(0).cpu().numpy()]))
            # print(f'Native sequence: {native_seq}')
            designed_seq = ''.join(list(self.NUM_TO_LETTER[samples[i].cpu().numpy()]))
            #print(f'Generated sequence {i+1}: {designed_seq}')
            recovery_ = samples[i].eq(struct_data['seq'][0]).float().mean().item()
            #print(f'Recovery rate {i+1}: {recovery_:.4f}')
            output.append((designed_seq, recovery_))
        return output