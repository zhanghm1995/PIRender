'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-08-06 21:57:41
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import math

import torch
import torch.nn as nn

from trainers.base import BaseTrainer
from util.trainer import accumulate, get_optimizer
from loss.perceptual  import PerceptualLoss

class Face2FaceTrainer(BaseTrainer):
    r"""Initialize lambda model trainer.

    Args:
        cfg (obj): Global configuration.
        net_G (obj): Generator network.
        opt_G (obj): Optimizer for the generator network.
        sch_G (obj): Scheduler for the generator optimizer.
        train_data_loader (obj): Train data loader.
        val_data_loader (obj): Validation data loader.
    """

    def __init__(self, opt, net_G, opt_G, sch_G,
                 train_data_loader, val_data_loader=None):
        super(Face2FaceTrainer, self).__init__(opt, net_G, opt_G, sch_G, train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))

    def _init_loss(self, opt):
        self._assign_criteria(
            'perceptual_final',
            PerceptualLoss(
                network=opt.trainer.vgg_param_final.network,
                layers=opt.trainer.vgg_param_final.layers,
                num_scales=getattr(opt.trainer.vgg_param_final, 'num_scales', 1),
                use_style_loss=getattr(opt.trainer.vgg_param_final, 'use_style_loss', False),
                weight_style_to_perceptual=getattr(opt.trainer.vgg_param_final, 'style_to_perceptual', 0)
                ).to('cuda'),
            opt.trainer.loss_weight.weight_perceptual_final)
        
        self.weights['l1_loss'] = 1.0

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        self.gen_losses = {}
        
        # source_semantic = data['source_semantics']
        source_semantic = None
        reference_image = data['reference_image']
        blended_image = data['blended_image']

        gt_image = data['source_image']

        output_dict = self.net_G(reference_image, blended_image, source_semantic)

        fake_img = output_dict['fake_image']
        self.gen_losses["perceptual_final"] = self.criteria['perceptual_final'](fake_img, gt_image)
        self.gen_losses['l1_loss'] = nn.functional.l1_loss(fake_img, gt_image)

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

    def _get_visualizations(self, data):
        blended_image, reference_image = data['blended_image'], data['reference_image']
        rendered_image = data['rendered_image']
        source_semantic = None
        # source_semantic = data['source_semantics']

        gt_image = data['source_image']

        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
                reference_image, blended_image, source_semantic
                )
            fake_img = output_dict['fake_image'] # (B, 3, H, W)

            sample = torch.cat([rendered_image, blended_image, reference_image, fake_img, gt_image], dim=3)
        return sample

    def test(self, data_loader, output_dir, current_iteration=-1):
        pass

    def _compute_metrics(self, data, current_iteration):
        blended_image, reference_image = data['blended_image'], data['reference_image']
        # source_semantic = data['source_semantics']
        source_semantic = None

        gt_image = data['source_image']

        metrics = {}
        with torch.no_grad():
            self.net_G_ema.eval()
            output_dict = self.net_G_ema(
                reference_image, blended_image, source_semantic
                )
            fake_image = output_dict['fake_image']
            metrics['lpips'] = self.lpips(fake_image, gt_image).mean()
        return metrics