import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
from torchvision import transforms
from dataset import params
from model import get_model


class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:, None, None]).add_(mean[:, None, None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        # inps.sub_(mean[:,None,None]).div_(std[:,None,None])
        inps = (inps - mean[:, None, None]) / std[:, None, None]
        return inps

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1, 2, 0])  # c,h,w to h,w,c
            image[image < 0] = 0
            image[image > 1] = 1
            image = Image.fromarray((image.detach().cpu().numpy() * 255).astype(np.uint8))
            # print ('Saving to ', save_path)
            image.save(save_path)

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images


class ATT(BaseAttack):
    def __init__(self, model_name, args, sample_num_batches=130, steps=10, epsilon=16 / 255, target=False, decay=1.0):
        super(ATT, self).__init__('ATT', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.decay = decay

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224 / 16) ** 2)

        self.model_name = model_name
        self.im_fea = None
        self.im_grad = None
        self.size = 16
        self.lam = args.lam
        self.patch_index = self.Patch_index(self.size)

        assert self.sample_num_batches <= self.max_num_batches
        self._register_model()

    def TR_01_PC(self, num, length):
        rate_l = num
        tensor = torch.cat((torch.ones(rate_l), torch.zeros(length - rate_l)))
        return tensor

    def _register_model(self):
        self.var_A = 0
        self.var_qkv = 0
        self.var_mlp = 0
        self.gamma = 0.5
        if self.model_name == 'vit_base_patch16_224':
            self.back_attn = 11
            self.truncate_layers = self.TR_01_PC(10, 12)
            self.weaken_factor =  [0.45, 0.7, 0.65]
            self.scale = 0.4
            self.offset = 0.4            
        elif self.model_name == 'pit_b_224':
            self.back_attn = 12
            self.truncate_layers = self.TR_01_PC(9, 13)
            self.weaken_factor = [0.25, 0.6, 0.65]
            self.scale = 0.3
            self.offset = 0.45
        elif self.model_name == 'cait_s24_224':
            self.back_attn = 24
            self.truncate_layers = self.TR_01_PC(4, 25)
            self.weaken_factor = [0.3, 1., 0.6]
            self.scale = 0.35
            self.offset = 0.4
        elif self.model_name == 'visformer_small':
            self.back_attn = 7
            self.truncate_layers = self.TR_01_PC(8, 8)
            self.weaken_factor = [0.4, 0.8, 0.3]
            self.scale = 0.15
            self.offset = 0.25

        def attn_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.truncate_layers[self.back_attn] * self.weaken_factor[0]
            out_grad = mask * grad_in[0][:]
            if self.var_A != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            if self.model_name in ['vit_base_patch16_224', 'visformer_small', 'pit_b_224']:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = min_all // H
                min_all_W = min_all % H

                out_grad[:, range(C), max_all_H, :] *= GPF_
                out_grad[:, range(C), :, max_all_W] *= GPF_
                out_grad[:, range(C), min_all_H, :] *= GPF_
                out_grad[:, range(C), :, min_all_W] *= GPF_

            if self.model_name in ['cait_s24_224']:
                B, H, W, C = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H * W, C)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)
                min_all_H = min_all // H
                min_all_W = min_all % H

                out_grad[:, max_all_H, :, range(C)] *= GPF_
                out_grad[:, :, max_all_W, range(C)] *= GPF_
                out_grad[:, min_all_H, :, range(C)] *= GPF_
                out_grad[:, :, min_all_W, range(C)] *= GPF_

            self.var_A = torch.var(out_grad)

            self.back_attn -= 1
            return (out_grad,)

        def attn_cait_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.truncate_layers[self.back_attn] * self.weaken_factor[0]

            out_grad = mask * grad_in[0][:]
            if self.var_A != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_A))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            B, H, W, C = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0, :, 0, :], axis=0)
            min_all = np.argmin(out_grad_cpu[0, :, 0, :], axis=0)

            out_grad[:, max_all, :, range(C)] *= GPF_
            out_grad[:, min_all, :, range(C)] *= GPF_

            self.var_A = torch.var(out_grad)
            self.back_attn -= 1
            return (out_grad,)

        def q_ATT(module, grad_in, grad_out):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF_ = self.gamma
            out_grad[:] *= GPF_
            self.var_qkv = torch.var(out_grad)
            return (out_grad, grad_in[1], grad_in[2])

        def v_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[1]
            out_grad = mask * grad_in[0][:]
            if self.var_qkv != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_qkv))).clamp(0, 1)
            else:
                GPF_ = self.gamma

            if self.model_name in ['visformer_small']:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = min_all // H
                min_all_W = min_all % H

                out_grad[:, range(C), max_all_H, max_all_W] *= GPF_
                out_grad[:, range(C), min_all_H, min_all_W] *= GPF_

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] *= GPF_
                out_grad[:, min_all, range(c)] *= GPF_
                
            self.var_qkv = torch.var(out_grad)
            return (out_grad, grad_in[1])

        def mlp_ATT(module, grad_in, grad_out):
            mask = torch.ones_like(grad_in[0]) * self.weaken_factor[2]
            out_grad = mask * grad_in[0][:]
            if self.var_mlp != 0:
                GPF_ = (self.gamma + self.lam * (1 - torch.sqrt(torch.var(out_grad) / self.var_mlp))).clamp(0, 1)
            else:
                GPF_ = self.gamma

            if self.model_name in ['visformer_small']:
                B, C, H, W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, C, H * W)
                max_all = np.argmax(out_grad_cpu[0, :, :], axis=1)
                max_all_H = max_all // H
                max_all_W = max_all % H
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=1)
                min_all_H = min_all // H
                min_all_W = min_all % H
                out_grad[:, range(C), max_all_H, max_all_W] *= GPF_
                out_grad[:, range(C), min_all_H, min_all_W] *= GPF_

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()

                max_all = np.argmax(out_grad_cpu[0, :, :], axis=0)
                min_all = np.argmin(out_grad_cpu[0, :, :], axis=0)

                out_grad[:, max_all, range(c)] *= GPF_
                out_grad[:, min_all, range(c)] *= GPF_
                
            self.var_mlp = torch.var(out_grad)
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics

        def get_fea(module, input, output):
            self.im_fea = output.clone()

        def get_grad(module, input, output):
            self.im_grad = output[0].clone()

        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            self.get_fea_hook = self.model.blocks[10].register_forward_hook(get_fea)
            self.get_grad_hook = self.model.blocks[10].register_backward_hook(get_grad)
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_ATT)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_ATT)
                self.model.blocks[i].mlp.register_backward_hook(mlp_ATT)
        elif self.model_name == 'pit_b_224':
            self.get_fea_hook = self.model.transformers[2].blocks[2].register_forward_hook(get_fea)
            self.get_grad_hook = self.model.transformers[2].blocks[2].register_backward_hook(get_grad)
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_ATT)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_ATT)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_ATT)
        elif self.model_name == 'cait_s24_224':
            self.get_fea_hook = self.model.blocks[23].register_forward_hook(get_fea)
            self.get_grad_hook = self.model.blocks[23].register_forward_hook(get_grad)
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_ATT)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_ATT)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_ATT)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind - 24].attn.attn_drop.register_backward_hook(attn_cait_ATT)
                    self.model.blocks_token_only[block_ind - 24].attn.q.register_backward_hook(q_ATT)
                    self.model.blocks_token_only[block_ind - 24].attn.k.register_backward_hook(v_ATT)
                    self.model.blocks_token_only[block_ind - 24].attn.v.register_backward_hook(v_ATT)
                    self.model.blocks_token_only[block_ind - 24].mlp.register_backward_hook(mlp_ATT)
        elif self.model_name == 'visformer_small':
            self.get_fea_hook = self.model.stage3[2].register_forward_hook(get_fea)
            self.get_grad_hook = self.model.stage3[2].register_forward_hook(get_grad)
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_ATT)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_ATT)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_ATT)
                elif block_ind >= 4:
                    self.model.stage3[block_ind - 4].attn.attn_drop.register_backward_hook(attn_ATT)
                    self.model.stage3[block_ind - 4].attn.qkv.register_backward_hook(v_ATT)
                    self.model.stage3[block_ind - 4].mlp.register_backward_hook(mlp_ATT)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size / self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:, :, r * self.crop_length:(r + 1) * self.crop_length,
            c * self.crop_length:(c + 1) * self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def Patch_index(self, size):
        img_size = 224
        filterSize = size
        stride = size
        P = np.floor((img_size - filterSize) / stride) + 1
        P = P.astype(np.int32)
        Q = P
        index = np.ones([P * Q, filterSize * filterSize], dtype=int)
        tmpidx = 0
        for q in range(Q):
            plus1 = q * stride * img_size
            for p in range(P):
                plus2 = p * stride
                index_ = np.array([], dtype=int)
                for i in range(filterSize):
                    plus = i * img_size + plus1 + plus2
                    index_ = np.append(index_, np.arange(plus, plus + filterSize, dtype=int))
                index[tmpidx] = index_
                tmpidx += 1
        index = torch.LongTensor(np.tile(index, (1, 1, 1))).cuda()
        return index

    def norm_patchs(self, GF, index, patch, scale, offset):
        patch_size = patch ** 2
        for i in range(len(GF)):
            tmp = torch.take(GF[i], index[i])
            norm_tmp = torch.mean(tmp, dim=-1)
            scale_norm = scale * ((norm_tmp - norm_tmp.min()) / (norm_tmp.max() - norm_tmp.min())) + offset
            tmp_bi = torch.as_tensor(scale_norm.repeat_interleave(patch_size)) * 1.0
            GF[i] = GF[i].put_(index[i], tmp_bi)
        return GF

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()

        output = self.model(inps)
        output.backward(torch.ones_like(output))
        resize = transforms.Resize((224, 224))
        if self.model_name == 'vit_base_patch16_224':
            GF = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
            GF = resize(GF.reshape(1, 14, 14))
        elif self.model_name == 'pit_b_224':
            GF = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
            GF = resize(GF.reshape(1, 8, 8))
        elif self.model_name == 'cait_s24_224':
            GF = (self.im_fea[0] * self.im_grad).sum(-1)
            GF = resize(GF.reshape(1, 14, 14))
        elif self.model_name == 'visformer_small':
            GF = (self.im_fea[0] * self.im_grad).sum(0)
            GF = resize(GF.unsqueeze(0))

        GF_patchs_t = self.norm_patchs(GF, self.patch_index, self.size, self.scale, self.offset)
        GF_patchs_start = torch.ones_like(GF_patchs_t).cuda() * 0.99
        GF_offset = (GF_patchs_start - GF_patchs_t) / self.steps

        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            self.var_A = 0
            self.var_qkv = 0
            self.var_mlp = 0
            if self.model_name == 'pit_b_224':
                self.back_attn = 12
            elif self.model_name == 'vit_base_patch16_224':
                self.back_attn = 11
            elif self.model_name == 'visformer_small':
                self.back_attn = 7
            elif self.model_name == 'cait_s24_224':
                self.back_attn = 24
            # add_perturbation = self._generate_samples_for_interactions(perts, i)
            # outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))
            ##### If you use patch out, please uncomment the previous two lines and comment line 440-443.

            torch.manual_seed(i)
            random_patch = torch.rand(14, 14).repeat_interleave(16).reshape(14,14*16).repeat(1,16).reshape(224,224).cuda()
            GF_patchs = torch.where(torch.as_tensor(random_patch > GF_patchs_start - GF_offset * (i + 1)), 0., 1.).cuda()
            outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts * GF_patchs.detach())))
            cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1, 2, 3], keepdim=True)
            grad += momentum * self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps + perts.data)).detach(), None
