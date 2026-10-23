from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
import torch
import os

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.perf import enable_tf32, apply_allocator_tuning

from torch.optim import Adam

class nnUNetTrainerLightMUNet(nnUNetTrainerNoDeepSupervision):

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        # Small, safe speed tweak + allocator tuning
        try:
            enable_tf32()
        except Exception:
            pass
        try:
            apply_allocator_tuning()
        except Exception:
            pass
        # cuDNN benchmarking: default to False (avoids high-memory algo picks), allow opt-in via env
        # Set NNUNET_CUDNN_BENCHMARK=1 to enable on stable shapes for extra speed
        try:
            _bmk = os.environ.get('NNUNET_CUDNN_BENCHMARK', None)
            if _bmk is None:
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = _bmk.lower() in ('1', 'true', 't', 'yes', 'y')
        except Exception:
            pass
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.grad_scaler = None
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        # Optional gradient accumulation to lower peak VRAM. Default off (1)
        try:
            self.accum_steps = max(1, int(os.environ.get('ACCUM_STEPS', '1')))
        except Exception:
            self.accum_steps = 1
        self._accum_i = 0

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        
        label_manager = plans_manager.get_label_manager(dataset_json)

        # Lazy import to avoid importing mamba/LightMUNet during trainer class discovery
        from nnunetv2.nets.LightMUNet import LightMUNet

        model = LightMUNet(
            spatial_dims = len(configuration_manager.patch_size),
            init_filters = 32,
            in_channels=num_input_channels,
            out_channels=label_manager.num_segmentation_heads,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
        )
        # Prefer channels-last memory format for conv perf and slightly reduced workspace
        spatial_dims = len(configuration_manager.patch_size)
        mf = torch.channels_last if spatial_dims == 2 else getattr(torch, 'channels_last_3d', torch.channels_last)
        model = model.to(memory_format=mf)

        return model
    

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # Use channels-last memory format for inputs when possible
        if data.ndim == 4:
            data = data.contiguous(memory_format=torch.channels_last)
        elif data.ndim == 5:
            data = data.contiguous(memory_format=getattr(torch, 'channels_last_3d', torch.contiguous_format))
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Gradient accumulation: only clear grads at the start of a window
        if self._accum_i == 0:
            self.optimizer.zero_grad(set_to_none=True)
        # BF16 autocast for lower VRAM and faster compute (trim cache)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=False):
            # pre-cast inputs to bf16 to reduce transient casts
            data = data.to(torch.bfloat16, non_blocking=True)
            output = self.network(data)
            l = self.loss(output, target)
        # average gradient over accumulation window
        (l / self.accum_steps).backward()

        loss_item = l.detach().float().cpu().numpy()
        # Free large intermediates ASAP
        del output
        del data
        del target

        self._accum_i += 1
        if self._accum_i >= self.accum_steps:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_i = 0

        return {'loss': loss_item}
    

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        # Use channels-last memory format for inputs when possible
        if data.ndim == 4:
            data = data.contiguous(memory_format=torch.channels_last)
        elif data.ndim == 5:
            data = data.contiguous(memory_format=getattr(torch, 'channels_last_3d', torch.contiguous_format))
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Validation does not need gradients, keep BF16 autocast for memory
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, cache_enabled=False):
                # pre-cast inputs to bf16 to reduce transient casts
                data = data.to(torch.bfloat16, non_blocking=True)
                output = self.network(data)
                del data
                l = self.loss(output, target)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            # Match one-hot dtype to logits to avoid FP32 buffers
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=output.dtype
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # Keep mask in same dtype as logits to avoid upcasting
                mask = mask.to(output.dtype)
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                mask = mask.to(output.dtype)
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        # Cast small reduction tensors to float32 on CPU before converting to numpy (bf16 unsupported by numpy)
        tp_hard = tp.detach().float().cpu().numpy()
        fp_hard = fp.detach().float().cpu().numpy()
        fn_hard = fn.detach().float().cpu().numpy()
        # Free GPU tensors early to reduce peaks
        del tp, fp, fn
        del predicted_segmentation_onehot
        del mask
        del output
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        loss_val = l.detach().float().cpu().numpy()
        del l
        del target
        return {'loss': loss_val, 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
    
    def configure_optimizers(self):

        # Prefer fused Adam on CUDA if available; fall back gracefully
        try:
            optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5, fused=True)
        except TypeError:
            optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.9)

        return optimizer, scheduler
    
    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    def on_train_epoch_end(self, train_outputs: list[dict]):
        # If there are leftover microbatches, flush a final step with proper rescaling
        if getattr(self, 'accum_steps', 1) > 1 and getattr(self, '_accum_i', 0) != 0:
            # Rescale grads to match mean over the last partial window
            k = max(1, self._accum_i)
            scale = float(self.accum_steps) / float(k)
            try:
                for p in self.network.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_i = 0

        # Continue with default logging behavior
        return super().on_train_epoch_end(train_outputs)
