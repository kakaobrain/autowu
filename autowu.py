from collections import deque
import math
import warnings

import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      LambdaLR, ReduceLROnPlateau)

import gpytorch
from gpytorch.utils.cholesky import NotPSDError


class CustomGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(CustomGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.base_covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Regressor(torch.nn.Module):

    def __init__(self, lengthscale=1.0, device=torch.device('cpu')):
        super(Regressor, self).__init__()

        self.device = device

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = CustomGP(None, None, self.likelihood).to(self.device)

        self.gp_model.initialize(**{
            'base_covar_module.base_kernel.lengthscale': lengthscale,
            'likelihood.noise': 1e-3,
        })

        # Lengthscale is excluded from training.
        trainable_gp_params = [p for n, p in self.gp_model.named_parameters()
                               if 'lengthscale' not in n]
        self.gp_optim = torch.optim.Adam(trainable_gp_params, lr=0.01)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp_model)

        self.register_buffer('full_data', None)
        self.full_data = torch.empty(0, device=self.device)

        self.lengthscale = lengthscale

        self._rng_generator = torch.Generator().manual_seed(0)

    def reset_data(self):
        self.full_data = torch.empty(0, device=self.device)

    def add_data(self, new_losses):
        # We only store y-data; x-data is 1-dim grid.
        self.full_data = torch.cat([self.full_data, new_losses.to(device=self.device)])

        len_data = len(self.full_data)
        not_inf_or_nan = ~torch.isinf(self.full_data) & ~torch.isnan(self.full_data)
        valid_indices = torch.arange(len_data)[not_inf_or_nan]
        self.gp_model.initialize(**{
            'mean_module.constant': self.full_data[valid_indices].mean(),
        })

    def subsample_and_condition(self, max_train_data=100):
        len_data = len(self.full_data)

        # Subsample train data
        not_inf_or_nan = ~torch.isinf(self.full_data) & ~torch.isnan(self.full_data)
        valid_indices = torch.arange(len_data)[not_inf_or_nan]
        len_valid_data = len(valid_indices)

        if len_valid_data > max_train_data:
            # sampling relies on a fixed rng generator to ensure same behavior in every processes if distributed
            subindices = torch.ones(len_valid_data).multinomial(max_train_data, generator=self._rng_generator)
            subindices = subindices.sort().values
            sampled_indices = valid_indices[subindices]
        else:
            sampled_indices = valid_indices

        train_x = sampled_indices.to(dtype=torch.float, device=self.device) / len_data
        train_y = self.full_data[sampled_indices]

        self.gp_model.set_train_data(train_x, train_y, strict=False)

    def fit(self, num_iter=200):

        train_x = self.gp_model.train_inputs[0]
        train_y = self.gp_model.train_targets

        # Fit GP model
        self.gp_model.train()

        for _ in range(num_iter):
            self.gp_optim.zero_grad()
            output = self.gp_model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            self.gp_optim.step()

    @torch.no_grad()
    def predict(self, pred_locs, noise=False):
        self.gp_model.eval()
        pred_locs = pred_locs.to(self.device)
        if noise:
            pred_dist = self.likelihood(self.gp_model(pred_locs))
        else:
            pred_dist = self.gp_model(pred_locs)
        return pred_dist


class AutoWU:
    """Automatic LR scheduler using GP regression of diagnostics.

    Args:
        optimizer (torch.optim.Optimizer): Base optimizer.
        steps_per_epoch (int): Number of steps in an epoch.
        total_epochs (int): Total number of epochs.
        max_warmup_fraction (float, optional): The maximum fraction of total steps for
            the warm-up phase. Switching to other phases may occur before the full fraction
            is spent. (default: 0.5)
        immediate_cooldown (bool, optional): If ``True``, the decay phase is skipped. This
            implies that arguments `decay_phase_factor` and `final_phase_fraction` are
            ignored. (default: False)
        decay_phase_stat (str, optional): If ``loss``, ReduceOnPlateau with per-epoch average loss
            as the metric is used for the decay phase scheduler. (default: None)
        decay_phase_factor (float, optional): Multiplicative factor used to decay LR. If
            `decay_phase_stat` is set to be ``loss``, it is used for ReduceOnPlateau.
            Otherwise, LR is decayed per epoch in the decay phase. (default: 1.0)
        cooldown_fraction (float, optional): The fraction of total steps for the
            the final phase. (default: 0.2)
        cooldown_type (str, optional): The type of schedule used in the final phase,
            either "cosine" or "half_cosine" (default: cosine)
        device (torch.device or str): Device where the GP regression is computed on.
            (default: cpu)

    Note:
        Under current implementation, learning rate will be identical for all parameter groups
        regardless of the given initial value.
    """

    def __init__(self, optimizer, steps_per_epoch, total_epochs,
                 min_lr=1e-5, max_lr=1.0, max_warmup_fraction=0.5,
                 warmup_unit='step', warmup_type='exp',
                 warmup_only=False,  # only for ablation
                 immediate_cooldown=False, decay_phase_stat=None, decay_phase_factor=1.0,
                 cooldown_fraction=0.2, cooldown_type='cosine',
                 device=torch.device('cpu'),
                 ):

        if len(optimizer.param_groups) > 1:
            warnings.warn("LR will be identically set in all parameter groups")
        if warmup_unit not in ['step', 'epoch']:
            raise ValueError(f"Invalid warmup_unit type: {warmup_unit}")
        if cooldown_fraction > 1.0:
            raise ValueError(f"Invalid cooldown_fraction value: {cooldown_fraction} (must be between 0 and 1)")
        if cooldown_type not in ['cosine', 'half_cosine']:
            raise ValueError(f"Invalid type of cooldown phase schedule: {cooldown_type}")
        if decay_phase_stat not in [None, 'loss']:
            raise ValueError(f"Invalid type of decay phase stat: {decay_phase_stat}")
        if decay_phase_factor <= 0.0 or decay_phase_factor > 1.0:
            raise ValueError(f"Invalid decay factor: {decay_phase_factor}")

        self.optimizer = optimizer
        self.device = device

        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.total_steps = steps_per_epoch * total_epochs

        self.min_lr = min_lr
        self.max_lr = max_lr

        self.immediate_cooldown = immediate_cooldown
        self.decay_phase_factor = decay_phase_factor

        self.phase = None
        self.scheduler = None
        self.warmup_phase_regressor = Regressor(lengthscale=0.2, device=device)
        self.warmup_phase_states = dict(max_fraction=max_warmup_fraction, unit=warmup_unit, type=warmup_type,
                                        patience=3, n_tests=5, confidence=0.95)
        self._losses_last_epoch = None
        self._prev_test_results = None

        self.decay_phase_states = dict(decay_factor=decay_phase_factor, stat=decay_phase_stat)
        self._stats_last_epoch = None

        self.cooldown_phase_states = dict(fraction=cooldown_fraction, type=cooldown_type)

        self.last_step = 0

        self.warmup_only = warmup_only
        if self.warmup_only:
            self.start_decay_phase_once = self.start_warmup_phase_once
            self.start_cooldown_phase_once = self.start_warmup_phase_once

        self.start_warmup_phase_once()

    def step(self, loss):
        """Adjust LR when needed."""
        self.last_step += 1

        scheduler_args = tuple()

        time_to_cooldown = (self.last_step >= (1 - self.cooldown_phase_states['fraction']) * self.total_steps)
        if time_to_cooldown:
            self.start_cooldown_phase_once()

        must_step, switched, epoch_end = False, False, False
        if self.phase == 'warmup':

            max_warmup_exceeded = self.last_step >= int(self.warmup_phase_states['max_fraction'] * self.total_steps)
            if max_warmup_exceeded:
                self.start_decay_phase_once()

            else:
                self._losses_last_epoch.append(loss.detach().clone())
                epoch_end = len(self._losses_last_epoch) >= self.steps_per_epoch
                if epoch_end:
                    self.warmup_phase_regressor.add_data(torch.stack(self._losses_last_epoch))
                    switched = self.maybe_switch()
                    self._losses_last_epoch.clear()

        elif self.phase == 'decay':

            if self.decay_phase_states['stat'] == 'loss':
                stat = loss
                self._stats_last_epoch.append(stat.detach().clone())

            epoch_end = len(self._stats_last_epoch) >= self.steps_per_epoch
            if epoch_end and self.decay_phase_states['stat'] == 'loss':
                stats_avg = sum(self._stats_last_epoch) / len(self._stats_last_epoch)
                scheduler_args = (stats_avg,)
                self._stats_last_epoch.clear()

        if self.phase == 'warmup':
            must_step = not switched and ((self.warmup_phase_states['unit'] == 'step') or epoch_end)
        elif self.phase == 'decay':
            must_step = epoch_end
        else:
            must_step = True

        if must_step:
            self.scheduler.step(*scheduler_args)

    def maybe_switch(self):
        """Decide and act whether to switch from the warmup phase to the decay phase.
        It is tested that current loss value is confidently higher than the minimum
        of the past loss trajectory for `patience` consecutive tests.
        """

        # hyperparameters involved in this test
        n_tests = self.warmup_phase_states['n_tests']
        confidence = self.warmup_phase_states['confidence']
        patience = self.warmup_phase_states['patience']

        pred_xs, pred_means, pred_covars = self.regress(self.warmup_phase_regressor, n_preds=n_tests)
        past_minimum_probs = torch.stack([
            self.past_minimum_prob(pred_means[j], pred_covars[j])
            for j in range(n_tests)
        ])

        past_minimum_votes = (past_minimum_probs > confidence).sum()
        past_minimum_maj = (past_minimum_votes > n_tests / 2)
        self._prev_test_results.append(past_minimum_maj)

        must_switch = (sum(self._prev_test_results) >= patience)

        pred_xs_argmin = pred_xs[pred_means.argmin(dim=1)]
        current_lr = self.get_last_lr()[0]
        min_lr = self.min_lr
        if self.warmup_phase_states['type'] == 'linear':
            lrs_at_argmin = min_lr + (current_lr - min_lr) * pred_xs_argmin
        else:
            lrs_at_argmin = min_lr * ((current_lr/min_lr) ** pred_xs_argmin)
        lr_after_switch = lrs_at_argmin.mean().item()

        if must_switch:
            for g in self.optimizer.param_groups:
                g['lr'] = lr_after_switch
            self.start_decay_phase_once()

        return must_switch

    @staticmethod
    @torch.no_grad()
    def past_minimum_prob(loc, covar_mat):
        r"""Returns :math:`\max_{i<k} P(X_i < X_k)` where

        .. math::

            (X_1,...,X_k) \sim \mathcal{N}(loc, covar\_mat).

        Arguments:
            loc (torch.Tensor): 1-dim. Tensor of size k.
            covar_mat (torch.Tensor): 2-dim. Tensor of size k by k.
        """
        assert loc.dim() == 1 and covar_mat.dim() == 2

        k = loc.shape[-1]
        assert covar_mat.shape[-1] == covar_mat.shape[-2] == k

        proj_mat = torch.cat([
            torch.eye(k-1, device=loc.device),
            -torch.ones(k-1, 1, device=loc.device)
        ], dim=1)
        loc_diffs = proj_mat @ loc
        covar_diffs = proj_mat @ covar_mat @ proj_mat.T
        std_diffs = torch.diagonal(covar_diffs + 1e-4).pow(0.5)  # 1e-4 added for numerical stability

        marginal_diffs = torch.distributions.Normal(loc_diffs, std_diffs)
        prob = marginal_diffs.cdf(torch.zeros_like(loc_diffs)).max()

        return prob

    def regress(self, regressor: Regressor, n_preds=5, noise=False):
        """Fit and predict."""

        # Fit the regressor (GP hyperparams are trained)
        regressor.subsample_and_condition(max_train_data=100)
        regressor.fit(num_iter=100)

        # Make n_preds predictions, each conditioned on independently subsampled (at most n_pts) train samples
        n_pts = 500
        pred_xs = torch.linspace(0, 1.0, 500, device=self.device)

        pred_means, pred_covars = [], []

        n_errs = 0
        while len(pred_means) < n_preds:
            try:
                regressor.subsample_and_condition(max_train_data=n_pts)
                with torch.no_grad():
                    pred_dist = regressor.predict(pred_xs, noise=noise)
                    pred_mean = pred_dist.mean.clone()
                    pred_covar = pred_dist.covariance_matrix.clone()

                pred_means.append(pred_mean)
                pred_covars.append(pred_covar)
            except NotPSDError:
                n_errs += 1
                warnings.warn(f'NotPSDError detected (total {n_errs} times)')
                if n_errs >= 2 * n_preds:
                    raise RuntimeError(f'NotPSDError detected {2*n_preds} times')

        pred_means = torch.stack(pred_means)  # shape=[n_preds, 500]
        pred_covars = torch.stack(pred_covars)  # shape=[n_preds, 500, 500]

        return pred_xs, pred_means, pred_covars

    def start_warmup_phase_once(self):
        """Initialize warmup phase. Only once effective."""
        if self.warmup_phase_states['unit'] == 'step':
            max_warmup_steps = int(self.total_steps * self.warmup_phase_states['max_fraction'])
        else:
            max_warmup_steps = int(self.total_epochs * self.warmup_phase_states['max_fraction'])

        if max_warmup_steps <= 0:
            for g in self.optimizer.param_groups:
                if 'initial_lr' in g:
                    del g['initial_lr']
                g['lr'] = self.max_lr
            self.start_decay_phase_once()

        if self.phase not in ['warmup', 'decay', 'final']:
            for g in self.optimizer.param_groups:
                if 'initial_lr' in g:
                    del g['initial_lr']
                g['lr'] = self.min_lr

            if self.warmup_phase_states['type'] == 'linear':
                max_lr, min_lr = self.max_lr, self.min_lr

                # lambda scheduler takes form of base_lr * lr_lambda(step_idx), where base_lr == min_lr here
                def linear_fn(step_idx):
                    x = step_idx / max_warmup_steps
                    return (1-x) + x * (max_lr / min_lr)

                self.scheduler = LambdaLR(self.optimizer, lr_lambda=linear_fn)
            else:
                gamma = (self.max_lr / self.min_lr) ** (1 / max_warmup_steps)
                self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)

            patience = self.warmup_phase_states['patience']
            self._prev_test_results = deque(maxlen=patience)
            self._losses_last_epoch = []

            self.phase = 'warmup'

    def start_decay_phase_once(self):
        """Initialize decay phase. Only once effective."""
        if self.immediate_cooldown:
            self.start_cooldown_phase_once()

        elif self.phase not in ['decay', 'final']:
            for g in self.optimizer.param_groups:
                if 'initial_lr' in g:
                    del g['initial_lr']

            if self.decay_phase_states['stat'] == 'loss':
                self.scheduler = ReduceLROnPlateau(self.optimizer,
                                                   mode='min',
                                                   factor=self.decay_phase_states['decay_factor'],
                                                   patience=5
                                                   )
            else:
                self.scheduler = ExponentialLR(self.optimizer,
                                               gamma=self.decay_phase_states['decay_factor'])

            self._stats_last_epoch = []
            self.phase = 'decay'

    def start_cooldown_phase_once(self):
        """Initialize cooldown phase. Only once effective."""
        if self.phase not in ['final']:
            for g in self.optimizer.param_groups:
                if 'initial_lr' in g:
                    del g['initial_lr']

            self.phase = 'final'

            T_max = self.total_steps - self.last_step

            if T_max == 0:
                # exception handling
                self.scheduler = ExponentialLR(self.optimizer, gamma=1.0)

            elif self.cooldown_phase_states['type'] == 'cosine':
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max)

            elif self.cooldown_phase_states['type'] == 'half_cosine':

                def half_cosine_fn(step_idx):
                    return 0.5 + 0.5 * math.cos(0.5 * math.pi * (1 + step_idx/T_max))

                self.scheduler = LambdaLR(self.optimizer, lr_lambda=half_cosine_fn)

    def get_last_lr(self):
        """Return the last computed learning rates by current scheduler."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            return [g['lr'] for g in self.optimizer.param_groups]
