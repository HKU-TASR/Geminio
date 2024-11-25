import numpy
import torch
import time
import os

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup
import torchvision.transforms as transform

import logging

log = logging.getLogger(__name__)


class OptimizationBasedAttacker(_BaseAttacker):

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        except AttributeError:
            pass  # No regularizers selected.

        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.

    def __repr__(self):
        n = "\n"
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Hyperparameter Template: {self.cfg.type}

    Objective: {repr(self.objective)}
    Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False, custom=None):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            if self.cfg.regularization.group_regular.scale < 1e-8:
                for trial in range(self.cfg.restarts.num_trials):
                    candidate_solutions += [
                        self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun, custom)
                    ]
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
            else:
                candidate_solutions += self._run_trial2(rec_models, shared_data, labels, stats, 0, initial_data, dryrun, custom)
                scores = torch.zeros(len(candidate_solutions))
                for seed in range(len(candidate_solutions)):
                    scores[seed] = self._score_trial(candidate_solutions[seed], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False, custom=None):
        if 'gaussianblur' in self.cfg.keys():
            gabl = self.cfg.gaussianblur
            self.gaussian_post = transform.GaussianBlur(gabl.radius, gabl.post_std)
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.pixel_decay > 0:
                        candidate.data = (1-self.cfg.optim.pixel_decay) * candidate.data
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp
                    if custom is not None:
                        if "save_dir" not in self.cfg.keys():
                            save_path = f'/home/mxj/PycharmProjects/breaching/custom_data/recons/img4x1_ddpm{iteration+1}.jpg'
                        else:
                            save_path = self.cfg.save_dir + f'img4x1_ddpm{iteration+1}.jpg'
                        custom.save_recover(best_candidate, save_pth=save_path)


                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach()

    def _run_trial2(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False, custom=None):
        if 'gaussianblur' in self.cfg.keys():
            gabl = self.cfg.gaussianblur
            self.gaussian_tiny = transform.GaussianBlur(gabl.radius, gabl.tiny_std)
            self.gaussian_large = transform.GaussianBlur(gabl.radius, gabl.large_std)

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
        best_seeds = [i for i in range(self.cfg.regularization.group_regular.totalseeds)]
        candidate_all = []
        minimal_list = []
        ifWeight = self.cfg.regularization.group_regular.weighted

        # main reconstruct part
        # Initialize candidate reconstruction data
        for seed in range(self.cfg.regularization.group_regular.totalseeds):
            candidate_tmp = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
            candidate_all.append(candidate_tmp)
            minimal_list.append(torch.as_tensor(float("inf"), **self.setup))
        if initial_data is not None:
            candidate_all = []
            for seed in range(self.cfg.regularization.group_regular.totalseeds):
                if isinstance(initial_data, list):
                    candidate_all.append(initial_data[seed].data.clone().to(**self.setup))
                    candidate_all[seed].requires_grad = True
                else:
                    candidate_all.append(initial_data.data.clone().to(**self.setup))
                    candidate_all[seed].requires_grad = True

        # best_candidate = candidate_all[0].detach().clone()

        # Initialize optimizers
        optimizer, scheduler = [], []
        group_reg = self.cfg.regularization.group_regular
        calAvg = calAvgImage(custom.mean, custom.std, group_reg.mode)
        avgImage = calAvg.get_lazy_avg(candidate_all)
        if 'distance_constrain' in self.cfg.optim.keys():
            self.gradDe = gradDecay(self.data_shape[1], self.cfg.optim.distance_constrain)
        else: self.gradDe = gradDecay(self.data_shape[1])
        for seed in range(group_reg.totalseeds):
            opt_tmp, sch_tmp = self._init_optimizer([candidate_all[seed]])
            optimizer.append(opt_tmp)
            scheduler.append(sch_tmp)
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                # get avg images
                if iteration >= group_reg.startIter and (iteration-group_reg.startIter) % group_reg.updateRegPeriod == 0:
                    if group_reg.mode == "lazy":
                        avgImage = calAvg.get_lazy_avg(candidate_all)
                    else: avgImage = calAvg.get_reg_avg(candidate_all, minimal_list, ifWeight)
                # if iteration % self.cfg.optim.callback == 0:
                #     if "save_dir" not in self.cfg.keys():
                #         save_path = f'/home/mxj/PycharmProjects/breaching/custom_data/recons/img_avg_{iteration+1}.jpg'
                #     else:
                #         if not os.path.exists(self.cfg.save_dir):
                #             os.mkdir(self.cfg.save_dir)
                #         save_path = self.cfg.save_dir + f'img_avg_{iteration+1}.jpg'
                #     custom.save_recover(avgImage, save_pth=save_path)

                for seed in range(group_reg.totalseeds):
                    candidate = candidate_all[seed]
                    closure = self._compute_objective2(candidate, labels, rec_model, optimizer[seed], shared_data, iteration, avgImage)
                    objective_value, task_loss = optimizer[seed].step(closure), self.current_task_loss
                    scheduler[seed].step()

                    with torch.no_grad():
                        # Project into image space
                        if self.cfg.optim.pixel_decay > 0:
                            candidate.data = (1-self.cfg.optim.pixel_decay) * candidate.data
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        if "peroid_Add10" in self.cfg.objective.keys():
                            if objective_value < minimal_list[seed] or (iteration+1) % self.cfg.objective.peroid_Add10 == 0:
                                minimal_list[seed] = objective_value.detach()
                                best_candidate = candidate.detach().clone()
                                best_seeds[seed] = best_candidate
                        elif objective_value < minimal_list[seed]:
                            minimal_list[seed] = objective_value.detach()
                            best_candidate = candidate.detach().clone()
                            best_seeds[seed] = best_candidate

                    if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                        timestamp = time.time()
                        log.info(
                            f"|seed: {seed} | It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                        current_wallclock = timestamp
                        if custom is not None:
                            if "save_dir" not in self.cfg.keys():
                                save_path = f'/home/mxj/PycharmProjects/breaching/custom_data/recons/img_{iteration+1}.jpg'
                            else:
                                if not os.path.exists(self.cfg.save_dir):
                                    os.mkdir(self.cfg.save_dir)
                                save_path = self.cfg.save_dir + f'img_{iteration+1}.jpg'
                            # if 'sat_ratio' not in self.cfg.keys() or iteration+1 < self.cfg.regularization.deep_inversion.deep_inv_start+self.cfg.optim.callback:
                            custom.save_recover(best_seeds[seed], save_pth=save_path)
                            # else:
                            #     custom.save_recover(best_seeds[seed], save_pth=save_path, sature=self.cfg.sat_ratio)

                    if not torch.isfinite(objective_value):
                        log.info(f"Recovery loss is non-finite in iteration {iteration}, seed {seed}. Cancelling reconstruction!")
                        break

                    stats[f"Trial_{trial}_Val"].append(objective_value.item())

                    if dryrun:
                        break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in seed {seed}, iteration {iteration}!")
            pass

        return best_seeds

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            candidate_augmented = candidate
            # if iteration % 2000 == 0 and iteration <= 10000:
            if iteration % 200 == 0 and iteration <= 10000 and 'gaussianblur' in self.cfg.keys():
                if self.cfg.differentiable_augmentations:
                    candidate_augmented = self.gaussian_post(candidate)
                else:
                    candidate_augmented = candidate
                    candidate_augmented.data = self.gaussian_post(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _compute_objective2(self, candidate, labels, rec_model, optimizer, shared_data, iteration, avgImage):
        def closure():
            optimizer.zero_grad()

            candidate_augmented = candidate
            # if iteration % 200 == 0 and iteration <= 10000 and iteration > 2000:
            #     if self.cfg.differentiable_augmentations:
            #         candidate_augmented = self.augmentations(candidate)
            #     else:
            #         candidate_augmented = candidate
            #         candidate_augmented.data = self.augmentations(candidate.data)
            if iteration % 500 == 0 and iteration <= 4000 and iteration > self.cfg.gaussianblur_start and 'gaussianblur' in self.cfg.keys():
                if self.cfg.differentiable_augmentations:
                    candidate_augmented = self.gaussian_tiny(candidate)
                else:
                    candidate_augmented = candidate
                    candidate_augmented.data = self.gaussian_tiny(candidate.data)
            # elif iteration % 10000 == 0 and iteration < 10000 and iteration > self.cfg.gaussianblur_start and 'gaussianblur' in self.cfg.keys():
            #     if self.cfg.differentiable_augmentations:
            #         candidate_augmented = self.gaussian_large(candidate)
            #     else:
            #         candidate_augmented = candidate
            #         candidate_augmented.data = self.gaussian_large(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented, avgImg=avgImage, iter=iteration)

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                                1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    elif self.cfg.optim.signed == "hard_quantify":
                        self._optim_signed_quantify(candidate, iteration, mul_scale=self.cfg.optim.mul_scale)
                    elif self.cfg.optim.signed == "hard_quantify2":
                        self._optim_signed_quantify2(candidate, iteration, mul_scale=self.cfg.optim.mul_scale)
                    else:
                        pass
                if 'distance_constrain' in self.cfg.optim.keys() and iteration <= self.cfg.optim.distance_constrain.stop_iter:
                    self.gradDe.decay_grad(candidate)

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _optim_signed_quantify(self, candidate, iteration, level=4, mul_scale=2):
        # max = torch.max(candidate.grad).abs_()
        # min = torch.min(candidate.grad).abs_()
        # scale = level / 2 / max if max > min else level / 2 / min
        median = torch.median(torch.abs(candidate.grad))  # 1 for batch 8
        scale = level/2/(median+1e-10)
        alpha = 0.2
        tmp = torch.sign(candidate.grad) * alpha
        candidate.grad.mul_(scale).round_().div_(level/2.).clamp_(-1, 1).mul_(1-alpha)
        candidate.grad += tmp
        if iteration <= 100: candidate.grad.mul_(mul_scale)

    def _optim_signed_quantify2(self, candidate, iteration, level=4, mul_scale=2):
        if iteration <= 150:
            candidate.grad.sign_().mul_(mul_scale)
            return 0
        median = torch.median(torch.abs(candidate.grad))  # 1 for batch 8
        scale = level/2/(median+1e-10)
        alpha = 0.3
        tmp = torch.sign(candidate.grad) * alpha
        candidate.grad.mul_(scale).round_().div_(level/2.).clamp_(-1, 1).mul_(1-alpha)
        candidate.grad += tmp

    def _score_trial(self, candidate, labels, rec_model, shared_data):

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)


class calAvgImage:
    def __init__(self, mean, std, if_lazy):
        if if_lazy != 'lazy':
            self.alignMethod = imageAlign()
        self.mean = mean.cpu()
        self.std = std.cpu()
        self.trans = transform.Resize(224)

    def get_lazy_avg(self, candidate, scoreList = None, ifWeight = False):
        if not ifWeight:
            return (sum(candidate)/len(candidate)).data.clone()
        else:
            tmp = torch.zeros_like(candidate[0])
            minScore = min(scoreList)
            minIdx = scoreList.index(minScore)
            w0 = 0.25 if len(scoreList) > 2 else 1.25-0.25*(len(scoreList))
            w1 = (1-w0)/(len(scoreList)-1) if len(scoreList) > 1 else 0
            wList = [w1]*len(scoreList)
            wList[minIdx] = w0
            for x in zip(candidate, wList):
                tmp += x[0].data.clone() * x[1]
            return tmp

    def conver_to_0_1(self, tensor):
        return torch.clamp((tensor*self.std+self.mean), 0, 1)

    def back_to_normalized(self, tensor):
        return ((tensor.cpu()-self.mean)/self.std).cuda()

    def get_reg_avg(self, candidate, scoreList=None, ifWeight = False):
        aligned_all = []
        seeds = len(candidate)
        targetImg = self.get_lazy_avg(candidate, scoreList, ifWeight).cpu()
        targetImg = self.conver_to_0_1(targetImg)
        num_points = targetImg.shape[0]
        for seed in range(seeds):
            sourceImg = candidate[seed].detach().clone().cpu()
            sourceImg = self.conver_to_0_1(sourceImg)
            tmp_points = []
            for i in range(num_points):
                res = self.alignMethod.alignImages(sourceImg[i], targetImg[i], isTensor=True)
                tmp = self.back_to_normalized(self.trans(res['alignedTensor']))
                tmp_points.append(tmp)
            tmp_seed = torch.cat(tmp_points, dim=0)
            aligned_all.append(tmp_seed)
        regAvg = self.get_lazy_avg(aligned_all, scoreList, ifWeight).cuda()
        return regAvg

class gradDecay:
    def __init__(self, width=224, dc = None):
        self.width = width
        self.decay_dis_rate = 1 if dc is None else dc.decay_dis_rate
        self.dc = dc
        self.get_decay_index()

    def get_decay_index(self):
        x = []
        y = []
        halfW = self.width/2
        square_max_dis = (halfW * self.decay_dis_rate) ** 2
        for i in range(self.width):
            for j in range(self.width):
                square_dis = (i-halfW)**2 + (j-halfW)**2
                if square_dis > square_max_dis:
                    x.append(i)
                    y.append(j)
        self.x = x
        self.y = y

    def decay_grad(self, candidate):
        candidate.grad[:,:,self.x,self.y] *= self.dc.decay_rate
