
import os
import time
import logging

import random
import numpy
import math

from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


# log related
def get_str_datetime():
    """
    """
    str_datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
    return str_datetime

def create_logger(log_tag=None, str_datetime=None, dir_log=None, log_path=None):
    """
    """
    if log_path is None:
        if str_datetime is None:
            str_datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
        #
        filename = "log_%s_%s.txt" % (log_tag, str_datetime)
        log_path = os.path.abspath(os.path.join(dir_log, filename) )
        #
    #
    logger = logging.getLogger(log_path)  # use log_path as log_name
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.info('test')
    return logger

def close_logger(logger):
    for item in logger.handlers:
        item.close()
        print("logger handler item closed")

# frozen related
def frozen_modules(modules_frozen, except_list=[], print_name=False):
    """
    """
    for module in modules_frozen:
        for name, value in module.named_parameters():
            if name in except_list:
                if print_name:
                    print("training: %s" % name)
                value.requires_grad = True
            else:
                if print_name:
                    print("not training: %s" % name)
                value.requires_grad = False
                #
    #
    print("frozen_modules finished")
    #
#
def unfrozen_modules(modules_train, except_list=[], print_name=False):
    """
    """
    for module in modules_train:
        for name, value in module.named_parameters():
            if name in except_list:
                if print_name:
                    print("not training: %s" % name)
                value.requires_grad = False
            else:
                if print_name:
                    print("training: %s" % name)
                value.requires_grad = True
                #
    #
    print("unfrozen_modules finished")
    #
#
def set_variable_trainable(module, list_variable, is_trainable):
    """
    """
    for name, p in module.named_parameters():
        if name in list_variable:
            p.requires_grad = bool(is_trainable)
    #
#            

# params related
def print_params(module, print_name=False):
    """
    """
    n_tr, n_nontr = 0, 0
    for p in module.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_tr += n_params
        else:
            n_nontr += n_params
    #
    print("n_trainable_params: %d, n_nontrainable_params: %d" % (n_tr, n_nontr) )
    #
    if print_name:
        for name, value in module.named_parameters():
            if value.requires_grad:
                print("training: %s" % name)
            else:
                print("not training: %s" % name)
    #
    return n_tr, n_nontr
    #
#
def set_random_seed(seed):
    """
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #
#
def initialize_parameters(module=None, named_parameters=None,
                init_method=torch.nn.init.xavier_normal_,
                except_list=[], flag_print=False):
    """ named_parameters: list of (name, parameter)
    """
    if named_parameters is None:
        if module is None:
            assert False, "module and named_parameters cannot both be None"
        #
        named_parameters = module.named_parameters()
        #
    #
    n_reset, n_unreset = 0, 0
    #
    for name, p in named_parameters:
        n_params = torch.prod(torch.tensor(p.shape)).item()
        #
        if name in except_list:        # not reset
            n_unreset += n_params
            if flag_print:
                print("not reset: %s" % name)
        else: #       if p.requires_grad:
            if len(p.shape) > 1:
                init_method(p)
                if flag_print:
                    print("reset: %s" % name)
            else:
                stdv = 1. / math.sqrt(p.shape[0])
                torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                if flag_print:
                    print("reset: %s" % name)
            #
            n_reset += n_params
            #
        #
    #
    print("n_reset, n_unreset: %d, %d" % (n_reset, n_unreset) )
    #
#
def load_parameters_value_from_saved(model, model_saved_path=None, state_dict=None,
                    prefix_dict={}, flag_print=False):
    """ prefix_dict: name_in_model --> name_in_saved
    """
    if state_dict is None:
        if model_saved_path is None:
            print("model_saved_path and state_dict both be None")
            #
            return model.named_parameters()
            #
        try:
            state_dict = torch.load(model_saved_path, map_location="cpu")
        except Exception:
            print("Error when loading %s" % model_saved_path)
            #
            return model.named_parameters()
            #
    #
    # convert old format to new format if needed from a PyTorch state_dict
    #
    if flag_print:
        print("keys in saved_model state_dict:")
    #
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        #
        if flag_print:
            print(key)
        #
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    #
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    #
    # map and load
    #
    missing_keys = []
    #
    def get_key_in_saved_state_dict(key, prefix_dict):
        """
        """
        key_saved = key
        #
        for item_k, item_v in prefix_dict.items():
            if len(item_k) == 0:
                key_saved = item_v + key
            elif key.startswith(item_k):
                key_saved = item_v + key.lstrip(item_k)
                break
        #
        return key_saved
        #
    #
    for name, p in model.named_parameters():
        name_saved = get_key_in_saved_state_dict(name, prefix_dict)
        #
        if name_saved in state_dict:
            if flag_print:
                print("loaded: %s" % name)
            p.data.copy_(state_dict[name_saved].data)
        else:
            if flag_print:
                print("missing: %s" % name)
            missing_keys.append( (name, p) )
        #
    #
    print("parameters loaded, missing parameters: %d" % len(missing_keys))
    #
    # if flag_print:
    for item in missing_keys:
        print("missing keys: %s" % item[0])
    #
    return missing_keys
    #
#
def initialize_and_load(model, model_saved_path=None, state_dict=None, prefix_dict={},
                       init_method=torch.nn.init.xavier_normal_, except_list=[],
                       flag_print=True):
    """ prefix_dict: name_in_model --> name_in_saved
    """
    missing_parameters = load_parameters_value_from_saved(model, 
        model_saved_path=model_saved_path, state_dict=state_dict,
        prefix_dict=prefix_dict, flag_print=flag_print)
    #
    initialize_parameters(named_parameters=missing_parameters, init_method=init_method,
        except_list=except_list, flag_print=flag_print)
    #
#

#
class ModelEvaluator(object):
    """
    """
    def __init__(self, settings, model, eval_dataset, scores_calculator=None):
        """ model: forward function output tuple, loss_train = outputs[0]
        """
        # settings
        self.settings = settings
        #
        self.num_workers = self.settings.num_workers
        self.device = self.settings.device
        #
        # model
        self.model = model.to(self.device)
        self.scores_calculator = scores_calculator
        #
        # data
        self.eval_dataset = eval_dataset
        #
        eval_sampler = SequentialSampler(eval_dataset)
        #
        self.eval_dataloader = DataLoader(eval_dataset, num_workers=self.num_workers,
            sampler=eval_sampler, batch_size=self.settings.eval_batch_size)
        #
        # iterator info
        self.step_iterator_info = {
            "description": "Step",
            "disable": False
        }
        #
    #
    def initialize_and_load(self, model_saved_path=None, state_dict=None,
                            prefix_dict={}, init_method=torch.nn.init.xavier_normal_,
                            except_list=[], flag_print=False):
        """
        """
        if model_saved_path is None and state_dict is None:
            initialize_parameters(
                self.model, init_method=init_method, except_list=except_list,
                flag_print=flag_print )
            #
            return
            #
        #
        initialize_and_load(
            self.model, model_saved_path=model_saved_path, state_dict=state_dict,
            prefix_dict=prefix_dict, init_method=init_method,
            except_list=except_list, flag_print=flag_print )
        #
    #
    def evaluate(self):
        """
        """
        list_batches_evaluated = []
        #
        step_iterator = tqdm(
            self.eval_dataloader,
            desc = self.step_iterator_info["description"],
            disable = self.step_iterator_info["disable"]
        )
        for step, batch in enumerate(step_iterator):
            self.model.eval()
            #
            batch_d = [item.to(self.device) for item in batch]
            #
            outputs = self.model(*batch_d, is_training=False)
            #
            batch_list = [item.detach().cpu().numpy() for item in batch]
            out_list = [item.detach().cpu().numpy() for item in outputs]
            #
            list_batches_evaluated.append( (batch_list, out_list) )
            #
        #
        self.settings.logger.info("evaluation iteration finished.")
        #
        if self.scores_calculator:
            main_score, scores_dict = self.scores_calculator(list_batches_evaluated)
        else:
            main_score = 0.0
            scores_dict = {}
        #
        return main_score, scores_dict, list_batches_evaluated
        #
    #
#
class ModelTrainer(object):
    """
    """
    def __init__(self, settings, model, train_dataset):
        """ model: forward function output tuple, loss_train = outputs[0]
            dataloader: output batch_dict with keys as model forward arguments
        """
        # settings
        self.settings = settings
        self.logger = self.settings.logger
        #
        self.model_saved_path = self.settings.model_saved_path
        #
        self.num_accum_steps = self.settings.num_accum_steps
        self.num_gpu = self.settings.num_gpu
        #
        self.num_epochs = self.settings.num_epochs
        self.grad_norm_clip = self.settings.grad_norm_clip
        #
        self.num_workers = self.settings.num_workers
        self.device = self.settings.device
        #
        # model
        if self.num_gpu > 1:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model.to(self.device)
        #
        # data
        self.train_dataset = train_dataset
        #
        train_sampler = RandomSampler(train_dataset)
        # train_sampler = DistributedSampler(train_dataset)
        #
        if self.num_accum_steps > 1:
            batch_size = self.settings.train_batch_size // self.num_accum_steps
        else:
            batch_size = self.settings.train_batch_size
        #
        self.train_dataloader = DataLoader(train_dataset, num_workers=self.num_workers,
            sampler=train_sampler, batch_size=batch_size)
        #
        # iterator info
        self.epoch_iterator_info = {
            "num_epochs": self.num_epochs,
            "description": "Epoch",
            "disable": False
        }
        self.step_iterator_info = {
            "description": "Step",
            "disable": False
        }
        #
    #
    def initialize_and_load(self, model_saved_path=None, state_dict=None,
                            prefix_dict={}, init_method=torch.nn.init.xavier_normal_,
                            except_list=[], flag_print=False):
        """
        """
        if model_saved_path is None and state_dict is None:
            initialize_parameters(
                self.model, init_method=init_method, except_list=except_list,
                flag_print=flag_print )
            #
            return
            #
        #
        initialize_and_load(
            self.model, model_saved_path=model_saved_path, state_dict=state_dict,
            prefix_dict=prefix_dict, init_method=init_method,
            except_list=except_list, flag_print=flag_print )
        #
    #

    #
    def train_with_scheduler(self, optimizer, scheduler):
        """
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        #
        if self.num_accum_steps > 1:
            self._train_multi_step_single_gpu()
        elif self.num_gpu > 1:
            self._train_single_step_multi_gpu()
        else:
            self._train_single_step_single_gpu()
        #
    #
    def _train_single_step_single_gpu(self):
        """
        """
        # global_step = 0
        self.model.zero_grad()
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                self.model.train()
                #
                # batch = tuple(t.to(config.device) for t in batch)
                # inputs = {'input_ids':      batch[0],
                #         'attention_mask': batch[1],
                #         # XLM and RoBERTa don't use segment_ids
                #         'token_type_ids': batch[2],
                #         'labels':      batch[3],
                #         'e1_mask': batch[4],
                #         'e2_mask': batch[5],
                #         }
                #
                # outputs = self.model(**inputs)
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                # model outputs are always tuple in pytorch-transformers (see doc)
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #

                #
                # single-step, single-gpu
                #
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                # global_step += 1
                #
            #
            # save
            # self.model_saved_path_last = self.model_saved_path + "-epoch-%d" % (epoch + 1)
            torch.save(self.model.state_dict(), self.model_saved_path)
            #
            self.logger.info("model saved after the %d-th epoch" % epoch)
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d epochs" % self.num_epochs)
        #
        return 0
        #
    #
    def _train_multi_step_single_gpu(self):
        """
        """
        # global_step = 0
        self.model.zero_grad()
        index_update = self.num_accum_steps - 1
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                self.model.train()
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #
                # if config.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # if config.gradient_accumulation_steps > 1:
                #

                #
                # multi-step, single-gpu
                #
                loss = loss / self.num_accum_steps
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                if step % self.num_accum_steps == index_update:
                    #
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    # global_step += 1
                    #
                #

                #
                # if config.local_rank in [-1, 0] and config.save_steps > 0 and global_step % config.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(
                #         config.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     # Take care of distributed/parallel training
                #     model_to_save = model.module if hasattr(
                #         model, 'module') else model
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(config, os.path.join(
                #         output_dir, 'training_config.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)
                #
            #
            # save
            # self.model_saved_path_last = self.model_saved_path + "-epoch-%d" % (epoch + 1)
            torch.save(self.model.state_dict(), self.model_saved_path)
            #
            self.logger.info("model saved after the %d-th epoch" % epoch)
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d epochs" % self.num_epochs)
        #
        return 0
        #
    #
    def _train_single_step_multi_gpu(self):
        """
        """
        # global_step = 0
        self.model.zero_grad()
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                self.model.train()
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #

                # if config.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # if config.gradient_accumulation_steps > 1:
                #     loss = loss / config.gradient_accumulation_steps
                
                #
                # single-step, multi-gpu
                #
                loss = loss.mean()
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                # global_step += 1
                #
            #
            # save
            # self.model_saved_path_last = self.model_saved_path + "-epoch-%d" % (epoch + 1)
            torch.save(self.model.state_dict(), self.model_saved_path)
            #
            self.logger.info("model saved after the %d-th epoch" % epoch)
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d epochs" % self.num_epochs)
        #
        return 0
        #
    #

    #
    def train_with_adaptive_decay(self, optimizer_creator, model_evaluator):
        """
        """
        self.optimizer_creator = optimizer_creator
        self.model_evaluator = model_evaluator
        # self.model_saved_path_last = None
        #
        self.checkpoint_steps = self.settings.checkpoint_steps
        self.lr_current = self.settings.learning_rate
        self.lr_decay_rate = self.settings.lr_decay_rate
        self.lr_min = self.settings.lr_min
        self.decay_tolerance = self.settings.decay_tolerance
        #
        self.optimizer = self.optimizer_creator(self.lr_current)
        #
        if self.num_accum_steps > 1:
            self._train_multi_step_single_gpu_adaptive()
        elif self.num_gpu > 1:
            self._train_single_step_multi_gpu_adaptive()
        else:
            self._train_single_step_single_gpu_adaptive()
        #
    #
    def _train_single_step_single_gpu_adaptive(self):
        """
        """
        global_step = 0
        score_best = 0.0
        count_score_stagnant = 0
        flag_stop = 0
        index_update = self.num_accum_steps - 1
        flag_eval = 1
        #
        self.model.zero_grad()
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                #
                if global_step % self.checkpoint_steps == 0:
                    #
                    self.model_evaluator.initialize_and_load(
                        state_dict=self.model.state_dict() )
                    main_score, scores, _ = self.model_evaluator.evaluate()
                    #
                    if main_score > score_best:
                        score_best = main_score
                        count_score_stagnant = 0
                        #
                        # save
                        # self.model_saved_path_last = self.model_saved_path + "-step-%d" % (
                        #     global_step + 1)
                        torch.save(self.model.state_dict(), self.model_saved_path)
                        #
                        self.logger.info("model saved after the %d-th step" % global_step)
                        self.logger.info("scores: {}".format(scores))
                        self.logger.info("main_score: {}".format(main_score))
                        #
                    else:
                        count_score_stagnant += 1
                        #
                        if count_score_stagnant >= self.decay_tolerance:
                            count_score_stagnant = 0
                            #
                            self.lr_current *= self.lr_decay_rate
                            #
                            if self.lr_current < self.lr_min:
                                flag_stop = 1
                                self.logger.info("lr decayed below lr_min, stop training.")
                                break
                                #
                            #
                            self.optimizer = self.optimizer_creator(self.lr_current)
                            #
                            self.logger.info("learning rate decayed, curr: {}".format(
                                self.lr_current) )
                            #
                        #
                    #
                #

                #
                self.model.train()
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #

                #
                # single-step, single-gpu
                #
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                self.optimizer.step()
                # self.scheduler.step()
                self.model.zero_grad()
                global_step += 1
                #
            #
            if flag_stop:
                break
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d steps" % global_step)
        #
        return 0
        #
    #
    def _train_multi_step_single_gpu_adaptive(self):
        """
        """
        global_step = 0
        score_best = 0.0
        count_score_stagnant = 0
        flag_stop = 0
        index_update = self.num_accum_steps - 1
        flag_eval = 1
        #
        self.model.zero_grad()
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                #
                if global_step % self.checkpoint_steps == 0 and flag_eval:
                    #
                    flag_eval = 0
                    #
                    self.model_evaluator.initialize_and_load(
                        state_dict=self.model.state_dict() )
                    main_score, scores, _ = self.model_evaluator.evaluate()
                    #
                    if main_score > score_best:
                        score_best = main_score
                        count_score_stagnant = 0
                        #
                        # save
                        # self.model_saved_path_last = self.model_saved_path + "-step-%d" % (
                        #     global_step + 1)
                        torch.save(self.model.state_dict(), self.model_saved_path)
                        #
                        self.logger.info("model saved after the %d-th step" % global_step)
                        self.logger.info("scores: {}".format(scores))
                        self.logger.info("main_score: {}".format(main_score))
                        #
                    else:
                        count_score_stagnant += 1
                        #
                        if count_score_stagnant >= self.decay_tolerance:
                            count_score_stagnant = 0
                            #
                            self.lr_current *= self.lr_decay_rate
                            #
                            if self.lr_current < self.lr_min:
                                flag_stop = 1
                                self.logger.info("lr decayed below lr_min, stop training.")
                                break
                                #
                            #
                            self.optimizer = self.optimizer_creator(self.lr_current)
                            #
                            self.logger.info("learning rate decayed, curr: {}".format(
                                self.lr_current) )
                            #
                        #
                    #
                #

                #
                self.model.train()
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #

                #
                # multi-step, single-gpu
                #
                loss = loss / self.num_accum_steps
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                if step % self.num_accum_steps == index_update:
                    #
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    #
                    flag_eval = 1
                    #
                #
            #
            if flag_stop:
                break
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d steps" % global_step)
        #
        return 0
        #
    #
    def _train_single_step_multi_gpu_adaptive(self):
        """
        """
        global_step = 0
        score_best = 0.0
        count_score_stagnant = 0
        flag_stop = 0
        index_update = self.num_accum_steps - 1
        flag_eval = 1
        #
        self.model.zero_grad()
        #
        epoch_iterator = trange(
            int(self.epoch_iterator_info["num_epochs"]),
            desc = self.epoch_iterator_info["description"],
            disable = self.epoch_iterator_info["disable"]
        )
        #
        for epoch in epoch_iterator:
            step_iterator = tqdm(
                self.train_dataloader,
                desc = self.step_iterator_info["description"],
                disable = self.step_iterator_info["disable"]
            )
            for step, batch in enumerate(step_iterator):
                #
                if global_step % self.checkpoint_steps == 0:
                    #
                    self.model_evaluator.initialize_and_load(
                        state_dict=self.model.state_dict() )
                    main_score, scores, _ = self.model_evaluator.evaluate()
                    #
                    if main_score > score_best:
                        score_best = main_score
                        count_score_stagnant = 0
                        #
                        # save
                        # self.model_saved_path_last = self.model_saved_path + "-step-%d" % (
                        #     global_step + 1)
                        torch.save(self.model.state_dict(), self.model_saved_path)
                        #
                        self.logger.info("model saved after the %d-th step" % global_step)
                        self.logger.info("scores: {}".format(scores))
                        self.logger.info("main_score: {}".format(main_score))
                        #
                    else:
                        count_score_stagnant += 1
                        #
                        if count_score_stagnant >= self.decay_tolerance:
                            count_score_stagnant = 0
                            #
                            self.lr_current *= self.lr_decay_rate
                            #
                            if self.lr_current < self.lr_min:
                                flag_stop = 1
                                self.logger.info("lr decayed below lr_min, stop training.")
                                break
                                #
                            #
                            self.optimizer = self.optimizer_creator(self.lr_current)
                            #
                            self.logger.info("learning rate decayed, curr: {}".format(
                                self.lr_current) )
                            #
                        #
                    #
                #

                #
                self.model.train()
                #
                batch_d = [item.to(self.device) for item in batch]
                #
                outputs = self.model(*batch_d)
                loss = outputs[0]
                #

                #
                # single-step, multi-gpu
                #
                loss = loss.mean()
                loss.backward()
                #
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_norm_clip)
                #
                self.optimizer.step()
                # self.scheduler.step()
                self.model.zero_grad()
                global_step += 1
                #
            #
            if flag_stop:
                break
            #
        #
        # save
        torch.save(self.model.state_dict(), self.model_saved_path)
        #
        self.logger.info("model saved after %d steps" % global_step)
        #
        return 0
        #
    #
#


