
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import argparse
import collections

import torch

from models.prelm.bert_configuration import BertConfig
from models.prelm.bert_tokenization import BertTokenizer

from models.torch_utils.optimization import AdamW
from models.torch_utils.optimization import get_linear_schedule_with_warmup
from models.torch_utils.optimization import get_constant_schedule_with_warmup
from models.torch_utils.optimization import get_constant_schedule
from models.torch_utils.settings_baseboard import SettingsBaseboard
from models.torch_utils.model_utils import ModelTrainer, ModelEvaluator
from models.torch_utils.model_utils import set_random_seed

from models.model_classifier import ModelClassifier
from models.model_classifier import scores_calculator
from models.model_classifier import dataset_creator

from training_simple_dataset import SimProcessor


#
data_processors = {
    "sim": SimProcessor,
}

#
task_name = "sim"
data_dir = "../data_sim"
output_dir = "../model_sim"
#
bert_config_file = "../pretrained_bert_base_chinese/bert_config.json"
vocab_file = "../pretrained_bert_base_chinese/vocab.txt"
init_checkpoint = "../pretrained_bert_base_chinese/bert-base-chinese-pytorch_model.bin"
#
max_seq_len = 20
checkpoint_steps = 5
adaptive_decay = 1
#
with_multibatch = 0
grad_accum_steps = 4
train_batch_size = 32
num_workers = 0
#
do_train = 1
do_eval = 1
do_predict = 0
#
gpu_id = "0"
#
seed = 12345
#


#
def parsed_args():
    """
    """
    # Hyper Parameters
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task_name', default=task_name, type=str)
    parser.add_argument('--data_dir', default=data_dir, type=str)
    parser.add_argument('--output_dir', default=output_dir, type=str)
    #
    parser.add_argument('--vocab_file', default=vocab_file, type=str)
    parser.add_argument('--bert_config_file', default=bert_config_file, type=str)
    parser.add_argument('--init_checkpoint', default=init_checkpoint, type=str)
    #
    parser.add_argument('--do_lower_case', default=1, type=int)            # bool
    parser.add_argument('--max_seq_len', default=max_seq_len, type=int)
    #
    parser.add_argument('--with_multibatch', default=with_multibatch, type=int)   # bool
    parser.add_argument('--grad_accum_steps', default=grad_accum_steps, type=int)
    #
    parser.add_argument('--gpu_id', default=gpu_id, type=str)
    parser.add_argument('--device_type', default=None, type=str)
    parser.add_argument('--seed', default=seed, type=int)
    #
    parser.add_argument('--do_train', default=do_train, type=int)         # bool
    parser.add_argument('--do_eval', default=do_eval, type=int)           # bool
    parser.add_argument('--do_predict', default=do_predict, type=int)     # bool
    #
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--train_batch_size', default=train_batch_size, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--predict_batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=num_workers, type=int)
    #
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--grad_norm_clip', default=1.0, type=float)
    parser.add_argument('--checkpoint_steps', default=checkpoint_steps, type=int)
    #
    parser.add_argument('--adaptive_decay', default=adaptive_decay, type=int)  # bool
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_min', default=1e-6, type=float)
    parser.add_argument('--decay_tolerance', default=6, type=int)
    #
    args = parser.parse_args()
    #
    return args
    #
#
def main(args):
    """
    """
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #
    settings = SettingsBaseboard()
    settings.assign_info_from_namedspace(args)
    #
    log_path = "log_%s_%s.txt" % (settings.task_name, settings.str_datetime)
    logger = settings.create_logger(log_path)
    #
    if settings.device_type is None:
        settings.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        settings.device = torch.device(settings.device_type)
    #

    #
    # bert settings
    #
    bert_config = BertConfig()
    bert_config.from_json_file(settings.bert_config_file)
    tokenizer = BertTokenizer(settings.vocab_file)
    #
    settings.bert_config = bert_config
    #
    #
    # task settings
    #
    task_name = settings.task_name.lower()
    if task_name not in data_processors:
        raise ValueError("Task not found: %s" % (task_name))
    #
    processor = data_processors[task_name]()
    label_list = processor.get_labels()
    try:
        class_weights = processor.get_class_weights()
    except:
        class_weights = None
    #
    num_labels = len(label_list)
    #
    settings.num_labels = num_labels
    #

    # train settings
    settings.num_gpu = len(settings.gpu_id.strip().split(","))
    num_accum_steps = settings.grad_accum_steps if settings.with_multibatch else 0
    settings.num_accum_steps = num_accum_steps
    #
    if not os.path.exists(settings.output_dir):
        os.mkdir(settings.output_dir)
    #
    filename_model_saved = "model_%s.bin" % settings.task_name
    settings.model_saved_path = os.path.join(settings.output_dir, filename_model_saved)
    #
    set_random_seed(settings.seed)
    #

    #
    logger.info("settings: {}".format(settings))
    #

    #
    if settings.do_train:
        train_examples = processor.get_train_examples(settings.data_dir)
        num_train_steps = int(
            len(train_examples) / settings.train_batch_size * settings.num_epochs)
        num_warmup_steps = int(num_train_steps * settings.warmup_proportion)
        #
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", settings.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        #

        #
        train_dataset = dataset_creator(tokenizer, train_examples, label_list,
            max_seq_len=settings.max_seq_len, logger=settings.logger)
        #
        model = ModelClassifier(settings)
        model_trainer = ModelTrainer(settings, model, train_dataset)
        model_trainer.initialize_and_load(
            model_saved_path=settings.init_checkpoint,
            flag_print=True
        )
        #

        #
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.0001
        #
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        #
        if settings.adaptive_decay:
            #
            eval_examples = processor.get_dev_examples(settings.data_dir)
            eval_dataset = dataset_creator(tokenizer, eval_examples, label_list,
                max_seq_len=settings.max_seq_len, logger=settings.logger)
            #
            def optimizer_creator(lr):
                return AdamW(optimizer_grouped_parameters, lr=lr)
            #
            model_eval = ModelClassifier(settings)
            model_evaluator = ModelEvaluator(
                settings, model_eval, eval_dataset, scores_calculator)
            #
            model_trainer.train_with_adaptive_decay(optimizer_creator, model_evaluator)
            #
        else:
            #
            optimizer = AdamW(optimizer_grouped_parameters, lr=settings.learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
            #
            model_trainer.train_with_scheduler(optimizer, scheduler)
            #
        #
    #
    if settings.do_eval:
        eval_examples = processor.get_dev_examples(settings.data_dir)
        num_actual_eval_examples = len(eval_examples)
        #
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", settings.eval_batch_size)
        #

        #
        eval_dataset = dataset_creator(tokenizer, eval_examples, label_list,
            max_seq_len=settings.max_seq_len, logger=settings.logger)
        #
        model = ModelClassifier(settings)
        model_evaluator = ModelEvaluator(settings, model, eval_dataset, scores_calculator)
        model_evaluator.initialize_and_load(
            model_saved_path=settings.model_saved_path,
            flag_print=True
        )
        #
        eval_results = model_evaluator.evaluate()
        main_score, scores_dict, list_batches_evaluated = eval_results
        #
        logger.info("scores: {}".format(scores_dict))
        logger.info("main_score: {}".format(main_score))
        #
        file_path = os.path.join(settings.output_dir, "eval_results.txt")
        with open(file_path, "w", encoding="utf-8") as fp:
            fp.write("main_score: {}\n".format(main_score))
            fp.write("scores: {}\n".format(scores_dict))
        #
    #
    if settings.do_predict:
        predict_examples = processor.get_test_examples(settings.data_dir)
        num_actual_predict_examples = len(predict_examples)
        #
        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", settings.predict_batch_size)
        #

        #
        eval_dataset = dataset_creator(tokenizer, predict_examples, label_list,
            max_seq_len=settings.max_seq_len, logger=settings.logger)
        #
        model = ModelClassifier(settings)
        model_evaluator = ModelEvaluator(settings, model, eval_dataset)
        model_evaluator.initialize_and_load(
            model_saved_path=settings.model_saved_path,
            flag_print=True
        )
        #
        eval_results = model_evaluator.evaluate()
        main_score, scores_dict, list_batches_evaluated = eval_results
        #
    #
#

#
if __name__ == "__main__":
    """
    """
    args = parsed_args()
    main(args)
    #
#
