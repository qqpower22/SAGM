import argparse
import json
import os


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
from functools import wraps
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from contrastive_learning import StructureAwarePathContrastiveLearning
import config
import data_loader as data_loader
import utils
from model import SAGM
import random
import time
from tqdm import tqdm
from convert_results import remove_special_tokens


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, value in named_tensors:
        value = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str((tuple(value.shape), value.dtype)).encode('utf-8'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def repeatable_evaluation(method):

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng():
                random.seed(config.seed)
                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)
                return method(self, *args, **kwargs)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
    return wrapped


class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        if config.use_contrastive:
            strategy = getattr(config, 'negative_strategy', 'all')
            allowed = {'replace', 'extend', 'delete', 'shuffle'}
            enabled = allowed if strategy == 'all' else {
                item.strip() for item in strategy.split(',') if item.strip()
            }
            if not enabled or not enabled.issubset(allowed):
                raise ValueError('Invalid negative strategy: {}'.format(strategy))
            self.contrastive_module = StructureAwarePathContrastiveLearning(
                hidden_dim=config.lstm_hid_size,
                temperature=config.contrastive_temperature,
                use_replace='replace' in enabled,
                use_extend='extend' in enabled,
                use_delete='delete' in enabled,
                use_shuffle='shuffle' in enabled
            ).cuda()
        else:
            self.contrastive_module = None
        bert_param_ids = {id(p) for p in self.model.bert.parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in bert_param_ids]
        if config.use_contrastive:
            contrastive_params = list(self.contrastive_module.parameters())
        else:
            contrastive_params = []
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {
                'params': [
                    p for n, p in model.bert.named_parameters() if not any((nd in n for nd in no_decay))
                ],
                'lr': config.bert_learning_rate,
                'weight_decay': config.weight_decay
            },
            {
                'params': [
                    p for n, p in model.bert.named_parameters() if any((nd in n for nd in no_decay))
                ],
                'lr': config.bert_learning_rate,
                'weight_decay': 0.0
            },
            {
                'params': other_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay
            }
        ]
        if config.use_contrastive:
            params.append({
                'params': contrastive_params,
                'lr': config.learning_rate * 0.1,
                'weight_decay': config.weight_decay
            })
        self.optimizer = transformers.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warm_factor * updates_total,
            num_training_steps=updates_total
        )

    @staticmethod
    def filter_by_label(type_id, instance_list):
        filtered = []
        for inst in instance_list:
            filtered_inst = []
            for ent in inst:
                if ent[-1] == type_id:
                    filtered_inst.append(ent)
            filtered.append(filtered_inst)
        return filtered

    def train(self, epoch, data_loader):
        self.model.train()
        if config.use_contrastive:
            self.contrastive_module.train()
        loss_list = []
        pred_result = []
        label_result = []
        ce_loss_list = []
        contrastive_loss_list = []
        vib_loss_list = []
        vib_kl_loss_list = []
        for i, data_batch in enumerate(tqdm(data_loader)):
            entity_text_batch = data_batch[-1].copy()
            data_batch = [data.cuda() for data in data_batch[:-1]]
            (
                bert_inputs,
                grid_labels,
                grid_mask2d,
                pieces2word,
                dist_inputs,
                sent_length,
            ) = data_batch
            if args.diagnostic_steps:
                logger.info(
                    'DIAG step=%d input=%s',
                    i,
                    tensor_digest(((str(j), value) for j, value in enumerate(data_batch)))
                )
            outputs, grid_outputs, vib_outputs = model(
                bert_inputs,
                grid_mask2d,
                dist_inputs,
                pieces2word,
                sent_length
            )
            grid_mask2d = grid_mask2d.clone()
            ce_loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])
            vib_loss = 0
            vib_kl_loss = 0
            if config.use_vib and vib_outputs is not None:
                vib_loss = vib_outputs['vib_loss']
                vib_kl_loss = vib_outputs['vib_kl_loss']
                vib_loss_list.append(vib_loss.cpu().item())
                vib_kl_loss_list.append(vib_kl_loss.cpu().item())
            contrastive_loss = 0
            if (
                config.use_contrastive
                and self.contrastive_module is not None
                and (len(entity_text_batch) > 0)
            ):
                batch_entities = []
                for k, entity_text in enumerate(entity_text_batch):
                    entities = []
                    for ent_str in sorted(entity_text):
                        try:
                            index_str, type_str = ent_str.split('-#-')
                            indices = [int(x) for x in index_str.split('-')]
                            entities.append({'index': indices, 'type': type_str})
                        except:
                            continue
                    batch_entities.append(entities)
                contrastive_loss = self.contrastive_module(
                    grid_outputs,
                    batch_entities,
                    sent_length.cpu().numpy().tolist()
                )
                contrastive_loss_list.append(contrastive_loss.cpu().item())
            else:
                contrastive_loss = torch.tensor(0.0, device=outputs.device)
            if config.use_contrastive:
                contrastive_weight = getattr(config, 'contrastive_weight', 0.1)
                current_weight = contrastive_weight * min(1.0, epoch / 10.0)
                weighted_contrastive_loss = contrastive_loss * current_weight
            else:
                weighted_contrastive_loss = 0
            total_loss = ce_loss + weighted_contrastive_loss + vib_loss
            loss_list.append(total_loss.cpu().item())
            ce_loss_list.append(ce_loss.cpu().item())
            if args.diagnostic_steps:
                logger.info(
                    'DIAG step=%d output=%s CE=%.12g CL=%.12g VIB=%.12g',
                    i,
                    tensor_digest([('output', outputs)]),
                    ce_loss.item(),
                    float(contrastive_loss),
                    float(vib_loss)
                )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            if config.use_contrastive and self.contrastive_module is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.contrastive_module.parameters(),
                    config.clip_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            outputs_argmax = torch.argmax(outputs, -1)
            grid_labels_flat = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs_flat = outputs_argmax[grid_mask2d].contiguous().view(-1)
            label_result.append(grid_labels_flat.cpu())
            pred_result.append(outputs_flat.cpu())
            self.scheduler.step()
            if args.diagnostic_steps:
                logger.info(
                    'DIAG step=%d updated_model=%s',
                    i,
                    tensor_digest(self.model.state_dict().items())
                )
                if self.contrastive_module is not None:
                    logger.info(
                        'DIAG step=%d updated_contrastive=%s',
                        i,
                        tensor_digest(self.contrastive_module.state_dict().items())
                    )
                if i + 1 >= args.diagnostic_steps:
                    logger.info('DIAG done: no checkpoint or result JSON saved.')
                    raise SystemExit(0)
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average='macro'
        )
        table = pt.PrettyTable([
            'Train {}'.format(epoch),
            'Loss',
            'F1',
            'Precision',
            'Recall',
            'CE Loss',
            'Contrastive Loss',
            'VIB Loss',
            'KL Loss'
        ])
        table.add_row([
            'Label',
            '{:.4f}'.format(np.mean(loss_list))
        ] + ['{:3.4f}'.format(x) for x in [f1, p, r]] + [
            '{:.4f}'.format(np.mean(ce_loss_list)),
            '{:.4f}'.format(np.mean(contrastive_loss_list) if contrastive_loss_list else 0),
            '{:.4f}'.format(np.mean(vib_loss_list) if vib_loss_list else 0),
            '{:.4f}'.format(np.mean(vib_kl_loss_list) if vib_kl_loss_list else 0)
        ])
        logger.info('\n{}'.format(table))
        return f1

    @repeatable_evaluation
    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()
        pred_result = []
        label_result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch
                outputs, _, _ = model(
                    bert_inputs,
                    grid_mask2d,
                    dist_inputs,
                    pieces2word,
                    sent_length
                )
                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, _ = utils.decode(
                    outputs.cpu().numpy(),
                    entity_text,
                    length.cpu().numpy()
                )
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average='macro'
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
        title = 'EVAL' if not is_test else 'TEST'
        logger.info('{} Label F1 {}'.format(
            title,
            f1_score(label_result.numpy(), pred_result.numpy(), average=None)
        ))
        table = pt.PrettyTable(['{} {}'.format(title, epoch), 'F1', 'Precision', 'Recall'])
        table.add_row(['Label'] + ['{:3.4f}'.format(x) for x in [f1, p, r]])
        table.add_row(['Entity'] + ['{:3.4f}'.format(x) for x in [e_f1, e_p, e_r]])
        logger.info('\n{}'.format(table))
        return (e_f1, e_p, e_r)

    @repeatable_evaluation
    def predict(self, epoch, data_loader, data):
        self.model.eval()
        pred_result = []
        label_result = []
        result = []
        pred_indexes = []
        gold_indexes = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                (
                    bert_inputs,
                    grid_labels,
                    grid_mask2d,
                    pieces2word,
                    dist_inputs,
                    sent_length,
                ) = data_batch
                outputs, _, _ = model(
                    bert_inputs,
                    grid_mask2d,
                    dist_inputs,
                    pieces2word,
                    sent_length
                )
                length = sent_length
                grid_mask2d = grid_mask2d.clone()
                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(
                    outputs.cpu().numpy(),
                    entity_text,
                    length.cpu().numpy()
                )
                entity_index = [
                    [
                        [int(x) for x in ent.split('-') if x.isdigit()] for ent in inst
                    ] for inst in entity_text
                ]
                pred_index = [[x[0] + [x[1]] for x in inst] for inst in decode_entities]
                pred_indexes.extend(pred_index)
                gold_indexes.extend(entity_index)
                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence['sentence']
                    instance = {'sentence': sentence, 'entity': []}
                    for ent in ent_list:
                        instance['entity'].append({
                            'text': [sentence[x] for x in ent[0]],
                            'index': ent[0],
                            'type': config.vocab.id_to_label(ent[1])
                        })
                    result.append(instance)
                total_ent_r += ent_r
                total_ent_p += ent_p
                total_ent_c += ent_c
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(
            label_result.numpy(),
            pred_result.numpy(),
            average='macro'
        )
        e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)
        all_results = utils.extract_all_disc_metrics(gold_indexes, pred_indexes)
        assert all_results['all']['f1'] == e_f1
        assert all_results['all']['p'] == e_p
        assert all_results['all']['r'] == e_r
        title = 'TEST'
        logger.info('{} Label F1 {}'.format(
            'TEST',
            f1_score(label_result.numpy(), pred_result.numpy(), average=None)
        ))
        table = pt.PrettyTable(['{} {}'.format(title, epoch), 'F1', 'Precision', 'Recall'])
        table.add_row(['Label'] + ['{:3.4f}'.format(x) for x in [f1, p, r]])
        table.add_row(['Entity'] + ['{:3.4f}'.format(x) for x in [e_f1, e_p, e_r]])
        table.add_row(['Disc_Sent'] + ['{:3.4f}'.format(x) for x in [
            all_results['sent_disc']['f1'],
            all_results['sent_disc']['p'],
            all_results['sent_disc']['r']
        ]])
        table.add_row(['Disc_Ent'] + ['{:3.4f}'.format(x) for x in [
            all_results['ent_disc']['f1'],
            all_results['ent_disc']['p'],
            all_results['ent_disc']['r']
        ]])
        logger.info('\n{}'.format(table))
        logger.info(all_results)
        overall_results = {'all_entities': all_results}
        types = config.vocab.get_labels()
        type_results = {}
        for t in types:
            logger.info(t)
            t_id = config.vocab.label_to_id(t)
            filtered_gold = self.filter_by_label(t_id, gold_indexes)
            filtered_pred = self.filter_by_label(t_id, pred_indexes)
            type_results[t] = utils.extract_all_disc_metrics(filtered_gold, filtered_pred)
            logger.info(type_results[t])
        overall_results['per_entity'] = type_results
        with open('output/' + config.predict_path, 'w', encoding='utf-8') as f:
            has_pos = [bool(item['sentence']) and item['sentence'][0] == '[POS]' for item in result]
            if any(has_pos) and (not all(has_pos)):
                raise ValueError('Mixed [POS] and plain sentences; check preprocessing.')
            adjusted_preds = remove_special_tokens(result) if all(has_pos) and result else result
            json.dump(adjusted_preds, f, ensure_ascii=False)
        return overall_results

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cadec.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)
    parser.add_argument('--dilation', type=str, help='e.g. 1,2,3')
    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)
    parser.add_argument('--use_bert_last_4_layers', type=int, help='1: true, 0: false')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model_name', type=str, default=None, choices=['SAGM'])
    parser.add_argument('--session_name', type=str, default='Run_(date)')
    parser.add_argument('--early_stop', type=int, default=None)
    parser.add_argument('--use_vib', default=None, action='store_true')
    parser.add_argument('--use_contrastive', default=None, action='store_true')
    parser.add_argument('--vib_beta', type=float, default=None)
    parser.add_argument('--use_entropy_fusion', type=int, choices=[0, 1], default=None)
    parser.add_argument('--negative_strategy', type=str, default=None)
    parser.add_argument('--dev_only', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument(
        '--diagnostic_steps',
        type=int,
        default=0,
        help='Run N training batches, log fingerprints, then exit without saving.'
    )
    args = parser.parse_args()
    if args.diagnostic_steps < 0:
        parser.error('--diagnostic_steps must be nonnegative')
    config = config.Config(args)
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    logger.info('#############Loading Data###################')
    datasets, ori_data = data_loader.load_data_bert(config)
    train_loader, dev_loader, test_loader = (DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=data_loader.collate_fn,
        shuffle=i == 0,
        num_workers=4,
        drop_last=i == 0
    ) for i, dataset in enumerate(datasets))
    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info('Building Model')
    if config.model_name == 'SAGM':
        model = SAGM(config)
    else:
        raise Exception('Invalid model name')
    model = model.cuda()
    trainer = Trainer(model)
    logger.info('REPRO protocol: strict kernels, ordered parameters, fixed sampled evaluation')
    logger.info(
        'Runtime: torch=%s CUDA=%s cuDNN=%s GPU=%s',
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name()
    )
    if args.diagnostic_steps:
        logger.info('DIAG init_model=%s', tensor_digest(model.state_dict().items()))
        if trainer.contrastive_module is not None:
            logger.info(
                'DIAG init_contrastive=%s',
                tensor_digest(trainer.contrastive_module.state_dict().items())
            )
        trainer.train(0, train_loader)
        raise SystemExit('Diagnostic requires at least one full training batch.')
    if config.dev_only and config.test_only:
        raise ValueError('--dev_only and --test_only cannot be used together.')
    if config.test_only:
        checkpoint_path = 'output/' + config.save_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint_path))
        dev_result_path = 'output/{}_dev_result.json'.format(config.session_name)
        dev_result = {}
        if os.path.exists(dev_result_path):
            with open(dev_result_path, 'r', encoding='utf-8') as file:
                dev_result = json.load(file)
        logger.info('TEST-ONLY mode: loading %s', checkpoint_path)
        trainer.load(checkpoint_path)
        all_results = trainer.predict('Final', test_loader, ori_data[-1])
        save_results = {
            'config': {k: v for k, v in config.__dict__.items() if k not in ['logger', 'vocab']},
            'checkpoint': checkpoint_path,
            'best_epoch': dev_result.get('best_epoch'),
            'best_dev_results': dev_result.get('best_dev_results', {})
        }
        save_results.update(all_results)
        result_path = 'output/{}.json'.format(config.session_name)
        with open(result_path, 'w', encoding='utf-8') as file:
            json.dump(save_results, file, indent=4, ensure_ascii=False)
        logger.info('Test result saved to: %s', result_path)
        raise SystemExit(0)
    if os.path.exists('output/' + config.save_path) or os.path.exists(
        'output/{}_dev_result.json'.format(config.session_name)
    ):
        raise FileExistsError('Existing run found. Use a new --session_name to preserve it.')
    best_f1 = 0
    best_epoch = 0
    best_dev_results = {}
    train_start = time.time()
    for i in range(config.epochs):
        logger.info('Epoch: {}'.format(i))
        trainer.train(i, train_loader)
        f1, p, r = trainer.eval(i, dev_loader)
        if f1 > best_f1 or best_f1 == 0:
            best_f1 = f1
            best_epoch = i
            best_dev_results = {'f1': f1, 'p': p, 'r': r}
            trainer.save('output/' + config.save_path)
        if config.early_stop is not None and i - best_epoch > int(config.early_stop):
            break
    train_time = time.time() - train_start
    logger.info('Training time (s): %.4f' % train_time)
    logger.info('Best DEV F1: {:3.4f}'.format(best_f1))
    logger.info('Best DEV Precision: {:3.4f}'.format(best_dev_results['p']))
    logger.info('Best DEV Recall: {:3.4f}'.format(best_dev_results['r']))
    logger.info('Best epoch: {}'.format(best_epoch))
    dev_result = {
        'dataset': config.dataset,
        'vib_beta': float(config.vib_beta),
        'use_entropy_fusion': bool(config.use_entropy_fusion),
        'negative_strategy': getattr(config, 'negative_strategy', 'all'),
        'best_epoch': int(best_epoch),
        'best_dev_results': {
            'f1': float(best_dev_results['f1']),
            'p': float(best_dev_results['p']),
            'r': float(best_dev_results['r'])
        },
        'train_time': float(train_time)
    }
    dev_result_path = 'output/{}_dev_result.json'.format(config.session_name)
    with open(dev_result_path, 'w', encoding='utf-8') as file:
        json.dump(dev_result, file, indent=4, ensure_ascii=False)
    logger.info('Validation result saved to: {}'.format(dev_result_path))
    if config.dev_only:
        logger.info('DEV-ONLY mode: Test set is NOT evaluated.')
    else:
        trainer.load('output/' + config.save_path)
        all_results = trainer.predict('Final', test_loader, ori_data[-1])
        with open('output/%s.json' % config.session_name, 'w') as file:
            save_results = {
                'config': {k: v for k, v in config.__dict__.items() if k not in [
                    'logger',
                    'vocab'
                ]},
                'train_time': train_time,
                'best_epoch': best_epoch,
                'best_dev_results': best_dev_results
            }
            for k, v in all_results.items():
                if k == 'all':
                    k = 'test_results'
                save_results[k] = v
            json.dump(save_results, file, indent=4)
