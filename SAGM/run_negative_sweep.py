import json
import os
import subprocess
import sys


DATASET = 'cadec'
CONFIG_FILE = './config/cadec.json'
DEVICE = 0
SEED = 123
MAIN_FILE = 'main_repro.py'
VIB_BETA = 0.06
STRATEGIES = [
    ('extend', 'delete', 'shuffle'),
    ('replace', 'delete', 'shuffle'),
    ('replace', 'extend', 'shuffle'),
    ('replace', 'extend', 'delete')
]
os.makedirs('output', exist_ok=True)


def strategy_name(strategy):
    abbreviations = {'replace': 'R', 'extend': 'E', 'delete': 'D', 'shuffle': 'S'}
    return ''.join((abbreviations[item] for item in strategy))


def session_name(strategy):
    return 'negative_repro_loo_{}_{}_beta{:.2f}_seed{}'.format(
        DATASET,
        strategy_name(strategy),
        VIB_BETA,
        SEED
    )


def result_path(strategy):
    return os.path.join('output', session_name(strategy) + '_dev_result.json')


def is_completed(strategy):
    return os.path.exists(result_path(strategy))


def run_strategy(strategy):
    strategy_value = ','.join(strategy)
    name = session_name(strategy)
    print('\n' + '=' * 70)
    print('Running negative strategy: {}'.format(strategy_value))
    print('=' * 70)
    command = [
        sys.executable,
        MAIN_FILE,
        '--config',
        CONFIG_FILE,
        '--device',
        str(DEVICE),
        '--vib_beta',
        '{:.2f}'.format(VIB_BETA),
        '--seed',
        str(SEED),
        '--use_contrastive',
        '--negative_strategy',
        strategy_value,
        '--session_name',
        name,
        '--dev_only'
    ]
    print('Command: {}'.format(' '.join(command)))
    return subprocess.run(command).returncode == 0


def collect_results():
    results = []
    for strategy in STRATEGIES:
        path = result_path(strategy)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            dev = data['best_dev_results']
            results.append({
                'strategy': strategy,
                'precision': dev['p'],
                'recall': dev['r'],
                'f1': dev['f1'],
                'best_epoch': data['best_epoch']
            })
        except (OSError, KeyError, TypeError, ValueError) as error:
            print('Cannot read {}: {}'.format(path, error))
    return results


def print_summary():
    results = collect_results()
    if not results:
        print('\nNo completed experiments.')
        return
    print('\n' + '=' * 88)
    print('NEGATIVE-SAMPLING VALIDATION RESULTS (beta={:.2f})'.format(VIB_BETA))
    print('=' * 88)
    print('{:<28} {:<10} {:<10} {:<10} {:<8}'.format('Strategy', 'P', 'R', 'F1', 'Epoch'))
    print('-' * 72)
    for item in results:
        print('{:<28} {:<10.4f} {:<10.4f} {:<10.4f} {:<8}'.format(
            '+'.join(item['strategy']),
            item['precision'],
            item['recall'],
            item['f1'],
            item['best_epoch']
        ))
    best = max(results, key=lambda item: item['f1'])
    print('\nBEST STRATEGY BASED ON VALIDATION F1')
    print('Strategy = {}'.format('+'.join(best['strategy'])))
    print('Validation Precision = {:.4f}'.format(best['precision']))
    print('Validation Recall = {:.4f}'.format(best['recall']))
    print('Validation F1 = {:.4f}'.format(best['f1']))
    print('Best epoch = {}'.format(best['best_epoch']))
if __name__ == '__main__':
    print('Negative-sampling validation search')
    print('Dataset: {}'.format(DATASET))
    print('Fixed beta: {:.2f}'.format(VIB_BETA))
    print('Seed: {}'.format(SEED))
    print('Number of strategies: {}'.format(len(STRATEGIES)))
    for strategy in STRATEGIES:
        if is_completed(strategy):
            print('[SKIP] {} already completed.'.format('+'.join(strategy)))
            continue
        if not run_strategy(strategy):
            print('\n[STOP] {} failed or was interrupted.'.format('+'.join(strategy)))
            print('Run this script again to resume the search.')
            break
    print_summary()
