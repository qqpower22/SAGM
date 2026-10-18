import os
import sys
import json
import subprocess


BETAS = [0.1]
DATASET = 'share13'
CONFIG_FILE = './config/share13.json'
DEVICE = 0
SEED = 123
os.makedirs('output', exist_ok=True)


def is_completed(beta):
    beta_str = '{:.2f}'.format(beta)
    session_name = 'beta_repro_{}_{}_seed{}'.format(DATASET, beta_str, SEED)
    result_file = os.path.join('output', session_name + '_dev_result.json')
    return os.path.exists(result_file)


def run_beta(beta):
    beta_str = '{:.2f}'.format(beta)
    session_name = 'beta_repro_{}_{}_seed{}'.format(DATASET, beta_str, SEED)
    print('\n')
    print('=' * 70)
    print('Running beta = {}'.format(beta_str))
    print('=' * 70)
    command = [
        sys.executable,
        'main_repro.py',
        '--config',
        CONFIG_FILE,
        '--device',
        str(DEVICE),
        '--vib_beta',
        beta_str,
        '--seed',
        str(SEED),
        '--session_name',
        session_name,
        '--dev_only'
    ]
    print('Command:')
    print(' '.join(command))
    result = subprocess.run(command)
    return result.returncode == 0


def collect_results():
    results = []
    for beta in BETAS:
        beta_str = '{:.2f}'.format(beta)
        session_name = 'beta_repro_{}_{}_seed{}'.format(DATASET, beta_str, SEED)
        result_file = os.path.join('output', session_name + '_dev_result.json')
        if not os.path.exists(result_file):
            continue
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dev = data['best_dev_results']
            results.append({
                'beta': beta,
                'precision': dev['p'],
                'recall': dev['r'],
                'f1': dev['f1'],
                'best_epoch': data['best_epoch']
            })
        except Exception as e:
            print('Cannot read {}: {}'.format(result_file, e))
    return results


def print_summary():
    results = collect_results()
    if len(results) == 0:
        print('\nNo completed experiments.')
        return
    print('\n')
    print('=' * 70)
    print('VALIDATION RESULTS')
    print('=' * 70)
    print('{:<8} {:<12} {:<12} {:<12} {:<8}'.format('Beta', 'P', 'R', 'F1', 'Epoch'))
    print('-' * 60)
    for item in results:
        print('{:<8.2f} {:<12.4f} {:<12.4f} {:<12.4f} {:<8}'.format(
            item['beta'],
            item['precision'],
            item['recall'],
            item['f1'],
            item['best_epoch']
        ))
    best = max(results, key=lambda x: x['f1'])
    print('\n')
    print('=' * 70)
    print('BEST BETA BASED ON VALIDATION F1')
    print('=' * 70)
    print('Beta = {:.2f}'.format(best['beta']))
    print('Validation Precision = {:.4f}'.format(best['precision']))
    print('Validation Recall = {:.4f}'.format(best['recall']))
    print('Validation F1 = {:.4f}'.format(best['f1']))
if __name__ == '__main__':
    print('\n')
    print('=' * 70)
    print('Beta Validation Search')
    print('Dataset:', DATASET)
    print('=' * 70)
    for beta in BETAS:
        beta_str = '{:.2f}'.format(beta)
        if is_completed(beta):
            print('[SKIP] beta={} already completed.'.format(beta_str))
            continue
        success = run_beta(beta)
        if not success:
            print('\n')
            print('[STOP] beta={} was interrupted or failed.'.format(beta_str))
            print('Run this script again after reconnecting.')
            print('Completed beta values will be skipped automatically.')
            break
    print_summary()
