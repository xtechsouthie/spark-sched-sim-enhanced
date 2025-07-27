import os.path as osp
import argparse
import sys
from cfg_loader import load
from trainers import make_trainer

def main():
    parser = argparse.ArgumentParser(description='Train PPO with ciritc network')
    parser.add_argument('--config', type=str, default='test/test.yaml', help='path to config file')
    parser.add_argument('--no-critic', action='store_true', help='disable critic net')
    parser.add_argument('--eval-freq', type=int, default=10, help='frequency of evals')
    parser.add_argument('--artifacts-dir', type=str, help='override artifacts dir from config')
    args = parser.parse_args()

    if not osp.exists(args.config):
        print(f'Error: Config file {args.config} not found.')
        sys.exit(1)

    cfg = load(args.config)

    if args.no_critic:
        cfg['trainer']['use_critic'] = False
        print('training with critic network disabled')
    else:
        cfg['trainer']['use_critic'] = True
        print('training with critic network enabled')
    
    if args.artifacts_dir:
        cfg['trainer']['artifacts_dir'] = args.artifacts_dir
        print(f'artifacts will be saved to {args.artifacts_dir}')

    trainer = make_trainer(cfg)

    if hasattr(trainer, 'eval_freq'):
        trainer.eval_freq = args.eval_freq

    print('training started')

    try:
        print('Calling trainer.train()')
        try:
            trainer.train()
            print('trainer.train() completed')
        except Exception as e:
            print(f"Error during training: {e}")
            raise

        print('training completed sucessfully')

        print('Running final evaluation with both baselines')
        if hasattr(trainer, 'evaluate_both_baselines'):
            try:
                results = trainer.evaluate_both_baselines(cfg['env'])
                print('Evaluation completed')

                if 'comparison' in results:
                    comparison = results['comparison']
                    print('\nFinal Results:')
                    print(f"Original baseline avg job duration: {results['original_baseline']['avg_job_duration']:.3f}s")
                    print(f"Critic baseline avg job duration: {results['critic_baseline']['avg_job_duration']:.3f}s")
                    print(f"Improvement: {comparison['percentage_improvement']:.2f}%")
                    print(f"Better method: {comparison['better_method'].upper()}")
                else:
                    print('\nComparison results not available')
            except Exception as e:
                print(f"Error during evaluation: {e}")
                raise
    
    except KeyboardInterrupt:
        print('\ntraining inturrupted by the user')
    except Exception as e:
        print(f'\n Error during training: {e}')
        raise

if __name__ == '__main__':
    main()


