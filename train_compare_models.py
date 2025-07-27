
import yaml
from cfg_loader import load

def main():
    # Load configurations
    cfg = load('test/test.yaml')
    
    print("Configuration loaded:")
    print(f"Training iterations: {cfg['trainer']['num_iterations']}")
    print(f"Use critic: {cfg['trainer'].get('use_critic', False)}")
    print(f"Compare baselines: {cfg['trainer'].get('compare_baselines', False)}")
    
    # Import trainer after loading configs
    from trainers import make_trainer
    
    # Create a trainer instance to access the comparison method
    trainer = make_trainer(cfg)
    
    # Train and compare both models
    results = trainer.train_and_compare_models(cfg['trainer'], cfg['agent'], cfg['env'])
    
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS")
    print("="*50)
    print(f"Model without critic: {results['comparison']['without_critic_avg_duration']:.3f}s")
    print(f"Model with critic: {results['comparison']['with_critic_avg_duration']:.3f}s")
    print(f"Improvement: {results['comparison']['percentage_improvement']:.2f}%")
    print(f"Winner: {results['comparison']['better_method']}")
    
    # Save results
    import json
    import os
    
    results_dir = cfg['trainer'].get('artifacts_dir', 'test/artifacts')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/comparison_results.json")

if __name__ == "__main__":
    main()
