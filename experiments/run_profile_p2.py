"""
Run experiments for Profile P-2 (Security - Safety-Critical).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import argparse
from tqdm import tqdm

from src.data.loader import InsuranceDataLoader
from src.data.preprocessor import ProfileDrivenPreprocessor
from src.models.hybrid_vaegan import HybridVAEGAN
from src.translation.mechanism import TranslationMechanism
from src.compliance.closed_loop import ClosedLoopComplianceEngine
from src.compliance.metrics import MetricsAggregator
from src.attacks.autozoom import AutoZOOMAttack
from src.utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Run Profile P-2 (Security) experiments')
    parser.add_argument('--dataset', type=str, default='data/insurance_claims.csv')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results/profile_p2')
    parser.add_argument('--adversarial_attack', action='store_true',
                       help='Include adversarial attack evaluation')
    return parser.parse_args()


def train_epoch_with_adversarial(
    model: HybridVAEGAN,
    optimizer_vae: optim.Optimizer,
    optimizer_disc: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    loss_weights: dict,
    device: torch.device,
    adversarial_ratio: float = 0.1,
    epsilon: float = 0.1
) -> dict:
    """Train for one epoch with adversarial training."""
    model.train()
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'gan': 0.0,
        'adv_robust': 0.0
    }
    
    # Initialize AutoZOOM attack
    attack = AutoZOOMAttack(
        model=model,
        epsilon=epsilon,
        max_iterations=50,
        learning_rate=0.01
    )
    
    n_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        
        # Generate adversarial samples for a fraction of the batch
        if np.random.rand() < adversarial_ratio:
            data_adv, _ = attack.attack(data, target, device)
            data_mixed = data_adv
        else:
            data_mixed = data
        
        # Train discriminator
        optimizer_disc.zero_grad()
        outputs = model(data_mixed)
        
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(outputs['real_validity'])
        fake_labels = torch.zeros_like(outputs['fake_validity'])
        
        d_real_loss = criterion(outputs['real_validity'], real_labels)
        d_fake_loss = criterion(outputs['fake_validity'], fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        optimizer_disc.step()
        
        # Train VAE/Generator
        optimizer_vae.zero_grad()
        outputs = model(data_mixed)
        losses = model.compute_losses(data_mixed, outputs, loss_weights)
        
        # Total loss with emphasis on adversarial robustness
        total_loss = (
            loss_weights['recon'] * losses['recon'] +
            loss_weights['adv'] * losses['gan'] +
            loss_weights['sparse'] * losses['l1']
        )
        
        total_loss.backward()
        optimizer_vae.step()
        
        epoch_losses['total'] += total_loss.item()
        epoch_losses['recon'] += losses['recon'].item()
        epoch_losses['gan'] += losses['gan'].item()
        epoch_losses['adv_robust'] += losses['gan'].item()  # Use GAN loss as robustness proxy
        
        n_batches += 1
    
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    return epoch_losses


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger = setup_logging('profile_p2', 'logs')
    logger.log_config(vars(args))
    
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    with open('config/profiles.yaml') as f:
        profiles = yaml.safe_load(f)
    profile_config = profiles['profile_p2']
    
    with open('config/hyperparameters.yaml') as f:
        hyper_config = yaml.safe_load(f)
    
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("PROFILE P-2: SAFETY-CRITICAL (SECURITY FOCUS)")
    logger.get_logger().info("="*80)
    logger.get_logger().info(f"EU AI Act Article: {profile_config['eu_article']}")
    logger.get_logger().info(f"Target: Adversarial Robustness")
    
    # Load and preprocess data
    data_loader = InsuranceDataLoader(args.dataset, test_size=0.2, random_state=args.seed)
    train_df, test_df = data_loader.load()
    
    preprocessor = ProfileDrivenPreprocessor(profile_config)
    X_train, y_train, _ = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    # Translation Mechanism
    translator = TranslationMechanism(
        alpha=hyper_config['translation_mechanism']['alpha'],
        beta=hyper_config['translation_mechanism']['beta']
    )
    initial_weights = translator.translate(profile_config)
    
    logger.get_logger().info("\nDerived Loss Weights:")
    for weight_name, value in initial_weights.items():
        logger.get_logger().info(f"  λ_{weight_name} = {value:.2f}")
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = HybridVAEGAN(
        input_dim=input_dim,
        vae_config=hyper_config['vae'],
        gan_config=hyper_config['gan']
    ).to(device)
    
    optimizer_vae = optim.Adam(
        list(model.vae.parameters()),
        lr=hyper_config['training']['learning_rate'],
        betas=(hyper_config['training']['adam_beta1'], hyper_config['training']['adam_beta2'])
    )
    
    optimizer_disc = optim.Adam(
        model.discriminator.parameters(),
        lr=hyper_config['training']['learning_rate'],
        betas=(hyper_config['training']['adam_beta1'], hyper_config['training']['adam_beta2'])
    )
    
    # Closed-Loop Engine
    compliance_engine = ClosedLoopComplianceEngine(
        profile_config=profile_config,
        initial_weights=initial_weights,
        monitor_frequency=hyper_config['training']['monitor_frequency'],
        weight_increment=hyper_config['closed_loop']['weight_increment'],
        max_cycles=hyper_config['closed_loop']['max_cycles'],
        accuracy_floor=profile_config['thresholds']['accuracy_floor']
    )
    
    # Training loop with adversarial training
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("TRAINING WITH ADVERSARIAL SAMPLES")
    logger.get_logger().info("="*80)
    
    current_weights = initial_weights.copy()
    adversarial_ratio = profile_config['preprocessing'].get('adversarial_ratio', 0.1)
    epsilon = profile_config['preprocessing'].get('epsilon', 0.1)
    
    logger.get_logger().info(f"Adversarial ratio: {adversarial_ratio:.1%}")
    logger.get_logger().info(f"Perturbation budget (ε): {epsilon}")
    
    for epoch in range(1, args.epochs + 1):
        epoch_losses = train_epoch_with_adversarial(
            model=model,
            optimizer_vae=optimizer_vae,
            optimizer_disc=optimizer_disc,
            dataloader=train_loader,
            loss_weights=current_weights,
            device=device,
            adversarial_ratio=adversarial_ratio,
            epsilon=epsilon
        )
        
        logger.log_epoch(epoch, epoch_losses)
        
        # Compliance monitoring
        if epoch % hyper_config['training']['monitor_frequency'] == 0:
            X_val_tensor = torch.FloatTensor(X_test).to(device)
            y_val_tensor = torch.FloatTensor(y_test).to(device)
            
            # Generate adversarial test samples
            attack_eval = AutoZOOMAttack(model=model, epsilon=epsilon, max_iterations=100)
            X_adv_test, _ = attack_eval.attack(X_val_tensor, y_val_tensor, device)
            
            # Monitor with adversarial samples
            sensitive_attr = np.random.choice(['male', 'female'], len(X_test))
            should_continue, updated_weights = compliance_engine.monitor_and_adjust(
                epoch=epoch,
                model=model,
                x_val=X_val_tensor,
                y_val=y_val_tensor,
                sensitive_attr=sensitive_attr,
                device=device,
                x_adv=X_adv_test
            )
            
            if updated_weights != current_weights:
                current_weights = updated_weights
            
            if not should_continue:
                logger.get_logger().info(f"\n✓ Training completed at epoch {epoch}")
                break
    
    # Final evaluation with attack
    if args.adversarial_attack:
        logger.get_logger().info("\n" + "="*80)
        logger.get_logger().info("ADVERSARIAL ATTACK EVALUATION")
        logger.get_logger().info("="*80)
        
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        attack_final = AutoZOOMAttack(model=model, epsilon=epsilon, max_iterations=100)
        X_adv_final, attack_success = attack_final.attack(X_test_tensor, y_test_tensor, device)
        
        logger.get_logger().info(f"Attack Success Rate: {attack_success:.2%}")
        logger.get_logger().info(f"Attack FAILED Rate: {1-attack_success:.2%} (Model Robustness)")
    
    # Generate audit certificate
    audit_report = compliance_engine.generate_audit_report()
    audit_file = save_dir / f"audit_certificate_seed{args.seed}.txt"
    with open(audit_file, 'w') as f:
        f.write(audit_report)
    
    # Save model
    model_file = save_dir / f"model_seed{args.seed}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'profile_config': profile_config,
        'final_weights': current_weights
    }, model_file)
    
    logger.save_metrics()
    logger.get_logger().info("\n✓ EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
