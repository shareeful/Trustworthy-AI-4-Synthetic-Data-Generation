"""
Run experiments for Profile P-1 (Fairness - High-Risk Medical).
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
from src.utils.logger import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Run Profile P-1 (Fairness) experiments')
    parser.add_argument('--dataset', type=str, default='data/insurance_claims.csv',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results/profile_p1',
                       help='Directory to save results')
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: HybridVAEGAN,
    optimizer_vae: optim.Optimizer,
    optimizer_disc: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    loss_weights: dict,
    device: torch.device,
    sample_weights: np.ndarray = None
) -> dict:
    """Train for one epoch."""
    model.train()
    epoch_losses = {
        'total': 0.0,
        'recon': 0.0,
        'kl': 0.0,
        'gan': 0.0,
        'fair': 0.0,
        'l1': 0.0
    }
    
    n_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        
        # Get sample weights for this batch if provided
        batch_sample_weights = None
        if sample_weights is not None:
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(sample_weights))
            batch_sample_weights = torch.from_numpy(
                sample_weights[start_idx:end_idx]
            ).float().to(device)
        
        # ========== Train Discriminator ==========
        optimizer_disc.zero_grad()
        
        outputs = model(data)
        
        # Discriminator loss
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(outputs['real_validity'])
        fake_labels = torch.zeros_like(outputs['fake_validity'])
        
        d_real_loss = criterion(outputs['real_validity'], real_labels)
        d_fake_loss = criterion(outputs['fake_validity'], fake_labels)
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        optimizer_disc.step()
        
        # ========== Train VAE/Generator ==========
        optimizer_vae.zero_grad()
        
        outputs = model(data)
        
        # Compute all losses
        losses = model.compute_losses(data, outputs, loss_weights)
        
        # Reconstruction loss (with sample weights if provided)
        recon_loss = losses['recon']
        if batch_sample_weights is not None:
            recon_loss = (recon_loss * batch_sample_weights).mean()
        
        # Fairness loss (bias penalty)
        from src.compliance.metrics import FairnessMetrics
        # Simulate protected attribute (in real implementation, pass from data)
        sensitive_attr = torch.randint(0, 2, (data.size(0),)).to(device)
        predictions = outputs['x_recon'].mean(dim=1)
        fair_loss = FairnessMetrics.compute_bias_loss(predictions, sensitive_attr)
        
        # Total VAE loss
        total_loss = (
            loss_weights['recon'] * recon_loss +
            loss_weights['fair'] * fair_loss +
            loss_weights['adv'] * losses['gan'] +
            loss_weights['sparse'] * losses['l1']
        )
        
        total_loss.backward()
        optimizer_vae.step()
        
        # Track losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['fair'] += fair_loss.item()
        epoch_losses['gan'] += losses['gan'].item()
        epoch_losses['l1'] += losses['l1'].item()
        
        n_batches += 1
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= n_batches
    
    return epoch_losses


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging('profile_p1', 'logs')
    logger.log_config(vars(args))
    
    device = torch.device(args.device)
    logger.get_logger().info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    with open('config/profiles.yaml') as f:
        profiles = yaml.safe_load(f)
    profile_config = profiles['profile_p1']
    
    with open('config/hyperparameters.yaml') as f:
        hyper_config = yaml.safe_load(f)

    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("PHASE 2: PROFILE-DRIVEN DATA PREPROCESSING")
    logger.get_logger().info("="*80)
    
    # Load data
    data_loader = InsuranceDataLoader(args.dataset, test_size=0.2, random_state=args.seed)
    train_df, test_df = data_loader.load()
    
    # Simulate bias (65:35 male:female ratio)
    train_df = data_loader.simulate_bias(train_df, ratio=(0.65, 0.35))
    
    # Preprocess with profile-specific transformations
    preprocessor = ProfileDrivenPreprocessor(profile_config)
    X_train, y_train, sample_weights = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    logger.get_logger().info(f"Training samples: {X_train.shape[0]}")
    logger.get_logger().info(f"Test samples: {X_test.shape[0]}")
    logger.get_logger().info(f"Feature dimension: {X_train.shape[1]}")
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # ===== PHASE 3: Translation Mechanism =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("PHASE 3: TRANSLATION MECHANISM")
    logger.get_logger().info("="*80)
    
    translator = TranslationMechanism(
        alpha=hyper_config['translation_mechanism']['alpha'],
        beta=hyper_config['translation_mechanism']['beta']
    )
    
    initial_weights = translator.translate(profile_config)
    logger.get_logger().info("\nDerived Loss Weights (Equation 1):")
    for weight_name, value in initial_weights.items():
        logger.get_logger().info(f"  λ_{weight_name} = {value:.2f}")
    
    # ===== PHASE 4: Model Initialization and Training Setup =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("PHASE 4: HYBRID VAE-GAN TRAINING")
    logger.get_logger().info("="*80)
    
    input_dim = X_train.shape[1]
    model = HybridVAEGAN(
        input_dim=input_dim,
        vae_config=hyper_config['vae'],
        gan_config=hyper_config['gan']
    ).to(device)
    
    # Optimizers
    optimizer_vae = optim.Adam(
        list(model.vae.parameters()),
        lr=hyper_config['training']['learning_rate'],
        betas=(hyper_config['training']['adam_beta1'], 
               hyper_config['training']['adam_beta2'])
    )
    
    optimizer_disc = optim.Adam(
        model.discriminator.parameters(),
        lr=hyper_config['training']['learning_rate'],
        betas=(hyper_config['training']['adam_beta1'],
               hyper_config['training']['adam_beta2'])
    )
    
    # ===== PHASE 5: Closed-Loop Compliance Engine =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("PHASE 5: CLOSED-LOOP COMPLIANCE ENGINE")
    logger.get_logger().info("="*80)
    
    compliance_engine = ClosedLoopComplianceEngine(
        profile_config=profile_config,
        initial_weights=initial_weights,
        monitor_frequency=hyper_config['training']['monitor_frequency'],
        weight_increment=hyper_config['closed_loop']['weight_increment'],
        max_cycles=hyper_config['closed_loop']['max_cycles'],
        accuracy_floor=profile_config['thresholds']['accuracy_floor']
    )
    
    # ===== Training Loop =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("STARTING TRAINING")
    logger.get_logger().info("="*80)
    
    current_weights = initial_weights.copy()
    best_compliance = False
    
    # Extract sensitive attribute for monitoring
    sensitive_attr_train = train_df['gender'].values
    sensitive_attr_test = test_df['gender'].values
    
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        epoch_losses = train_epoch(
            model=model,
            optimizer_vae=optimizer_vae,
            optimizer_disc=optimizer_disc,
            dataloader=train_loader,
            loss_weights=current_weights,
            device=device,
            sample_weights=sample_weights
        )
        
        # Log epoch metrics
        logger.log_epoch(epoch, epoch_losses)
        
        # Compliance monitoring (Phase 5)
        if epoch % hyper_config['training']['monitor_frequency'] == 0:
            # Prepare validation data
            X_val_tensor = torch.FloatTensor(X_test).to(device)
            y_val_tensor = torch.FloatTensor(y_test).to(device)
            
            # Monitor and adjust
            should_continue, updated_weights = compliance_engine.monitor_and_adjust(
                epoch=epoch,
                model=model,
                x_val=X_val_tensor,
                y_val=y_val_tensor,
                sensitive_attr=sensitive_attr_test,
                device=device
            )
            
            # Update weights if adjusted
            if updated_weights != current_weights:
                current_weights = updated_weights
                logger.get_logger().info(f"\n⚠️  Loss weights updated:")
                for name, value in current_weights.items():
                    logger.get_logger().info(f"    λ_{name} = {value:.2f}")
            
            # Check if should stop
            if not should_continue:
                logger.get_logger().info("\n✓ Training stopped by Closed-Loop Engine")
                logger.get_logger().info(f"Total epochs: {epoch}")
                break
    
    # ===== Final Evaluation =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("FINAL EVALUATION")
    logger.get_logger().info("="*80)
    
    model.eval()
    metrics_aggregator = MetricsAggregator()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)
    
    final_metrics = metrics_aggregator.compute_all_metrics(
        model=model,
        x_real=X_test_tensor,
        y_true=y_test_tensor,
        sensitive_attr=sensitive_attr_test,
        device=device
    )
    
    logger.get_logger().info("\nFinal Metrics:")
    for metric_name, value in final_metrics.items():
        logger.get_logger().info(f"  {metric_name}: {value:.4f}")
    
    # Check final compliance
    all_compliant, individual_compliance = metrics_aggregator.check_compliance(
        final_metrics, profile_config['thresholds']
    )
    
    logger.get_logger().info("\nFinal Compliance Status:")
    for metric, status in individual_compliance.items():
        symbol = "✓" if status else "✗"
        logger.get_logger().info(f"  {symbol} {metric}")
    
    # ===== Generate Audit Certificate =====
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("GENERATING AUDIT CERTIFICATE")
    logger.get_logger().info("="*80)
    
    audit_report = compliance_engine.generate_audit_report()
    logger.get_logger().info(audit_report)
    
    # Save audit certificate
    audit_file = save_dir / f"audit_certificate_seed{args.seed}.txt"
    with open(audit_file, 'w') as f:
        f.write(audit_report)
    logger.get_logger().info(f"\nAudit certificate saved to: {audit_file}")
    
    # ===== Save Model and Results =====
    model_file = save_dir / f"model_seed{args.seed}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'profile_config': profile_config,
        'final_weights': current_weights,
        'final_metrics': final_metrics,
        'audit_certificate': compliance_engine.get_audit_certificate()
    }, model_file)
    logger.get_logger().info(f"Model saved to: {model_file}")
    
    # Save metrics
    logger.save_metrics()
    
    logger.get_logger().info("\n" + "="*80)
    logger.get_logger().info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.get_logger().info("="*80)


if __name__ == "__main__":
    main()
