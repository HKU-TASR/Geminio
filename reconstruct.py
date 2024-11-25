from core.models import GeminioResNet34
from core.dataset import CustomData
import breaching
import logging
import torch
import sys
import os

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

# List of supported semantic queries
SUPPORTED_QUERIES = [
    "Any jewelry?",
    "Any human faces?", 
    "Any males with a beard?",
    "Any guns?",
    "Any females riding a horse?"
]

def reconstruct_image(cfg, setup, query=None):
    """
    Reconstruct private training images using either baseline or query-based approach.
    
    Args:
        cfg: Configuration object containing model and training parameters
        setup: Dictionary containing device and dtype settings
        query: Optional semantic query string for targeted reconstruction
        
    Returns:
        None (Saves reconstructed images to disk)
    """
    # Initialize model and components
    model = GeminioResNet34(num_classes=cfg.case.data.classes)
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, model, setup)
    
    # Load query-specific model if query is provided
    if query:
        if query not in SUPPORTED_QUERIES:
            raise ValueError(
                f"Query '{query}' is not supported. Supported queries are: {SUPPORTED_QUERIES}.\n"
                "We will release the training script after acceptance of our paper."
            )
        model_path = f'./malicious_models/{query.replace(" ", "_")}.pt'
        model_state = torch.load(model_path)
        if not any(k.startswith('clf.') for k in model_state.keys()):
            model_state = {'clf.%s' % key: value for key, value in model_state.items()}
        model.model.load_state_dict(model_state, strict=False)

    # Setup attack components
    attacker_loss = torch.nn.CrossEntropyLoss()
    attacker = breaching.attacks.prepare_attack(server.model, attacker_loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    # Get server payload
    server_payload = server.distribute_payload()

    # Create save directory if it doesn't exist
    if not os.path.exists(cfg.attack.save_dir):
        os.mkdir(cfg.attack.save_dir)

    # Load and process data
    cus_data = CustomData(
        data_dir='./assets/private_samples/', 
        dataset_name='ImageNet',
        number_data_points=cfg.case.user.num_data_points
    )
    
    # Compute updates and save ground truth
    shared_data, true_user_data = user.compute_local_updates(
        server_payload, 
        custom_data=cus_data.process_data()
    )
    true_pat = cfg.attack.save_dir + 'a_truth.jpg'
    cus_data.save_recover(true_user_data, save_pth=true_pat)

    # Perform reconstruction and save results
    reconstructed_user_data, stats = attacker.reconstruct(
        [server_payload], 
        [shared_data], 
        {}, 
        dryrun=cfg.dryrun,
        custom=cus_data
    )
    recon_path__ = cfg.attack.save_dir + 'final_rec.jpg'
    cus_data.save_recover(reconstructed_user_data, true_user_data, recon_path__)

if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Image reconstruction using Geminio')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline', action='store_true', help='Run baseline reconstruction')
    group.add_argument('--geminio-query', type=str, help='Query for Geminio reconstruction')
    args = parser.parse_args()

    # Initialize configuration and setup
    cfg = breaching.get_config(overrides=["case=geminio_demo", "attack=hfgradinv"])
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

    # Run reconstruction
    reconstruct_image(cfg, setup, args.geminio_query if args.geminio_query else None)