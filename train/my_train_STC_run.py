import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import os, sys
import yaml # Import YAML
import argparse # Import argparse

# Assume these modules are correctly set up in your project structure
from model.AAE import AAE, AAE_Conv
from utils.dataset import Dataset
from utils.metrics import Evaluator
from model.UPerNet import UPerNet
from model.segformer import SegFormer
from model.u_net import UNet
from model.TransUNet import get_TransUNet as TransUNet
from model.deeplab import DeepLab
# from utils.losses import FocalLoss # Uncomment if you use FocalLoss

# --- Helper Functions ---

def setup_seed(seed):
    """Sets the random seed for reproducibility."""
    # np.random.seed(seed) # If using numpy random directly elsewhere
    # random.seed(seed) # If using python random directly elsewhere
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multiple GPUs
    # Comment out the following two lines if you need non-deterministic behavior for speed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device(config):
    """Gets the device based on config and availability."""
    if config['general']['device'] == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("Warning: CUDA requested but not available. Using CPU.")
            return torch.device('cpu')
    elif config['general']['device'] == 'cpu':
        return torch.device('cpu')
    else: # 'auto' or default
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def get_optimizer(model_params, config):
    """Creates an optimizer based on the config."""
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type'].lower()
    lr = opt_config['lr']

    if opt_type == 'adam':
        return optim.Adam(model_params, lr=lr, eps=opt_config.get('eps', 1e-8),
                          weight_decay=opt_config.get('weight_decay', 0))
    elif opt_type == 'sgd':
        return optim.SGD(model_params, lr=lr, momentum=opt_config.get('momentum', 0),
                         weight_decay=opt_config.get('weight_decay', 0),
                         nesterov=opt_config.get('nesterov', False))
    # Add other optimizers like AdamW etc. here
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_config['type']}")

def get_criterion(config):
    """Creates a loss function based on the config."""
    crit_config = config['training']['criterion']
    crit_type = crit_config['type']
    if crit_type == 'BCELoss':
        return nn.BCELoss()
    elif crit_type == 'MSELoss': # Added for AAEs
        return nn.MSELoss()
    elif crit_type == 'BCEWithLogitsLoss': # Added for AAEs
        return nn.BCEWithLogitsLoss()
    # elif crit_type == 'focalloss':
    #     params = crit_config.get('params', {})
    #     return FocalLoss(**params) # Make sure FocalLoss is defined/imported
    # Add other loss functions here
    else:
        raise ValueError(f"Unsupported criterion type: {crit_config['type']}")

def get_scheduler(optimizer, config):
    """Creates a learning rate scheduler based on the config."""
    sched_config = config['training']['scheduler']
    sched_type = sched_config['type'].lower()
    params = sched_config.get('params', {})

    if sched_type == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, **params)
    elif sched_type == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **params)
    elif sched_type == 'none' or sched_type is None:
        return None
    # Add other schedulers here
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_config['type']}")

def get_aae_components(config, device):
    """Initializes enabled AAE models, losses, and optimizers."""
    aae_components = {}
    aae_config = config['aae']

    for aae_name in ['saae', 'taae', 'caae']:
        if aae_config[aae_name]['enabled']:
            comp = {}
            model_type = aae_config[aae_name]['model_type']
            lr = aae_config[aae_name]['lr']
            opt_type = aae_config[aae_name]['optimizer']

            # Instantiate Model
            if model_type == 'AAE_Conv':
                comp['model'] = AAE_Conv().to(device)
            elif model_type == 'AAE':
                input_dim = aae_config[aae_name].get('params', {}).get('input_dim', 768) # Default or from config
                comp['model'] = AAE(input_dim).to(device)
            else:
                raise ValueError(f"Unsupported AAE model type: {model_type}")

            # Instantiate Losses
            comp['rec_loss_fn'] = get_criterion({'training': {'criterion': {'type': aae_config[aae_name]['rec_loss']}}})
            comp['adv_loss_fn'] = get_criterion({'training': {'criterion': {'type': aae_config[aae_name]['adv_loss']}}})

            # Instantiate Optimizer
            if opt_type.lower() == 'adam':
                 comp['optimizer'] = optim.Adam(comp['model'].parameters(), lr=lr)
            elif opt_type.lower() == 'sgd':
                    comp['optimizer'] = optim.SGD(comp['model'].parameters(), lr=lr,
                                                momentum=aae_config[aae_name].get('momentum', 0),
                                                weight_decay=aae_config[aae_name].get('weight_decay', 0))
            # Add other optimizer options if needed
            else:
                 raise ValueError(f"Unsupported AAE optimizer type: {opt_type}")

            aae_components[aae_name] = comp
            print(f"Initialized {aae_name.upper()}")

    return aae_components

# --- Core Functions (Modified) ---

def mask_img(img, mask):
    mask = torch.where(mask >= 0.5, 1, 0)
    maskimg = (img * mask).float()
    return maskimg

def calculate_histogram_pytorch(input_tensor):
    N, C, W, H = input_tensor.shape
    histogram_N = []
    for n in range(N):
        histograms = []
        for i in range(C):
            channel_pixels = input_tensor[n, i, :, :].reshape(-1)
            # Note: Original histc max=0 seems wrong, should be max=255 for typical images,
            # or max=1 if images are normalized to [0, 1]. Assuming [0, 1] based on sigmoid output.
            # Adjust bins/min/max if your image range is different.
            histogram = torch.histc(channel_pixels, bins=256, min=0, max=1)
            # Avoid division by zero if sum is zero
            hist_sum = torch.sum(histogram)
            if hist_sum > 0:
                 histogram = histogram / hist_sum
            histograms.append(histogram)
        histogram_N.append(torch.stack(histograms, dim=0))
    return torch.stack(histogram_N)

class NeuralNetworkCheckpoint:
    def __init__(self, model, optimizer, checkpoint_dir, file_prefix):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.file_prefix = file_prefix
        os.makedirs(self.checkpoint_dir, exist_ok=True) # Ensure directory exists

    def save_checkpoint(self, epoch, loss, is_best=False, filename_suffix=""):
        """Saves checkpoint, optionally marking the best one."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss # Or maybe validation metric like IoU
        }
        base_filename = f"{self.file_prefix}_epoch{epoch}{filename_suffix}.pt"
        filepath = os.path.join(self.checkpoint_dir, base_filename)
        torch.save(state, filepath)
        print(f'Saved checkpoint: {filepath}')

        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, f"{self.file_prefix}_best{filename_suffix}.pt")
            torch.save(state, best_filepath)
            print(f'Saved best checkpoint: {best_filepath}')


    def load_checkpoint(self, filename):
        """Loads checkpoint from a file."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(filepath):
            print(f"Loading checkpoint '{filepath}'")
            checkpoint = torch.load(filepath, map_location=torch.device('cpu')) # Load to CPU first
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', -1) # Use .get for backward compatibility
            loss = checkpoint.get('loss', float('inf'))
            print(f'Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}')
            return epoch # Return epoch number to potentially resume training
        else:
            print(f"Warning: No checkpoint found at '{filepath}'")
            return -1 # Indicate no checkpoint was loaded


def train(config, net, train_iter, test_iter, criterion, optimizer, scheduler, device, aae_components):
    net = net.to(device)
    num_epochs = config['training']['num_epochs']
    log_config = config['logging']
    loss_coeffs = config['training']['loss_coefficients']
    evaluator = Evaluator(num_class=2) # Assuming binary classification based on nclass=1

    # Checkpoint Manager
    checkpoint_manager = NeuralNetworkCheckpoint(
        model=net,
        optimizer=optimizer,
        checkpoint_dir=log_config['checkpoint_dir'],
        file_prefix=f"{log_config['save_name_prefix']}" # Suffix added later
    )

    min_IoU = 0  # Track best IoU for saving best model
    Train_Loss = []
    Seg_Loss = []
    S_Loss = []
    T_Loss = []
    C_Loss = []
    Val_Loss = []
    Val_Accuracy = []
    Val_Precision = []
    Val_ReCall = []
    Val_F1_score = []
    Val_IoU = []

    # --- AAE Training Functions (defined inside train for scope access) ---
    def saae_train(pred, label):
        comp = aae_components['saae']
        x = torch.where(pred >= 0.5, 1, 0).float()
        y = label.float()
        recon_pred, z_pred = comp['model'](x)
        recon_label, z_label = comp['model'](y)
        rec_loss = comp['rec_loss_fn'](recon_pred, x) + comp['rec_loss_fn'](recon_label, y)
        adv_loss = comp['adv_loss_fn'](z_pred, z_label) # Check if this loss needs specific target (e.g., all ones/zeros)
        return rec_loss , adv_loss

    def taae_train(pred, label, LBP):
        comp = aae_components['taae']
        x = mask_img(LBP, pred)
        y = mask_img(LBP, label)
        recon_pred, z_pred = comp['model'](x)
        recon_label, z_label = comp['model'](y)
        rec_loss = comp['rec_loss_fn'](recon_pred, x) + comp['rec_loss_fn'](recon_label, y)
        adv_loss = comp['adv_loss_fn'](z_pred, z_label) # Check if this loss needs specific target
        return rec_loss , adv_loss

    def caae_train(pred, label, img):
        comp = aae_components['caae']
        x = mask_img(img, pred)
        y = mask_img(img, label)
        # Ensure histogram calculation matches expected input range [0, 1]
        x_hist = calculate_histogram_pytorch(x).reshape(img.shape[0], -1)
        y_hist = calculate_histogram_pytorch(y).reshape(img.shape[0], -1)

        recon_pred, z_pred = comp['model'](x_hist)
        recon_label, z_label = comp['model'](y_hist)
        rec_loss = comp['rec_loss_fn'](recon_pred, x_hist) + comp['rec_loss_fn'](recon_label, y_hist)
        adv_loss = comp['adv_loss_fn'](z_pred, z_label) # Check if this loss needs specific target
        return rec_loss , adv_loss

    # --- Training Loop ---
    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        train_loss_sum = 0.0
        seg_loss_sum = 0.0
        s_loss_sum = 0.0
        t_loss_sum = 0.0
        c_loss_sum = 0.0
        tbar = tqdm(train_iter, file=sys.stdout, desc=f"Epoch {epoch+1}/{num_epochs}")

        for count, sample in enumerate(tbar):
            X, y = sample['image'].to(device), sample['label'].to(device)
            LBP = sample.get('LBP', None) # Handle cases where LBP might not be present
            if LBP is not None:
                 LBP = LBP.to(device)

            # --- Generator (Segmentation Model) Update ---
            optimizer.zero_grad()
            y_hat_logits = net(X) # Assuming output before sigmoid
            y_hat = torch.sigmoid(y_hat_logits) # Apply sigmoid for loss & AAEs

            loss_seg = criterion(y_hat, y) # Use sigmoid output for BCE
            loss_G = loss_coeffs['seg_loss'] * loss_seg

            # --- AAE Forward Pass & Loss Calculation (for Generator) ---
            s_rec_loss, s_adv_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            t_rec_loss, t_adv_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            c_rec_loss, c_adv_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            if 'saae' in aae_components:
                aae_components['saae']['optimizer'].zero_grad()
                s_rec_loss, s_adv_loss = saae_train(y_hat.detach(), y) # Detach y_hat for SAAE training? Check logic. Usually G wants to fool D.
                loss_G = loss_G + loss_coeffs['s_adv_loss'] * s_adv_loss
            if 'taae' in aae_components and LBP is not None:
                aae_components['taae']['optimizer'].zero_grad()
                t_rec_loss, t_adv_loss = taae_train(y_hat.detach(), y, LBP) # Detach?
                loss_G = loss_G + loss_coeffs['t_adv_loss'] * t_adv_loss
            if 'caae' in aae_components:
                aae_components['caae']['optimizer'].zero_grad()
                c_rec_loss, c_adv_loss = caae_train(y_hat.detach(), y, X) # Detach?
                loss_G = loss_G + loss_coeffs['c_adv_loss'] * c_adv_loss

            loss_G.backward(retain_graph=True if aae_components else False) # Retain graph only if AAEs need it
            optimizer.step()

            # --- Discriminator (AAE) Update ---
            if 'saae' in aae_components:
                # aae_components['saae']['optimizer'].zero_grad()
                # Recompute SAAE with non-detached y_hat for D update
                # s_rec_loss_d, s_adv_loss_d = saae_train(y_hat, y)
                loss_D_s = s_rec_loss + loss_coeffs['s_disc_adv_loss'] * s_adv_loss # Maximize rec + minimize adv for D? Check AAE paper. Original code: rec - coeff * adv
                loss_D_s.backward()
                aae_components['saae']['optimizer'].step()

            if 'taae' in aae_components and LBP is not None:
                # aae_components['taae']['optimizer'].zero_grad()
                # t_rec_loss_d, t_adv_loss_d = taae_train(y_hat, y, LBP)
                loss_D_t = t_rec_loss + loss_coeffs['t_disc_adv_loss'] * t_adv_loss
                loss_D_t.backward()
                aae_components['taae']['optimizer'].step()

            if 'caae' in aae_components:
                # aae_components['caae']['optimizer'].zero_grad()
                # c_rec_loss_d, c_adv_loss_d = caae_train(y_hat, y, X)
                loss_D_c = c_rec_loss + loss_coeffs['c_disc_adv_loss'] * c_adv_loss
                loss_D_c.backward()
                aae_components['caae']['optimizer'].step()


            train_loss_sum += loss_G.item() # Log generator loss
            seg_loss_sum += loss_seg.item()
            s_loss_sum += s_adv_loss.item()
            t_loss_sum += t_adv_loss.item()
            c_loss_sum += c_adv_loss.item()
            tbar.set_description(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_sum / (count + 1):.4f}")

        Train_Loss.append(train_loss_sum / (count + 1))
        Seg_Loss.append(seg_loss_sum/ (count + 1))
        S_Loss.append(s_loss_sum/ (count + 1))
        T_Loss.append(t_loss_sum/ (count + 1))
        C_Loss.append(c_loss_sum/ (count + 1))

        # --- Validation Phase ---
        net.eval()
        evaluator.reset()
        test_loss_sum = 0.0
        with torch.no_grad():
            for n, sample in enumerate(test_iter):
                X, y = sample['image'].to(device), sample['label'].to(device)
                pred_logits = net(X)
                pred = torch.sigmoid(pred_logits)

                test_loss_sum += criterion(pred, y).item()

                # Evaluate based on thresholded predictions
                pred_binary = (pred >= 0.5).cpu().numpy().astype(np.int16)
                target = y.cpu().numpy().astype(np.int16)
                evaluator.add_batch(target, pred_binary)

        val_loss = test_loss_sum / (n + 1)
        Acc = evaluator.Pixel_Accuracy_Class() # Check if this is Mean Pixel Accuracy
        Precision = evaluator.Precision()[1] # Assuming class 1 is foreground
        Recall = evaluator.ReCall()[1]
        F1 = evaluator.F1_score()[1]
        IoU = evaluator.IoU()[1]

        Val_Loss.append(val_loss)
        Val_Accuracy.append(Acc)
        Val_Precision.append(Precision)
        Val_ReCall.append(Recall)
        Val_F1_score.append(F1)
        Val_IoU.append(IoU)

        print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {time.time() - start:.2f}s")
        print(f"  Train Loss: {Train_Loss[-1]:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Metrics: MPA: {Acc:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1: {F1:.4f}, IoU: {IoU:.4f}")

        # Step the scheduler
        if scheduler:
            scheduler.step()
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}") # Print current LR

        # --- Save Checkpoint and Log ---
        is_best = IoU > min_IoU
        if is_best:
            min_IoU = IoU
            print(f"  New best IoU: {min_IoU:.4f}!")

        # Save the latest checkpoint and optionally the best one
        # checkpoint_manager.save_checkpoint(
        #     epoch=epoch + 1,
        #     loss=val_loss, # Save validation loss or IoU in checkpoint?
        #     is_best=is_best,
        #     filename_suffix=log_config['save_name_suffix']
        # )

        # Periodic checkpoint saving (if enabled)
        save_strategy = log_config.get('save_checkpoint_strategy', 'best_only')  # Default to best_only if missing
        save_freq = log_config.get('save_freq', 1)  # Default to 1 if missing
        if save_strategy == 'periodic_and_best':
            # Save at specified frequency or if it's the last epoch
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                print(f"  Saving periodic checkpoint for epoch {epoch + 1}...")
                # Define periodic checkpoint filename
                # Save using the manager
                checkpoint_manager.save_checkpoint(
                    epoch=epoch + 1,
                    loss=val_loss,
                    is_best=is_best,
                    filename_suffix=log_config['save_name_suffix']
                )
        else:
            # Save the best one
            if is_best:
                epoch_filename = f"{log_config['save_name_prefix']}_best{log_config['save_name_suffix']}.pt"
                # epoch_filepath = os.path.join(log_config['checkpoint_dir'], epoch_filename)
                # best_state={'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': val_loss}
                # torch.save(best_state, epoch_filepath)
                # print(f'Saved best checkpoint: {epoch_filepath}')
        # --- End Checkpoint Saving Logic ---

        # Save log file periodically or at the end
        if (epoch + 1) % log_config['save_freq'] == 0 or epoch == num_epochs - 1:
            log_df = pd.DataFrame({
                'Epoch': list(range(1, epoch + 2)),
                'Train loss': Train_Loss,
                'Seg loss' : Seg_Loss,
                'Shape loss': S_Loss,
                'Texture loss': T_Loss,
                'Color loss': C_Loss,
                'Val loss': Val_Loss,
                'Val_MPA': Val_Accuracy,
                'Val_Precision': Val_Precision,
                'Val_ReCall': Val_ReCall,
                'Val_F1_score': Val_F1_score,
                'Val_IoU': Val_IoU
            })
            log_filename = f"log_{log_config['save_name_prefix']}_epochs{num_epochs}{log_config['save_name_suffix']}.csv"
            log_filepath = os.path.join(log_config['log_dir'], log_filename)
            os.makedirs(log_config['log_dir'], exist_ok=True)
            log_df.to_csv(log_filepath, index=False, sep=',') # Use CSV for better compatibility
            print(f"Saved training log to {log_filepath}")

    print("Training finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Segmentation Model with optional AAEs from Config")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)

    # 2. Setup Seed and Device
    setup_seed(config['general']['seed'])
    device = get_device(config)
    print(f"Using device: {device}")

    # 3. Load Datasets and Dataloaders
    data_cfg = config['data']
    train_dataset = Dataset(imagepath=data_cfg['train_image_path'],
                            labelpath=data_cfg['train_label_path'])
    test_dataset = Dataset(imagepath=data_cfg['test_image_path'],
                           labelpath=data_cfg['test_label_path'])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_iter = DataLoader(dataset=train_dataset, batch_size=data_cfg['batch_size'],
                            shuffle=True, num_workers=data_cfg['num_workers'], drop_last=True, pin_memory=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=data_cfg['batch_size'],
                           shuffle=False, num_workers=data_cfg['num_workers'], drop_last=False, pin_memory=True) # Usually don't drop last in test

    # 4. Initialize Model
    model_cfg = config['model']
    if model_cfg['name'] == 'UPerNet':
        # Pass parameters directly from config
        model = UPerNet(in_channel=model_cfg['params']['in_channel'],num_class=model_cfg['params']['num_class'])
    elif model_cfg['name'] == 'TransUNet':
        model = TransUNet(img_size=model_cfg['params']['img_size'],num_classes=model_cfg['params']['num_class'])
    elif model_cfg['name'] == 'DeepLab':
        model = DeepLab(backbone=model_cfg['params']['backbone'], num_classes=model_cfg['params']['num_class'],
                        output_stride=model_cfg['params']['output_stride'],freeze_bn=model_cfg['params']['freeze_bn'],
                        pretrained=model_cfg['params']['pretrained'])
    elif model_cfg['name'] == 'UNet':
        model = UNet(n_channels=model_cfg['params']['n_channels'], n_classes=model_cfg['params']['num_class'],
                     bilinear=model_cfg['params']['bilinear'])
    elif model_cfg['name'] == 'SegFormer':
        model = SegFormer(num_classes=model_cfg['params']['num_class'], phi=model_cfg['params']['phi'],
                          pretrained=model_cfg['params']['pretrained'])
    else:
        raise ValueError(f"Unsupported model name: {model_cfg['name']}")
    # print(model) # Optional: print model structure

    # 5. Initialize Optimizers, Criterion, Scheduler
    optimizer = get_optimizer(model.parameters(), config)
    criterion = get_criterion(config)
    scheduler = get_scheduler(optimizer, config)

    # 6. Initialize AAE Components (if enabled)
    aae_components = get_aae_components(config, device)

    # 7. Start Training
    train(config, model, train_iter, test_iter, criterion, optimizer, scheduler, device, aae_components)
