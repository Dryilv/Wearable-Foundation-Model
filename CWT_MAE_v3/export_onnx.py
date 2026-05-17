import os
import argparse
import yaml
import torch
import copy

from model_finetune import TF_MAE_Classifier


def export_onnx(config_path, checkpoint_path, output_path=None, opset_version=14):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    data_cfg = config['data']
    
    print(f"Initializing CWT-MAE Classifier...")
    print(f"  - Pretrained: {model_cfg.get('pretrained_path', 'None')}")
    print(f"  - Signal Length: {data_cfg['signal_len']}")
    print(f"  - Num Classes: {data_cfg['num_classes']}")
    
    model = TF_MAE_Classifier(
        pretrained_path=model_cfg.get('pretrained_path'),
        num_classes=data_cfg['num_classes'],
        signal_len=data_cfg['signal_len'],
        cwt_scales=model_cfg.get('cwt_scales', 64),
        patch_size_time=model_cfg.get('patch_size_time', 25),
        patch_size_freq=model_cfg.get('patch_size_freq', 8),
        embed_dim=model_cfg.get('embed_dim', 768),
        depth=model_cfg.get('depth', 12),
        num_heads=model_cfg.get('num_heads', 12),
        use_diff=model_cfg.get('use_diff', False),
        decoder_embed_dim=model_cfg.get('decoder_embed_dim', 512),
        decoder_depth=model_cfg.get('decoder_depth', 8),
        decoder_num_heads=model_cfg.get('decoder_num_heads', 16),
        use_cot=model_cfg.get('use_cot', True),
        num_reasoning_tokens=model_cfg.get('num_reasoning_tokens', 16),
        use_stats_features=model_cfg.get('use_stats_features', False)
    )
    
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    
    model_cpu = model.cpu()
    model_cpu.eval()
    
    if output_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_cpu.onnx")
    
    signal_len = data_cfg['signal_len']
    dummy_x = torch.randn(1, 1, signal_len, device='cpu')
    
    print(f"\nExporting ONNX model to: {output_path}")
    print(f"  - Input shape: (batch_size, channels, {signal_len})")
    print(f"  - Opset version: {opset_version}")
    print(f"  - Dynamic axes: batch_size, channels")
    
    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            (dummy_x,),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'channels'},
                'output': {0: 'batch_size'}
            }
        )
    
    print(f"\n✓ ONNX model successfully exported to: {output_path}")
    
    print("\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
        
        print("\nONNX Model Info:")
        print(f"  - Inputs: {[inp.name for inp in onnx_model.graph.input]}")
        print(f"  - Outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except ImportError:
        print("Note: onnx package not installed. Skipping verification.")
        print("Install with: pip install onnx")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export trained model to ONNX format')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., finetune_config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX file path (default: same dir as checkpoint)')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version (default: 14)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    export_onnx(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()