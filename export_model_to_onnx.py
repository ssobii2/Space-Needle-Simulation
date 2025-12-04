"""
Export PPO Model to ONNX Format for Browser Inference
------------------------------------------------------
Converts stable-baselines3 PPO model to ONNX format for use with ONNX.js in the browser.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
import onnx
import onnxruntime as ort

def export_ppo_to_onnx(model_path="ppo_needle_rw_residual.zip", output_path="ppo_needle_rw_residual.onnx"):
    """
    Export PPO model's policy network to ONNX format.
    
    Args:
        model_path: Path to the saved PPO model (.zip file)
        output_path: Output path for the ONNX model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading PPO model from {model_path}...")
    model = PPO.load(model_path)
    
    # Get the policy network (PyTorch module)
    policy = model.policy
    
    # Set to evaluation mode
    policy.eval()
    
    # Create a wrapper class that combines feature extraction and actor network
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, obs):
            # Extract features using policy's feature extractor
            features = self.policy.extract_features(obs)
            # Get latent representation for actor
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            # Get mean action (deterministic prediction)
            mean_actions = self.policy.action_net(latent_pi)
            # Clip actions to [-1, 1] to match original model's predict() behavior
            mean_actions = torch.clamp(mean_actions, -1.0, 1.0)
            return mean_actions
    
    # Wrap the policy
    wrapped_policy = PolicyWrapper(policy)
    wrapped_policy.eval()
    
    # Create dummy input matching observation space shape
    # Observation: [jx, jy, ox, oy, w1, w3, w2, w4] - 8 values
    dummy_input = torch.zeros((1, 8), dtype=torch.float32)
    
    print("Exporting policy network to ONNX...")
    
    # Export to ONNX
    # Use opset 13 for better compatibility with ONNX Runtime Web
    # ONNX Runtime Web has better support for opset 13
    torch.onnx.export(
        wrapped_policy,
        dummy_input,
        output_path,
        input_names=['observation'],
        output_names=['action'],
        opset_version=18,  # Keep opset 18 (PyTorch requires it, and ONNX Runtime Web supports it)
        do_constant_folding=True,
        verbose=False,
    )
    
    print(f"Model exported to {output_path}")
    
    # Ensure ONNX is stored as a single file (no external .data)
    print("Re-saving ONNX model to embed parameters...")
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save_model(onnx_model, output_path, save_as_external_data=False)
    
    # Remove any leftover external data file
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"Removed external data file: {data_path}")
    
    # Verify the exported model
    print("Verifying exported ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference with ONNX Runtime
    print("Testing ONNX model inference...")
    session = ort.InferenceSession(output_path)
    
    # Test with dummy observation
    test_obs = np.zeros((1, 8), dtype=np.float32)
    outputs = session.run(None, {'observation': test_obs})
    print(f"ONNX model output shape: {outputs[0].shape}")
    print(f"ONNX model output sample: {outputs[0][0]}")
    
    # Compare with original model
    print("Comparing with original model...")
    original_action, _ = model.predict(test_obs[0], deterministic=True)
    print(f"Original model output: {original_action}")
    print(f"ONNX model output: {outputs[0][0]}")
    
    print("Export completed successfully!")

if __name__ == "__main__":
    export_ppo_to_onnx()

