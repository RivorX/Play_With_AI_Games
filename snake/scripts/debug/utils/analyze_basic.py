"""
🔍 BASIC ANALYSIS MODULE - UPDATED FOR MULTI-QUERY CROSS-ATTENTION
- analyze_basic_states: Analiza podstawowych stanów, aktywacje, attention heatmaps
- visualize_cnn_output: Wizualizacja warstw CNN + Attention weights
- visualize_viewport: Wizualizacja viewport 12x12
- generate_attention_heatmap: Generowanie attention heatmap (IMPROVED: shows query patterns)
✅ UPDATED: Multi-Query Attention support + Query-specific analysis
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def analyze_basic_states(model, env, output_dirs, action_names, config):
    """
    Analiza podstawowych stanów: aktywacje, attention heatmaps, gradienty
    🆕 UPDATED: Multi-Query Cross-Attention + Positional Encoding
    ✅ FIXED: Obsługuje tryb z i bez scalarów (CNN-ONLY mode)
    """
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # 🎯 DETECT SCALARS MODE
    scalars_enabled = features_extractor.scalars_enabled
    
    action_probs_list = []
    detailed_activations = []
    layer_gradients = []
    attention_heatmaps = []
    
    # Inicjalizuj stany LSTM
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    # Analiza 3 stanów
    for state_idx in range(3):
        obs, _ = env.reset()
        
        # Konwersja obs do tensora
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.tensor(value, dtype=torch.float32, device=policy.device).unsqueeze(0)
            else:
                obs_tensor[key] = torch.tensor([value], dtype=torch.float32, device=policy.device).unsqueeze(0)
        
        episode_starts_tensor = torch.tensor(episode_starts, dtype=torch.bool, device=policy.device)
        
        # Forward pass (bez gradów)
        with torch.no_grad():
            # CNN features
            image = obs_tensor['image']
            if image.dim() == 4 and image.shape[-1] == 1:
                image = image.permute(0, 3, 1, 2)
            
            # 🔥 HYBRID CNN: 5×5 → 3×3 → 3×3 (BEST OF BOTH!)
            x = image
            
            # Block 1: Fast global context with 5×5 (RF: 5×5)
            x = features_extractor.conv1(x)
            x = features_extractor.norm1(x)
            x = torch.nn.functional.silu(x)
            
            # Block 2: Local refinement with 3×3 (RF: 7×7)
            x = features_extractor.conv2(x)
            x = features_extractor.norm2(x)
            x = torch.nn.functional.silu(x)
            
            # Block 3: Final features + STRIDED DOWNSAMPLE with 3×3 (RF: 9×9)
            x = features_extractor.conv3(x)
            x = features_extractor.norm3(x)
            x = torch.nn.functional.silu(x)
            x = features_extractor.dropout3(x)
            
            # 🔥 NO MAXPOOL! Downsampling done by strided conv
            
            # 🆕 POSITIONAL ENCODING
            x = features_extractor.pos_encoding(x)
            
            # Flatten spatial features for attention
            spatial_features = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # ✅ CONDITIONAL: Scalars only if enabled
            if scalars_enabled:
                scalars = torch.cat([
                    obs_tensor['direction'],
                    obs_tensor['dx_head'],
                    obs_tensor['dy_head'],
                    obs_tensor['front_coll'],
                    obs_tensor['left_coll'],
                    obs_tensor['right_coll'],
                    obs_tensor['snake_length']
                ], dim=-1)
                
                scalars = features_extractor.scalar_input_dropout(scalars)
                scalar_features = features_extractor.scalar_network(scalars)
                
                # ==================== PATH 1: ATTENTION ====================
                # 🔥 FIX: Normalize BEFORE attention
                normalized_features = features_extractor.cnn_prenorm(spatial_features)
                attended_cnn = features_extractor.cross_attention(normalized_features, scalar_features)
            else:
                # CNN-ONLY: no attention, no scalars
                attended_cnn = torch.zeros(1, 0, device=policy.device)
                scalar_features = torch.zeros(1, 0, device=policy.device)
            
            # ==================== PATH 2: SKIP CONNECTION ====================
            spatial_features_flat = spatial_features.flatten(1)
            
            if scalars_enabled and features_extractor.cnn_direct is not None:
                # With scalars: use compressed direct CNN
                direct_cnn = features_extractor.cnn_direct(spatial_features_flat)
            elif not scalars_enabled and hasattr(features_extractor, 'cnn_only_direct') and features_extractor.cnn_only_direct is not None:
                # CNN-ONLY: use cnn_only_direct compression
                direct_cnn = features_extractor.cnn_only_direct(spatial_features_flat)
            else:
                # Fallback for old models: detect expected fusion input size
                # Old models: fusion expects 1024 (512 attended + 512 direct)
                # But we have 1728 raw features, so we need to compress
                expected_fusion_size = features_extractor.fusion[0].in_features
                current_size = attended_cnn.shape[-1] + spatial_features_flat.shape[-1] + scalar_features.shape[-1]
                
                if current_size > expected_fusion_size:
                    # Need compression - create temporary linear layer
                    direct_size = expected_fusion_size - attended_cnn.shape[-1] - scalar_features.shape[-1]
                    if not hasattr(features_extractor, '_temp_direct_projection'):
                        features_extractor._temp_direct_projection = torch.nn.Linear(
                            spatial_features_flat.shape[-1], 
                            direct_size,
                            device=spatial_features_flat.device
                        )
                    direct_cnn = features_extractor._temp_direct_projection(spatial_features_flat)
                else:
                    direct_cnn = spatial_features_flat
            
            # ==================== FUSION - THREE STREAMS ====================
            fused = torch.cat([attended_cnn, direct_cnn, scalar_features], dim=-1)
            features_final = features_extractor.fusion(fused)
            
            # LSTM
            if lstm_states is None:
                batch_size = features_final.shape[0]
                num_layers = policy.lstm_actor.num_layers
                hidden_size = policy.lstm_actor.hidden_size
                
                lstm_states = (
                    np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32),
                    np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
                )
            
            lstm_states_tensor = (
                torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device),
                torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device)
            )
            
            features_seq = features_final.unsqueeze(1)
            lstm_out, new_lstm_states = policy.lstm_actor(features_seq, lstm_states_tensor)
            lstm_states = (new_lstm_states[0].cpu().numpy(), new_lstm_states[1].cpu().numpy())
            
            latent_pi = lstm_out.squeeze(1)
            latent_vf = lstm_out.squeeze(1)
            
            # MLP
            latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi)
            latent_vf_mlp = policy.mlp_extractor.value_net(latent_vf)
            
            # Action distribution
            action_logits = policy.action_net(latent_pi_mlp)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1)
            
            # Value
            value = policy.value_net(latent_vf_mlp)
        
        # Zapisz dane
        action_probs_np = action_probs[0].cpu().numpy()
        action_probs_list.append({
            'state': state_idx,
            'probs': action_probs_np,
            'action': action.item(),
            'value': value.item()
        })
        
        # ✅ Better metrics: RMS (Root Mean Square)
        attended_np = attended_cnn[0].cpu().numpy()
        direct_np = direct_cnn[0].cpu().numpy()  # ✨ NEW!
        scalar_np = scalar_features[0].cpu().numpy()
        features_np = features_final[0].cpu().numpy()
        lstm_np = latent_pi[0].cpu().numpy()
        
        # ✅ CONDITIONAL: Handle empty arrays when scalars disabled
        activation_data = {
            'state': state_idx,
            # RMS = sqrt(mean(x^2))
            'direct_rms': np.sqrt(np.mean(direct_np**2)),        # ✨ NEW!
            'direct_absmax': np.abs(direct_np).max() if direct_np.size > 0 else 0,            # ✨ NEW!
            'direct_active_ratio': (np.abs(direct_np) > 0.01).mean() if direct_np.size > 0 else 0,  # ✨ NEW!
            'features_rms': np.sqrt(np.mean(features_np**2)),
            'lstm_rms': np.sqrt(np.mean(lstm_np**2)),
            'lstm_hidden_rms': np.sqrt(np.mean(lstm_states[0][0]**2)),
            'lstm_cell_rms': np.sqrt(np.mean(lstm_states[1][0]**2)),
        }
        
        # Add attention metrics only if scalars enabled
        if scalars_enabled and attended_np.size > 0:
            activation_data['attended_rms'] = np.sqrt(np.mean(attended_np**2))
            activation_data['attended_absmax'] = np.abs(attended_np).max()
            activation_data['attended_active_ratio'] = (np.abs(attended_np) > 0.01).mean()
        else:
            activation_data['attended_rms'] = 0
            activation_data['attended_absmax'] = 0
            activation_data['attended_active_ratio'] = 0
        
        # Add scalar metrics only if scalars enabled
        if scalars_enabled and scalar_np.size > 0:
            activation_data['scalar_rms'] = np.sqrt(np.mean(scalar_np**2))
            activation_data['scalar_absmax'] = np.abs(scalar_np).max()
            activation_data['scalar_active_ratio'] = (np.abs(scalar_np) > 0.01).mean()
        else:
            activation_data['scalar_rms'] = 0
            activation_data['scalar_absmax'] = 0
            activation_data['scalar_active_ratio'] = 0
        
        detailed_activations.append(activation_data)
        
        # Wizualizacja CNN output + Attention
        visualize_cnn_output(obs_tensor, features_extractor, output_dirs['conv_viz'], state_idx, scalars_enabled)
        
        # Wizualizacja viewport
        visualize_viewport(obs, output_dirs['viewport'], state_idx)
        
        # Compute layer gradients
        from utils.analyze_gradients import compute_layer_gradients
        state_layer_grads = compute_layer_gradients(
            model, obs, obs_tensor, lstm_states, action.item(), features_extractor
        )
        layer_gradients.append(state_layer_grads)
        
        # Generate attention heatmap (🆕 IMPROVED: shows query patterns)
        attention_map = generate_attention_heatmap(
            model, obs, obs_tensor, lstm_states, action.item(),
            output_dirs['heatmap'], state_idx, action_names
        )
        if attention_map is not None:
            attention_heatmaps.append(attention_map)
        
        print(f"   Stan {state_idx}: Akcja={action_names[action.item()]} (probs={action_probs_np})")
    
    return action_probs_list, detailed_activations, layer_gradients, attention_heatmaps


def generate_attention_heatmap(model, obs, obs_tensor, lstm_states, action_idx, output_dir, state_idx, action_names):
    """
    Generuje attention heatmap używając gradientów
    🆕 UPDATED: Multi-Query Cross-Attention visualization
    ✅ FIXED: Obsługuje tryb z i bez scalarów
    """
    policy = model.policy
    features_extractor = policy.features_extractor
    
    # 🎯 DETECT SCALARS MODE
    scalars_enabled = features_extractor.scalars_enabled
    
    # Ustaw tryby
    features_extractor.eval()
    policy.lstm_actor.train()
    policy.mlp_extractor.train()
    policy.action_net.train()
    
    obs_grad = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            obs_grad[k] = torch.tensor(v, dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
        else:
            obs_grad[k] = torch.tensor([v], dtype=torch.float32, device=policy.device, requires_grad=True).unsqueeze(0)
    
    # Forward pass
    image_grad = obs_grad['image']
    if image_grad.dim() == 4 and image_grad.shape[-1] == 1:
        image_grad = image_grad.permute(0, 3, 1, 2)
    
    image_grad.retain_grad()
    
    # CNN
    x = image_grad
    x = features_extractor.conv1(x)
    x = features_extractor.norm1(x)
    x = torch.nn.functional.silu(x)
    
    x = features_extractor.conv2(x)
    x = features_extractor.norm2(x)
    x = torch.nn.functional.silu(x)
    
    x = features_extractor.conv3(x)
    x = features_extractor.norm3(x)
    x = torch.nn.functional.silu(x)
    x = features_extractor.dropout3(x)
    
    # 🔥 NO MAXPOOL! Downsampling done by strided conv in conv3
    
    x = features_extractor.pos_encoding(x)
    
    spatial_features = x.flatten(2).transpose(1, 2)
    
    # ✅ CONDITIONAL: Scalars only if enabled
    if scalars_enabled and features_extractor.scalar_network is not None:
        scalars_grad = torch.cat([
            obs_grad['direction'],
            obs_grad['dx_head'],
            obs_grad['dy_head'],
            obs_grad['front_coll'],
            obs_grad['left_coll'],
            obs_grad['right_coll'],
            obs_grad['snake_length']
        ], dim=-1)
        
        scalars_grad = features_extractor.scalar_input_dropout(scalars_grad)
        scalar_features = features_extractor.scalar_network(scalars_grad)
        
        # ==================== PATH 1: ATTENTION with gradient tracking ====================
        # 🔥 FIX: Normalize BEFORE attention
        normalized_features = features_extractor.cnn_prenorm(spatial_features)
        normalized_features.retain_grad()
        
        attended_cnn = features_extractor.cross_attention(normalized_features, scalar_features)
        attended_cnn.retain_grad()
        
        # ==================== PATH 2: SKIP CONNECTION with gradient tracking ====================
        spatial_features_flat = spatial_features.flatten(1)
        spatial_features_flat.retain_grad()
        
        direct_cnn = features_extractor.cnn_direct(spatial_features_flat)
        direct_cnn.retain_grad()
    else:
        # CNN-ONLY: no attention, no scalars
        attended_cnn = torch.zeros(1, 0, device=policy.device, requires_grad=True)
        scalar_features = torch.zeros(1, 0, device=policy.device, requires_grad=True)
        
        spatial_features_flat = spatial_features.flatten(1)
        spatial_features_flat.retain_grad()
        
        # ✅ Use cnn_only_direct compression for CNN-ONLY mode
        if hasattr(features_extractor, 'cnn_only_direct') and features_extractor.cnn_only_direct is not None:
            direct_cnn = features_extractor.cnn_only_direct(spatial_features_flat)
        else:
            # Fallback for old models: detect expected fusion input size
            expected_fusion_size = features_extractor.fusion[0].in_features
            current_size = attended_cnn.shape[-1] + spatial_features_flat.shape[-1] + scalar_features.shape[-1]
            
            if current_size > expected_fusion_size:
                # Need compression
                direct_size = expected_fusion_size - attended_cnn.shape[-1] - scalar_features.shape[-1]
                if not hasattr(features_extractor, '_temp_direct_projection'):
                    features_extractor._temp_direct_projection = torch.nn.Linear(
                        spatial_features_flat.shape[-1], 
                        direct_size,
                        device=spatial_features_flat.device
                    )
                direct_cnn = features_extractor._temp_direct_projection(spatial_features_flat)
            else:
                direct_cnn = spatial_features_flat
        direct_cnn.retain_grad()
        
        attended_cnn.retain_grad()
        scalar_features.retain_grad()
    
    # ==================== FUSION - THREE STREAMS ====================
    fused = torch.cat([attended_cnn, direct_cnn, scalar_features], dim=-1)
    fused.retain_grad()
    features_final = features_extractor.fusion(fused)
    features_final.retain_grad()
    
    # LSTM
    lstm_states_tensor = (
        torch.tensor(lstm_states[0], dtype=torch.float32, device=policy.device, requires_grad=False),
        torch.tensor(lstm_states[1], dtype=torch.float32, device=policy.device, requires_grad=False)
    )
    
    features_seq = features_final.unsqueeze(1)
    lstm_out, _ = policy.lstm_actor(features_seq, lstm_states_tensor)
    latent_pi_grad = lstm_out.squeeze(1)
    
    latent_pi_mlp = policy.mlp_extractor.policy_net(latent_pi_grad)
    logits_grad = policy.action_net(latent_pi_mlp)
    selected_logit = logits_grad[0, action_idx]
    
    # Backward
    model.policy.zero_grad()
    selected_logit.backward()
    
    # Gradient względem input image
    if image_grad.grad is not None:
        grad_map = image_grad.grad[0, 0].cpu().numpy()
        
        # Normalizuj gradient
        grad_map = np.abs(grad_map)
        if grad_map.max() > 0:
            grad_map = grad_map / grad_map.max()
        
        # Wizualizacja
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original viewport
        viewport = obs['image'][:, :, 0]
        axes[0].imshow(viewport, cmap='viridis', interpolation='nearest')
        axes[0].set_title(f'Original Viewport - Stan {state_idx}')
        axes[0].axis('off')
        
        # Attention heatmap
        im = axes[1].imshow(grad_map, cmap='hot', interpolation='bilinear', alpha=0.8)
        axes[1].set_title(f'Attention Heatmap - Akcja: {action_names[action_idx]}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Gradient Magnitude')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, f'attention_state_{state_idx}.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        
        print(f'  ✅ Attention heatmap zapisana: {heatmap_path}')
    
    model.policy.eval()
    return None


def visualize_cnn_output(obs_tensor, features_extractor, output_dir, state_idx, scalars_enabled=False):
    """
    Wizualizuje output wszystkich warstw CNN + Attention + Skip Connection
    🆕 UPDATED: Fixed CNN Architecture with Skip Connection (Direct CNN Path)
    ✅ FIXED: Obsługuje 3 warstwy CNN (5×5 → 3×3 → 3×3) bez MaxPool
    
    Trzy główne ścieżki:
    PATH 1 (Attention): CNN → Prenorm → Attention → 448 dim (gdy scalary enabled)
    PATH 2 (Skip): CNN → Direct projection → 256 dim ✨ NEW!
    PATH 3 (Scalars): Scalar network → 256 dim
    """
    # 🎯 DETECT SCALARS MODE
    scalars_enabled = features_extractor.scalars_enabled
    
    with torch.no_grad():
        image = obs_tensor['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)

        x = image

        # 🔥 HYBRID CNN: 5×5 → 3×3 → 3×3
        
        # Conv1: Fast global context with 5×5 (RF: 5×5)
        x = features_extractor.conv1(image)
        x = features_extractor.norm1(x)
        x = torch.nn.functional.silu(x)
        conv1_output = x[0].cpu().numpy()

        # Conv2: Local refinement with 3×3 (RF: 7×7)
        x = features_extractor.conv2(x)
        x = features_extractor.norm2(x)
        x = torch.nn.functional.silu(x)
        conv2_output = x[0].cpu().numpy()
        
        # Conv3: Final features + STRIDED DOWNSAMPLE with 3×3 (RF: 9×9)
        x = features_extractor.conv3(x)
        x = features_extractor.norm3(x)
        x = torch.nn.functional.silu(x)
        x = features_extractor.dropout3(x)
        conv3_output = x[0].cpu().numpy()
        
        # 🔥 NO MAXPOOL! Downsampling done by strided conv in conv3
        
        # Positional Encoding
        x = features_extractor.pos_encoding(x)
        pos_encoded_output = x[0].cpu().numpy()

        activations = {
            'conv1_output': conv1_output,
            'conv2_output': conv2_output,
            'conv3_output': conv3_output,
            'pos_encoded': pos_encoded_output
        }
        
        # Flatten for attention
        spatial_features = x.flatten(2).transpose(1, 2)
        spatial_features_flat = spatial_features.flatten(1)  # Raw CNN dla skip connection

        # ✅ CONDITIONAL: Scalars only if enabled
        if scalars_enabled and features_extractor.scalar_network is not None:
            scalars = torch.cat([
                obs_tensor['direction'],
                obs_tensor['dx_head'],
                obs_tensor['dy_head'],
                obs_tensor['front_coll'],
                obs_tensor['left_coll'],
                obs_tensor['right_coll'],
                obs_tensor['snake_length']
            ], dim=-1)
            scalars = features_extractor.scalar_input_dropout(scalars)
            scalar_features = features_extractor.scalar_network(scalars)
            activations['scalar'] = scalar_features[0].cpu().numpy()
            
            # ==================== PATH 1: ATTENTION PATH ====================
            normalized_features = features_extractor.cnn_prenorm(spatial_features)
            attended_cnn = features_extractor.cross_attention(normalized_features, scalar_features)
            activations['attended_cnn'] = attended_cnn[0].cpu().numpy()
            
            # ==================== PATH 2: SKIP CONNECTION (DIRECT CNN) ====================
            direct_cnn = features_extractor.cnn_direct(spatial_features_flat)
            activations['direct_cnn'] = direct_cnn[0].cpu().numpy()
        else:
            # CNN-ONLY: no attention, no scalars
            attended_cnn = torch.zeros(1, 0, device=features_extractor.conv1.weight.device)
            scalar_features = torch.zeros(1, 0, device=features_extractor.conv1.weight.device)
            
            # ✅ Use cnn_only_direct compression for CNN-ONLY mode
            if hasattr(features_extractor, 'cnn_only_direct') and features_extractor.cnn_only_direct is not None:
                direct_cnn = features_extractor.cnn_only_direct(spatial_features_flat)
            else:
                # Fallback for old models: detect expected fusion input size
                expected_fusion_size = features_extractor.fusion[0].in_features
                current_size = attended_cnn.shape[-1] + spatial_features_flat.shape[-1] + scalar_features.shape[-1]
                
                if current_size > expected_fusion_size:
                    # Need compression
                    direct_size = expected_fusion_size - attended_cnn.shape[-1] - scalar_features.shape[-1]
                    if not hasattr(features_extractor, '_temp_direct_projection'):
                        features_extractor._temp_direct_projection = torch.nn.Linear(
                            spatial_features_flat.shape[-1], 
                            direct_size,
                            device=spatial_features_flat.device
                        )
                    direct_cnn = features_extractor._temp_direct_projection(spatial_features_flat)
                else:
                    direct_cnn = spatial_features_flat
            
            activations['scalar'] = np.array([])
            activations['attended_cnn'] = np.array([])
            activations['direct_cnn'] = direct_cnn[0].cpu().numpy()

        # Fusion z oboma ścieżkami
        fused = torch.cat([attended_cnn, direct_cnn, scalar_features], dim=-1)
        fusion = features_extractor.fusion(fused)
        activations['fusion'] = fusion[0].cpu().numpy()

    # ==================== WIZUALIZACJA ====================

    # 1. CONV1 OUTPUT (5×5)
    visualize_conv_layer(
        activations['conv1_output'],
        layer_name='Conv1 Output (5×5 kernel, RF=5×5)',
        output_path=os.path.join(output_dir, f'cnn_conv1_state_{state_idx}.png'),
        num_channels=min(16, activations['conv1_output'].shape[0])
    )

    # 2. CONV2 OUTPUT (3×3)
    visualize_conv_layer(
        activations['conv2_output'],
        layer_name='Conv2 Output (3×3 kernel, RF=7×7)',
        output_path=os.path.join(output_dir, f'cnn_conv2_state_{state_idx}.png'),
        num_channels=min(16, activations['conv2_output'].shape[0])
    )
    
    # 3. CONV3 OUTPUT (3×3 strided, RF=9×9) 🔥 NEW!
    visualize_conv_layer(
        activations['conv3_output'],
        layer_name='Conv3 Output (3×3 strided kernel, RF=9×9, stride=2)',
        output_path=os.path.join(output_dir, f'cnn_conv3_state_{state_idx}.png'),
        num_channels=min(16, activations['conv3_output'].shape[0])
    )
    
    # 4. POSITIONAL ENCODING
    visualize_conv_layer(
        activations['pos_encoded'],
        layer_name='After Positional Encoding',
        output_path=os.path.join(output_dir, f'cnn_pos_enc_state_{state_idx}.png'),
        num_channels=min(16, activations['pos_encoded'].shape[0])
    )

    # 5. BOTH CNN PATHS COMPARISON (1D) - 🔥 NEW!
    paths_dict = {}
    
    if scalars_enabled:
        paths_dict[f'Attended CNN via Attention ({activations["attended_cnn"].shape[0]} dim)'] = activations['attended_cnn']
    
    paths_dict[f'Direct CNN ({activations["direct_cnn"].shape[0]} dim) {"✨ (Raw CNN)" if not scalars_enabled else "(Skip Connection)"}'] = activations['direct_cnn']
    
    visualize_1d_features(
        paths_dict,
        output_path=os.path.join(output_dir, f'cnn_paths_comparison_state_{state_idx}.png'),
        state_idx=state_idx
    )

    # 6. FUSION WITH ALL STREAMS (1D features)
    fusion_dict = {}
    
    if scalars_enabled and activations['scalar'].size > 0:
        fusion_dict[f'Scalar features ({activations["scalar"].shape[0]} dim)'] = activations['scalar']
    
    fusion_dict[f'Fused output ({activations["fusion"].shape[0]} dim)'] = activations['fusion']
    
    visualize_1d_features(
        fusion_dict,
        output_path=os.path.join(output_dir, f'fusions_state_{state_idx}.png'),
        state_idx=state_idx
    )

    # 7. HEATMAPA WSZYSTKICH KANAŁÓW CONV3 (final features before pos enc)
    visualize_all_channels_heatmap(
        activations['conv3_output'],
        layer_name='Conv3 All Channels (Final CNN features)',
        output_path=os.path.join(output_dir, f'cnn_conv3_all_channels_state_{state_idx}.png')
    )

    print(f'  ✅ CNN+Attention+Skip Connection visualization zapisana dla stanu {state_idx}')


def visualize_conv_layer(activation, layer_name, output_path, num_channels=16):
    """Wizualizuje wybrane kanały warstwy konwolucyjnej (grid 4x4)"""
    num_channels = min(num_channels, activation.shape[0])
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(num_channels):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(activation[i], cmap='viridis', interpolation='nearest')
        axes[row, col].set_title(f'Ch {i}', fontsize=10)
        axes[row, col].axis('off')
    
    # Wyłącz puste subploty
    for i in range(num_channels, 16):
        axes[i // 4, i % 4].axis('off')
    
    plt.suptitle(f'{layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_1d_features(features_dict, output_path, state_idx):
    """
    Wizualizuje 1D features (attention, projection, final output)
    jako heatmapy poziome
    ✅ FIXED: Obsługuje puste tablice (gdy brak scalarów)
    """
    fig, axes = plt.subplots(len(features_dict), 1, figsize=(16, 6))
    
    if len(features_dict) == 1:
        axes = [axes]
    
    for idx, (name, features) in enumerate(features_dict.items()):
        # ✅ Skip empty features
        if features.size == 0:
            axes[idx].text(0.5, 0.5, f'{name}\n(disabled)', 
                          ha='center', va='center', fontsize=12, transform=axes[idx].transAxes)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        # Reshape do 2D dla imshow
        features_2d = features.reshape(1, -1)
        
        im = axes[idx].imshow(features_2d, cmap='viridis', aspect='auto', interpolation='nearest')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Feature Index')
        axes[idx].set_yticks([])
        
        # Colorbar
        plt.colorbar(im, ax=axes[idx], orientation='horizontal', pad=0.1)
        
        # Statystyki
        mean_val = features.mean()
        std_val = features.std()
        max_val = features.max() if features.size > 0 else 0
        min_val = features.min() if features.size > 0 else 0
        axes[idx].text(0.02, 0.95, f'Mean: {mean_val:.3f}, Std: {std_val:.3f}, Max: {max_val:.3f}, Min: {min_val:.3f}',
                      transform=axes[idx].transAxes, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_all_channels_heatmap(activation, layer_name, output_path):
    """
    Wizualizuje WSZYSTKIE kanały jako heatmapę (channels x spatial)
    """
    num_channels, height, width = activation.shape
    
    # Flatten spatial dimensions
    activation_flat = activation.reshape(num_channels, -1)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(activation_flat, cmap='viridis', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('Spatial Position (flattened)', fontsize=12)
    ax.set_ylabel('Channel Index', fontsize=12)
    ax.set_title(f'{layer_name} - All Channels Heatmap', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation Value')
    
    # Dodaj linie co width pikseli
    for i in range(1, height):
        ax.axvline(i * width - 0.5, color='white', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_viewport(obs, output_dir, state_idx):
    """Wizualizuje viewport 12x12"""
    viewport = obs['image'][:, :, 0]
    plt.figure(figsize=(8, 8))
    plt.imshow(viewport, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Wartość')
    plt.title(f'Viewport 12x12 - Stan {state_idx}')
    
    legend_elements = [
        Patch(facecolor='purple', label='Ściana (-1.0)'),
        Patch(facecolor='black', label='Puste (0.0)'),
        Patch(facecolor='cyan', label='Ciało (0.5)'),
        Patch(facecolor='orange', label='Jedzenie (0.75)'),
        Patch(facecolor='yellow', label='Głowa (1.0)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    viewport_path = os.path.join(output_dir, f'viewport_state_{state_idx}.png')
    plt.savefig(viewport_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_activation_overview(detailed_activations, action_probs_list, action_names, output_dirs):
    """Generuje wykresy przeglądu aktywacji - używa RMS zamiast mean"""
    
    # Wykres 1: Przegląd aktywacji neuronów (RMS = siła sygnału)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    states = [0, 1, 2]
    
    # 📊 SUBPLOT 1: RMS (Root Mean Square)
    attended_rms = [d['attended_rms'] for d in detailed_activations]
    scalar_rms = [d['scalar_rms'] for d in detailed_activations]
    features_rms = [d['features_rms'] for d in detailed_activations]
    
    x = np.arange(len(states))
    width = 0.25
    
    axes[0, 0].bar(x - width, attended_rms, width, label='Attended CNN RMS', color='#e74c3c')
    axes[0, 0].bar(x, scalar_rms, width, label='Scalar RMS', color='#3498db')
    axes[0, 0].bar(x + width, features_rms, width, label='Fused RMS', color='#2ecc71')
    axes[0, 0].set_xlabel('Stan')
    axes[0, 0].set_ylabel('RMS Magnitude')
    axes[0, 0].set_title('🔥 Siła Sygnału (RMS) - Attended CNN vs Scalars')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Stan {s}' for s in states])
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 📊 SUBPLOT 2: Max Absolute Value
    attended_max = [d['attended_absmax'] for d in detailed_activations]
    scalar_max = [d['scalar_absmax'] for d in detailed_activations]
    
    axes[0, 1].bar(x - width/2, attended_max, width, label='Attended Max', color='#e74c3c')
    axes[0, 1].bar(x + width/2, scalar_max, width, label='Scalar Max', color='#3498db')
    axes[0, 1].set_xlabel('Stan')
    axes[0, 1].set_ylabel('Max |Activation|')
    axes[0, 1].set_title('⚡ Peak Responses')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Stan {s}' for s in states])
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 📊 SUBPLOT 3: Active Neuron Ratio
    attended_active = [d['attended_active_ratio'] * 100 for d in detailed_activations]
    scalar_active = [d['scalar_active_ratio'] * 100 for d in detailed_activations]
    
    axes[1, 0].bar(x - width/2, attended_active, width, label='Attended Active %', color='#e74c3c')
    axes[1, 0].bar(x + width/2, scalar_active, width, label='Scalar Active %', color='#3498db')
    axes[1, 0].set_xlabel('Stan')
    axes[1, 0].set_ylabel('Active Neurons (%)')
    axes[1, 0].set_title('🎯 Sparsity (% neurons |x| > 0.01)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'Stan {s}' for s in states])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim(0, 100)
    
    # 📊 SUBPLOT 4: Action Probabilities
    action_probs = [a['probs'] for a in action_probs_list]
    action_probs_np = np.array(action_probs)
    
    for action_idx in range(len(action_names)):
        axes[1, 1].plot(states, action_probs_np[:, action_idx], marker='o', label=action_names[action_idx], linewidth=2)
    
    axes[1, 1].set_xlabel('Stan')
    axes[1, 1].set_ylabel('Prawdopodobieństwo')
    axes[1, 1].set_title('Prawdopodobieństwa Akcji')
    axes[1, 1].set_xticks(states)
    axes[1, 1].set_xticklabels([f'Stan {s}' for s in states])
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dirs['main'], 'neuron_activations_overview.png')
    plt.savefig(overview_path, dpi=150)
    plt.close()
    print(f'\n✅ Przegląd aktywacji zapisany: {overview_path}')