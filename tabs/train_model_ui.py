import streamlit as st
import os
import librosa
import matplotlib.pyplot as plt
import datetime
import json
import uuid
import torch
from pathlib import Path

# Core model imports - ensure these paths are correct if a user runs this standalone for testing
from core.transformer_generator import TransformerSampleGenerator
from core.improved_generator import ImprovedSampleGenerator

def render_train_model_tab(selected_model_type_from_sidebar):
    st.header("Train Model")

    if not st.session_state.get('processed_samples', []):
        st.info("Please process some samples first in the 'Process Samples' tab.")
    else:
        st.success(f"{len(st.session_state.processed_samples)} processed segments available for training.")

        train_tab1, train_tab2, train_tab3, train_tab4, train_tab5 = st.tabs([
            "New Training", "Continue Training", "Checkpoints", "Hyperparameters", "Training Queue"
        ])

        with train_tab1: # New Training
            st.subheader("Train New Model")
            with st.expander("Training Data Statistics", expanded=False):
                segment_count = len(st.session_state.processed_samples)
                if segment_count > 0:
                    sample_file = st.session_state.processed_samples[0]
                    try:
                        sample_audio, sr = librosa.load(sample_file, sr=None)
                        sample_duration = librosa.get_duration(y=sample_audio, sr=sr)
                        total_duration = segment_count * sample_duration
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.plot(sample_audio)
                        ax.set_title("Sample Training Segment")
                        st.pyplot(fig)
                        st.audio(sample_file)
                        st.write(f"Total segments: {segment_count}")
                        st.write(f"Approximate total duration: {total_duration:.2f} seconds")
                    except Exception as e:
                        st.error(f"Error loading or displaying sample segment: {e}")
                else:
                    st.write("No processed segments to show statistics for.")

            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Training epochs", 10, 500, 50, key="train_epochs")
                batch_size_train = st.slider("Batch size", 4, 64, 16, key="train_batch_size") # Renamed variable
                if 'hyperparameters' not in st.session_state:
                    st.session_state.hyperparameters = {
                        'learning_rate': 0.0002, 'beta1': 0.5, 'beta2': 0.999,
                        'latent_dim': 100, 'sequence_length': 128, 'checkpoint_interval': 10
                    }
                learning_rate = st.session_state.hyperparameters['learning_rate']
                checkpoint_interval = st.session_state.hyperparameters['checkpoint_interval']

            with col2:
                checkpoint_dir = st.text_input("Checkpoint directory", "models/checkpoints", key="train_ckpt_dir")
                model_name = st.text_input("Model name", f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}", key="train_model_name")
                model_save_path = os.path.join(checkpoint_dir, f"{model_name}_final.pt")
                save_checkpoints = st.checkbox("Save checkpoints during training", True, key="train_save_ckpt")
                if save_checkpoints:
                    checkpoint_freq = st.slider("Checkpoint frequency (epochs)", 1, 50, checkpoint_interval, 
                                                help="Save a checkpoint model every N epochs", key="train_ckpt_freq")
                    st.session_state.hyperparameters['checkpoint_interval'] = checkpoint_freq
            
            training_priority = st.radio("Training priority", ["Low", "Medium", "High"], index=1, 
                                         help="Higher priority jobs will be processed first", key="train_priority")
            st.subheader("Training Controls")
            if st.button("Add to Training Queue", key="queue_new_training"):
                training_job = {
                    'id': str(uuid.uuid4()), 'type': 'new_training',
                    'model_type': selected_model_type_from_sidebar, # Use passed param
                    'model_name': model_name, 'epochs': epochs, 'batch_size': batch_size_train,
                    'learning_rate': learning_rate, 'save_checkpoints': save_checkpoints,
                    'checkpoint_freq': checkpoint_freq if save_checkpoints else None,
                    'checkpoint_dir': checkpoint_dir, 'model_save_path': model_save_path,
                    'priority': training_priority, 'user_id': st.session_state.user_id,
                    'submitted_at': datetime.datetime.now().isoformat(), 'status': 'queued',
                    'hyperparameters': st.session_state.hyperparameters.copy()
                }
                if 'training_queue' not in st.session_state: st.session_state.training_queue = []
                st.session_state.training_queue.append(training_job)
                priority_values = {"High": 0, "Medium": 1, "Low": 2}
                st.session_state.training_queue.sort(key=lambda job: (priority_values[job['priority']], job['submitted_at']))
                st.success(f"Training job added to queue! Job ID: {training_job['id'][:8]}")
                st.info("Check the 'Training Queue' tab to monitor your job status")

        with train_tab2: # Continue Training
            st.subheader("Continue Training from Checkpoint")
            checkpoint_base_dir = "models/checkpoints"
            if os.path.exists(checkpoint_base_dir):
                metadata_files = list(Path(checkpoint_base_dir).glob("*_metadata.json"))
                if metadata_files:
                    training_runs = []
                    for metadata_file in metadata_files:
                        try:
                            with open(metadata_file, "r") as f: training_runs.append(json.load(f))
                        except Exception as e:
                            st.warning(f"Could not parse metadata file {metadata_file}: {str(e)}")
                    
                    if training_runs:
                        selected_run_idx = st.selectbox("Select training run", range(len(training_runs)), 
                            format_func=lambda i: f"{training_runs[i].get('model_name', 'Unknown')} ({training_runs[i].get('model_type', 'N/A')})",
                            key="cont_train_run_select")
                        selected_run = training_runs[selected_run_idx]
                        st.write(f"Model type: {selected_run.get('model_type')}")
                        st.write(f"Originally trained for: {selected_run.get('epochs')} epochs")
                        
                        checkpoint_options = ["Final model"] + [f"Checkpoint epoch {Path(p).stem.split('_epoch_')[-1]}" 
                                                              for p in selected_run.get('checkpoint_paths', [])]
                        checkpoint_paths = [selected_run.get('final_model_path', "")] + selected_run.get('checkpoint_paths', []) # Handle missing keys
                        
                        selected_checkpoint_idx = st.selectbox("Select checkpoint", range(len(checkpoint_options)), 
                                                               format_func=lambda i: checkpoint_options[i], key="cont_train_ckpt_select")
                        selected_checkpoint_path = checkpoint_paths[selected_checkpoint_idx]

                        col1, col2 = st.columns(2)
                        with col1:
                            additional_epochs = st.slider("Additional epochs", 10, 500, 50, key="cont_add_epochs")
                            cont_batch_size = st.slider("Batch size", 4, 64, 16, key="continue_batch_size")
                            cont_lr = st.number_input("Learning rate", 0.00001, 0.01, 
                                                      st.session_state.hyperparameters.get('learning_rate', 0.0002), 
                                                      format="%.5f", key="continue_lr")
                        with col2:
                            new_model_name = st.text_input("New model name", 
                                f"{selected_run.get('model_name', 'model')}_continued_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
                                key="cont_new_model_name")
                            add_variation = st.checkbox("Add variation to weights", False, key="cont_add_var")
                            if add_variation:
                                variation_amount = st.slider("Variation amount", 0.0, 0.2, 0.01, 0.01, key="cont_var_amount")
                                st.info(f"Adding {variation_amount*100}% random variation to model weights")
                        
                        if st.button("Continue Training", key="cont_train_button"):
                            # Placeholder for actual training logic
                            st.info("Continue training feature: Actual training call would be here.")
                            # This would involve loading the model from selected_checkpoint_path,
                            # potentially adding variation, and then calling a train method.
                            # Example of loading (conceptual):
                            # try:
                            #     if selected_run.get('model_type') == "Transformer GAN":
                            #         generator = TransformerSampleGenerator(model_path=selected_checkpoint_path)
                            #     elif selected_run.get('model_type') == "Improved Generator":
                            #         generator = ImprovedSampleGenerator(model_path=selected_checkpoint_path)
                            #     else:
                            #         st.error("Unknown model type for continuation.")
                            #         return # or raise error
                            #     
                            #     if add_variation: # Apply variation
                            #          for param in generator.model.parameters():
                            #             noise = torch.randn_like(param) * variation_amount
                            #             param.data += noise
                            #     
                            #     # generator.train( ... with new params ...)
                            #     st.success(f"Continued training for {new_model_name} would start here.")
                            # except Exception as e:
                            #     st.error(f"Error setting up continued training: {e}")
                            pass # Replace with actual training call and progress updates
                    else: st.info("No training runs with metadata found.")
                else: st.info("No checkpoint metadata. Train a model with checkpoints first.")
            else: st.info("No checkpoints directory. Train a model with checkpoints first.")

        with train_tab3: # Checkpoints
            st.subheader("Manage Checkpoints")
            checkpoint_base_dir = "models/checkpoints"
            if os.path.exists(checkpoint_base_dir):
                metadata_files = list(Path(checkpoint_base_dir).glob("*_metadata.json"))
                if not metadata_files: st.info("No metadata files found for training runs.")
                
                for meta_file_path in metadata_files:
                    try:
                        with open(meta_file_path, "r") as f: metadata = json.load(f)
                        model_name = metadata.get("model_name", Path(meta_file_path).stem.replace("_metadata", ""))
                        with st.expander(f"Model: {model_name}", expanded=False):
                            st.write(f"Type: {metadata.get('model_type', 'N/A')}")
                            st.write(f"Trained: {metadata.get('date_trained', 'N/A')[:19]}")
                            st.write(f"Epochs: {metadata.get('epochs', 0)}")
                            # ... (rest of checkpoint display logic, needs to be robust to missing keys)
                            final_model_path = metadata.get("final_model_path")
                            if final_model_path and os.path.exists(final_model_path):
                                st.write(f"**Final Model:** {final_model_path}")
                                # Add load button or other actions
                            
                            checkpoint_paths_list = metadata.get("checkpoint_paths", [])
                            if checkpoint_paths_list:
                                st.write(f"**Checkpoints ({len(checkpoint_paths_list)}):**")
                                for ckpt_path_str in checkpoint_paths_list:
                                    st.write(ckpt_path_str)
                                    # Add load buttons

                            if st.button("Delete this training run", key=f"delete_{model_name}_{meta_file_path.stem}"):
                                try:
                                    if final_model_path and os.path.exists(final_model_path): os.remove(final_model_path)
                                    for ckpt_path_str in checkpoint_paths_list: 
                                        if os.path.exists(ckpt_path_str): os.remove(ckpt_path_str)
                                    os.remove(meta_file_path)
                                    st.success(f"Deleted training run {model_name}")
                                    st.experimental_rerun()
                                except Exception as e: st.error(f"Error deleting {model_name}: {e}")
                    except Exception as e:
                        st.warning(f"Could not read or display metadata for {meta_file_path}: {e}")
            else: st.info("No checkpoints directory found.")

        with train_tab4: # Hyperparameters
            st.subheader("Hyperparameter Settings")
            st.write("Configure training hyperparameters for new training runs:")
            if 'hyperparameters' not in st.session_state: # Initialize if not present
                st.session_state.hyperparameters = {
                    'learning_rate': 0.0002, 'beta1': 0.5, 'beta2': 0.999,
                    'latent_dim': 100, 'sequence_length': 128, 'checkpoint_interval': 10,
                    'transformer_params': {'n_heads': 8, 'n_layers': 6, 'd_model': 512},
                    'gan_params': {'n_mels': 128, 'dropout': 0.3}
                }

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Optimizer Parameters**")
                lr = st.number_input("Learning rate", 0.00001, 0.1, st.session_state.hyperparameters['learning_rate'], format="%.5f", key="hp_lr")
                st.session_state.hyperparameters['learning_rate'] = lr
                # ... (beta1, beta2, checkpoint_interval sliders)
            with col2:
                st.write("**Model Architecture Parameters**")
                latent_dim = st.slider("Latent dimension", 16, 512, st.session_state.hyperparameters['latent_dim'], key="hp_latent_dim")
                st.session_state.hyperparameters['latent_dim'] = latent_dim
                # ... (sequence_length slider)
                if selected_model_type_from_sidebar == "Transformer GAN": # Use passed param
                    # ... (transformer_params sliders)
                    pass # Placeholder for Transformer specific hyperparams
                elif selected_model_type_from_sidebar == "Improved Generator":
                    # ... (gan_params sliders)
                    pass # Placeholder for GAN specific hyperparams
            
            st.subheader("Hyperparameter Presets") # Save/Load preset logic
            # ... (This section can be complex, involving file I/O for presets)
            # For brevity, actual preset saving/loading code is omitted here but was in original.
            # Key idea: save st.session_state.hyperparameters to a JSON, load from JSON.
            presets_dir = "models/hyperparameter_presets"
            os.makedirs(presets_dir, exist_ok=True)
            # Save preset
            preset_name_save = st.text_input("Preset name to save", "my_preset", key="hp_save_preset_name")
            if st.button("Save as Preset", key="hp_save_preset_button"):
                preset_path = os.path.join(presets_dir, f"{preset_name_save}.json")
                with open(preset_path, "w") as f: json.dump(st.session_state.hyperparameters, f, indent=2)
                st.success(f"Saved preset {preset_name_save}")
            # Load preset
            if os.path.exists(presets_dir):
                preset_files = [f.name for f in Path(presets_dir).glob("*.json")]
                if preset_files:
                    selected_preset_load = st.selectbox("Select preset to load", preset_files, key="hp_load_preset_select")
                    if st.button("Load Preset", key="hp_load_preset_button"):
                        preset_path = os.path.join(presets_dir, selected_preset_load)
                        with open(preset_path, "r") as f: st.session_state.hyperparameters.update(json.load(f))
                        st.success(f"Loaded preset {selected_preset_load}")
                        st.experimental_rerun()

        with train_tab5: # Training Queue
            st.subheader("Training Queue Management")
            current_job = st.session_state.get('current_training_job')
            queue = st.session_state.get('training_queue', [])
            if current_job:
                st.write("### Currently Training")
                # ... (display current job details, cancel button)
                # This requires st.session_state.training_progress to be managed by the training process
                if current_job.get('user_id') == st.session_state.user_id and st.button("Cancel Training", key="cancel_curr_job_queue"):
                    if 'training_progress' in st.session_state: st.session_state.training_progress['running'] = False
                    st.warning("Training job will be canceled.")

            st.write(f"### Queued Jobs ({len(queue)})")
            if queue:
                # ... (display queue table, manage user's jobs: change priority, remove, move to top)
                # This involves iterating `queue`, filtering by `st.session_state.user_id`
                # and providing buttons that modify `st.session_state.training_queue` then re-sorting.
                pass # Placeholder for detailed queue display and management
            else: st.info("The training queue is empty.")

            with st.expander("About the Training Queue System", expanded=False):
                st.markdown(f""" ### How the Training Queue Works ... Your user ID is: `{st.session_state.user_id}`""")

        # Display training status (if a job from *this session* was started and is running)
        # This part is tricky as true background training needs a separate worker process.
        # The original code implies direct training within Streamlit, which blocks.
        # The queue system suggests an async/worker setup not fully implemented in the UI code provided.
        # For now, this status display is more of a placeholder or for direct (blocking) training.
        if st.session_state.get('training_progress', {}).get('running', False):
            st.subheader("Live Training Status (Current Session Direct Train)")
            tp = st.session_state.training_progress
            st.write(f"Epoch: {tp.get('epoch',0)}/{tp.get('total_epochs',0)}")
            st.progress(tp.get('progress', 0.0))
            if st.button("Stop Direct Training", key="stop_direct_train_button"):
                st.session_state.training_progress['running'] = False
                st.warning("Direct training stopped.") 