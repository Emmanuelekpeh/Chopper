import streamlit as st
import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def render_manual_chopper_tab():
    st.header("Manual Audio Chopper")

    st.write("Upload an audio file to chop it manually:")
    chopper_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="chopper_uploader")

    if chopper_file:
        os.makedirs("data/chopper", exist_ok=True)
        temp_audio_path = os.path.join("data/chopper", chopper_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(chopper_file.getbuffer())
        
        try:
            waveform, sr = librosa.load(temp_audio_path, sr=None)
            duration = librosa.get_duration(y=waveform, sr=sr)
            st.write(f"File loaded: {chopper_file.name} (Duration: {duration:.2f} seconds)")
            st.audio(temp_audio_path)

            fig, ax = plt.subplots(figsize=(12, 3))
            times = np.linspace(0, duration, len(waveform))
            ax.plot(times, waveform)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Waveform")
            st.pyplot(fig)

            st.subheader("Chopping Controls")
            chop_method = st.radio("Chopping Method", ["Time-based", "Marker-based"], key="manual_chop_method")

            if chop_method == "Time-based":
                segment_length = st.slider("Segment length (seconds)", 0.1, 5.0, 1.0, key="manual_chop_seg_len")
                overlap = st.slider("Overlap (%)", 0, 75, 0, key="manual_chop_overlap")
                overlap_fraction = overlap / 100
                hop_length_seconds = segment_length * (1 - overlap_fraction)
                if hop_length_seconds <= 0: # Prevent division by zero or infinite loops
                    st.error("Segment length and overlap result in non-positive hop length. Adjust parameters.")
                    return                
                num_segments = int((duration - segment_length) / hop_length_seconds) + 1 if duration >= segment_length else 0
                st.write(f"This will create approximately {num_segments} segments")

                if st.button("Chop Audio (Time-based)", key="manual_chop_time_button"):
                    if num_segments > 0:
                        with st.spinner("Chopping audio..."):
                            os.makedirs("output/chopped", exist_ok=True)
                            segment_samples = int(segment_length * sr)
                            hop_length_samples = int(hop_length_seconds * sr)
                            chopped_segments_paths = []
                            for i in range(num_segments):
                                start_sample = i * hop_length_samples
                                end_sample = start_sample + segment_samples
                                if end_sample <= len(waveform):
                                    segment = waveform[start_sample:end_sample]
                                    output_path = f"output/chopped/{os.path.splitext(chopper_file.name)[0]}_segment_{i+1}.wav"
                                    sf.write(output_path, segment, sr)
                                    chopped_segments_paths.append(output_path)
                            st.success(f"Created {len(chopped_segments_paths)} segments")
                            st.subheader("Chopped Segments")
                            for i, segment_path in enumerate(chopped_segments_paths):
                                # UI for displaying segments (as in original code)
                                col_disp1, col_disp2 = st.columns([1,3])
                                with col_disp1: 
                                    st.write(f"Segment {i+1}")
                                    st.audio(segment_path)
                                with col_disp2:
                                    seg_wf, _ = librosa.load(segment_path, sr=None)
                                    fig_seg, ax_seg = plt.subplots(figsize=(8,2))
                                    ax_seg.plot(seg_wf)
                                    st.pyplot(fig_seg)
                    else:
                        st.warning("No segments to create with current settings.")
            else: # Marker-based
                st.write("Place markers (seconds): Enter time values and click 'Add Marker'.")
                if 'markers' not in st.session_state: st.session_state.markers = []
                
                marker_time = st.number_input("Marker time (s)", 0.0, float(duration), step=0.1, key="manual_chop_marker_time")
                if st.button("Add Marker", key="manual_chop_add_marker"):
                    st.session_state.markers.append(marker_time)
                    st.session_state.markers.sort()
                    st.experimental_rerun() # Rerun to update display with new marker

                if st.session_state.markers:
                    st.write("Current markers (s): " + ", ".join([f"{m:.2f}" for m in st.session_state.markers]))
                    if st.button("Clear All Markers", key="manual_chop_clear_markers"): 
                        st.session_state.markers = []
                        st.experimental_rerun()
                    
                    fig_markers, ax_markers = plt.subplots(figsize=(12, 3))
                    ax_markers.plot(times, waveform)
                    for marker in st.session_state.markers: ax_markers.axvline(x=marker, color='r', linestyle='--')
                    st.pyplot(fig_markers)

                    if st.button("Chop at Markers", key="manual_chop_marker_button"):
                        with st.spinner("Chopping at markers..."):
                            os.makedirs("output/chopped", exist_ok=True)
                            boundaries = [0.0] + sorted(list(set(st.session_state.markers))) + [duration]
                            chopped_segments_paths = []
                            for i in range(len(boundaries) - 1):
                                start_time, end_time = boundaries[i], boundaries[i+1]
                                if start_time >= end_time: continue # Skip zero-length or negative segments
                                start_sample, end_sample = int(start_time * sr), int(end_time * sr)
                                segment = waveform[start_sample:end_sample]
                                if len(segment) > 0:
                                    output_path = f"output/chopped/{os.path.splitext(chopper_file.name)[0]}_marker_{i+1}.wav"
                                    sf.write(output_path, segment, sr)
                                    chopped_segments_paths.append(output_path)
                            st.success(f"Created {len(chopped_segments_paths)} segments from markers.")
                            # Display segments (similar to time-based)
                else:
                    st.info("Add markers to enable chopping.")
        except Exception as e:
            st.error(f"Error processing audio file {chopper_file.name}: {e}") 