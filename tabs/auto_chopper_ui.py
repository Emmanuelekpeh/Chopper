import streamlit as st
import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from core.chopping_engine import ChoppingEngine

def render_auto_chopper_tab():
    st.header("Automatic Audio Chopper")

    st.write("Upload an audio file for automatic chopping using beat or silence detection:")
    auto_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="auto_chopper_uploader")

    if auto_file:
        os.makedirs("data/chopper", exist_ok=True)
        auto_audio_path = os.path.join("data/chopper", auto_file.name)
        with open(auto_audio_path, "wb") as f:
            f.write(auto_file.getbuffer())
        
        try:
            waveform, sr = librosa.load(auto_audio_path, sr=None)
            duration = librosa.get_duration(y=waveform, sr=sr)
            st.write(f"File loaded: {auto_file.name} (Duration: {duration:.2f} seconds)")
            st.audio(auto_audio_path)

            fig_auto, ax_auto = plt.subplots(figsize=(12, 3))
            times = np.linspace(0, duration, len(waveform))
            ax_auto.plot(times, waveform)
            ax_auto.set_xlabel("Time (s)")
            ax_auto.set_ylabel("Amplitude")
            ax_auto.set_title("Waveform")
            st.pyplot(fig_auto)

            st.subheader("Chopping Method")
            method = st.radio("Detection Method", ["Beat Detection", "Silence Detection"], key="auto_chop_method")

            if method == "Beat Detection":
                st.write("Beat detection finds rhythmic elements.")
                if st.button("Detect Beats and Chop", key="auto_chop_beat_button"):
                    with st.spinner("Analyzing beats and chopping..."):
                        chopper = ChoppingEngine(sample_rate=sr)
                        beat_times = chopper.chop_by_beats(waveform)
                        os.makedirs("output/chopped", exist_ok=True)
                        segments_data = [] # Store dicts: path, start, end
                        for i in range(len(beat_times) - 1):
                            start_time, end_time = beat_times[i], beat_times[i+1]
                            start_sample, end_sample = int(start_time * sr), int(end_time * sr)
                            if start_sample < len(waveform) and end_sample <= len(waveform) and start_sample < end_sample:
                                segment = waveform[start_sample:end_sample]
                                output_path = f"output/chopped/{os.path.splitext(auto_file.name)[0]}_beat_{i+1}.wav"
                                sf.write(output_path, segment, sr)
                                segments_data.append({"path": output_path, "start": start_time, "end": end_time, "waveform": segment})
                        st.success(f"Detected {len(beat_times)} beats, created {len(segments_data)} segments.")
                        
                        fig_beats, ax_beats = plt.subplots(figsize=(12,3))
                        ax_beats.plot(times, waveform)
                        for beat in beat_times: ax_beats.axvline(x=beat, color='r', linestyle='--')
                        st.pyplot(fig_beats)
                        
                        st.subheader("Chopped Beat Segments")
                        for i, seg_data in enumerate(segments_data):
                            col1, col2, col3 = st.columns([1,2,1])
                            with col1:
                                st.write(f"Beat Segment {i+1}")
                                st.write(f"{seg_data['start']:.2f}s - {seg_data['end']:.2f}s")
                                st.audio(seg_data["path"])
                            with col2:
                                fig_seg_b, ax_seg_b = plt.subplots(figsize=(8,2))
                                ax_seg_b.plot(seg_data["waveform"])
                                st.pyplot(fig_seg_b)
                            with col3:
                                if st.button(f"Save as one-shot {i+1}", key=f"save_beat_{i}_auto"):
                                    # Save one-shot logic
                                    pass # Add save logic here
            else: # Silence Detection
                silence_threshold = st.slider("Silence Threshold (dB)", 10, 60, 30, key="auto_chop_silence_thresh")
                min_silence_ms = st.slider("Min Silence Duration (ms)", 50, 1000, 300, key="auto_chop_min_silence")
                if st.button("Detect Silence and Chop", key="auto_chop_silence_button"):
                    with st.spinner("Analyzing silence and chopping..."):
                        chopper = ChoppingEngine(sample_rate=sr)
                        segment_times = chopper.chop_by_silence(waveform, top_db=silence_threshold)
                        os.makedirs("output/chopped", exist_ok=True)
                        segments_data = []
                        for i, (start_time, end_time) in enumerate(segment_times):
                            if (end_time - start_time) * 1000 >= min_silence_ms:
                                start_sample, end_sample = int(start_time * sr), int(end_time * sr)
                                segment = waveform[start_sample:end_sample]
                                if len(segment) > 0:
                                    output_path = f"output/chopped/{os.path.splitext(auto_file.name)[0]}_silence_{i+1}.wav"
                                    sf.write(output_path, segment, sr)
                                    segments_data.append({"path": output_path, "start": start_time, "end": end_time, "waveform": segment})
                        st.success(f"Found {len(segments_data)} segments after silence removal & duration filter.")
                        
                        fig_silence, ax_silence = plt.subplots(figsize=(12,3))
                        ax_silence.plot(times, waveform)
                        for seg_data in segments_data: ax_silence.axvspan(seg_data["start"], seg_data["end"], alpha=0.2, color='green')
                        st.pyplot(fig_silence)

                        st.subheader("Chopped Segments (Silence Detection)")
                        for i, seg_data in enumerate(segments_data):
                            # Display logic similar to beat detection segments
                            col1, col2, col3 = st.columns([1,2,1])
                            with col1:
                                st.write(f"Segment {i+1}")
                                st.write(f"{seg_data['start']:.2f}s - {seg_data['end']:.2f}s")
                                st.audio(seg_data["path"])
                            with col2:
                                fig_seg_s, ax_seg_s = plt.subplots(figsize=(8,2))
                                ax_seg_s.plot(seg_data["waveform"])
                                st.pyplot(fig_seg_s)
                            with col3:
                                if st.button(f"Save as one-shot {i+1}", key=f"save_silence_{i}_auto"):
                                    # Save one-shot logic
                                    oneshot_dir = "output/oneshots"
                                    os.makedirs(oneshot_dir, exist_ok=True)
                                    oneshot_path = f"{oneshot_dir}/{os.path.splitext(auto_file.name)[0]}_silence_oneshot_{i+1}.wav"
                                    sf.write(oneshot_path, seg_data["waveform"], sr)
                                    st.success(f"Saved one-shot to {oneshot_path}")
        except Exception as e:
            st.error(f"Error processing audio file {auto_file.name}: {e}") 