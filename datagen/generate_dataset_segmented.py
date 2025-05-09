#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import glob

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate custom audio dataset for speech-music-noise separation")
    parser.add_argument("--music_path", type=str, required=True, help="Input folder for music")
    parser.add_argument("--speech_path", type=str, required=True, help="Input folder for speech")
    parser.add_argument("--noise_path", type=str, required=True, help="Input folder for noise")
    parser.add_argument("--output_base_path", type=str, required=True, help="Base output folder")
    
    # Global parameters
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio files")
    parser.add_argument("--speech_noise_snr", type=float, default=5.0, help="SNR between speech and noise in dB")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate")
    
    # Split control flags
    parser.add_argument("--skip_train", action="store_true", help="Skip generating training set")
    parser.add_argument("--skip_val", action="store_true", help="Skip generating validation set")
    parser.add_argument("--skip_test", action="store_true", help="Skip generating test set")
    
    # Split ratios
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of data for testing")
    
    # Train set parameters
    parser.add_argument("--train_sample_length_sec", type=float, default=1.0, 
                        help="Length of each training sample in seconds")
    parser.add_argument("--train_time_skip_sec", type=float, default=0.5, 
                        help="Time to skip forward after each training sample")
    
    # Validation set parameters
    parser.add_argument("--val_sample_length_sec", type=float, default=1.5, 
                        help="Length of each validation sample in seconds")
    parser.add_argument("--val_time_skip_sec", type=float, default=0.75, 
                        help="Time to skip forward after each validation sample")
    
    # Test set parameters
    parser.add_argument("--test_sample_length_sec", type=float, default=2.0, 
                        help="Length of each test sample in seconds")
    parser.add_argument("--test_time_skip_sec", type=float, default=1.0, 
                        help="Time to skip forward after each test sample")
    
    # Backward compatibility
    parser.add_argument("--sample_length_sec", type=float, default=None, 
                        help="Length of each sample in seconds (legacy, sets all splits to same value)")
    parser.add_argument("--time_skip_sec", type=float, default=None, 
                        help="Time to skip forward after each sample (legacy, sets all splits to same value)")
    
    return parser.parse_args()

def list_wav_files(directory):
    """Recursively find all .wav files in a directory"""
    return glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)

def list_audio_files(directory):
    """Recursively find all .wav and .mp3 files in a directory"""
    wav_files = glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)
    mp3_files = glob.glob(os.path.join(directory, "**", "*.mp3"), recursive=True)
    return wav_files + mp3_files

def split_dataset(files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """Split files into training, validation, and test sets"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    random.seed(random_seed)
    random.shuffle(files)
    
    n_files = len(files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    return {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

def load_audio(file_path, target_sr=16000):
    """Load audio file and resample to target sample rate"""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None  # Return None to signal loading failure

def extract_segment(audio, start_idx, segment_length, sr):
    """Extract a segment from audio starting at start_idx"""
    end_idx = start_idx + segment_length
    
    # If we don't have enough samples, pad with zeros
    if end_idx > len(audio):
        segment = np.zeros(segment_length)
        segment[:len(audio) - start_idx] = audio[start_idx:]
        return segment, True  # Return a flag indicating padding was needed
    
    return audio[start_idx:end_idx], False  # No padding needed

def scale_to_snr(target, reference, snr_db):
    """Scale target signal to achieve desired SNR with reference signal"""
    target_rms = np.sqrt(np.mean(target**2))
    reference_rms = np.sqrt(np.mean(reference**2))
    
    if target_rms < 1e-10 or reference_rms < 1e-10:
        return target
    
    gain = reference_rms / (target_rms * 10**(snr_db/20))
    return target * gain

def normalize_audio(audio, headroom_db=3.0):
    """Normalize audio to avoid clipping, leaving some headroom"""
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        gain = 10**(-headroom_db/20) / max_abs
        return audio * gain
    return audio

def save_audio(audio, file_path, sr=16000):
    """Save audio to file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sr, subtype='PCM_16')

def concatenate_speech_if_needed(speech_files, s_idx, speech, s_pos, sample_length, sr):
    """Concatenate speech files if the current file isn't long enough"""
    if s_pos + sample_length <= len(speech):
        return speech, s_idx, s_pos, False
    
    # We need to concatenate
    remaining_length = len(speech) - s_pos
    needed_length = sample_length - remaining_length
    
    # Get the first part from the current file
    first_part = speech[s_pos:]
    
    # Load the next file
    next_idx = (s_idx + 1) % len(speech_files)
    next_speech, _ = load_audio(speech_files[next_idx], sr)
    
    # If the next file is too short, we recursively concatenate
    if len(next_speech) < needed_length:
        concatenated, new_idx, _, _ = concatenate_speech_if_needed(
            speech_files, next_idx, next_speech, 0, needed_length, sr
        )
        result = np.concatenate([first_part, concatenated[:needed_length]])
        return result, new_idx, 0, True
    
    # The next file is long enough
    result = np.concatenate([first_part, next_speech[:needed_length]])
    return result, next_idx, needed_length, True

def generate_samples(music_files, speech_files, noise_files, output_path, 
                    sample_length_sec=1.0, time_skip_sec=0.5, max_samples=None, 
                    speech_noise_snr=5.0, music_mix_snr_range=(-5.0, 5.0), 
                    sr=16000, random_seed=42):
    """Generate mixed audio samples from music, speech, and noise files"""
    
    if not music_files or not speech_files or not noise_files:
        print("Error: Empty file list for one of the sources!")
        return 0
    
    print(f"Will use {len(music_files)} music files, {len(speech_files)} speech files, and {len(noise_files)} noise files")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Prepare tracking variables
    metadata = []
    sample_idx = 0
    
    # Track usage statistics
    music_file_usage = {file: 0 for file in music_files}
    speech_file_usage = {file: 0 for file in speech_files}
    noise_file_usage = {file: 0 for file in noise_files}
    
    # Track files that failed to load
    failed_files = []
    
    # Calculate sample length in samples
    sample_length = int(sample_length_sec * sr)
    skip_samples = int(time_skip_sec * sr)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Initialize indices
    m_idx, s_idx, n_idx = 0, 0, 0
    
    # Progress bar - use max_samples if set, otherwise track progress over music files
    if max_samples:
        pbar = tqdm(total=max_samples)
        pbar_unit = "samples"
    else:
        pbar = tqdm(total=len(music_files))
        pbar_unit = "music files"
        pbar.set_description(f"Processing {pbar_unit}")
    
    # Continue until we've processed all music files or reached max_samples
    while m_idx < len(music_files):
        # Check if we've hit max_samples
        if max_samples is not None and sample_idx >= max_samples:
            break
        
        # Load a music file
        current_music_file = None
        music = None
        
        # Try loading music files until one succeeds or we run out of files
        attempts = 0
        max_attempts = len(music_files)  # Don't try more than the number of available files
        
        while music is None and attempts < max_attempts:
            m_idx = m_idx % len(music_files)
            current_music_file = music_files[m_idx]
            
            # Skip if we already know this file failed
            if current_music_file in failed_files:
                m_idx += 1
                attempts += 1
                continue
                
            print(f"\nLoading music file: {os.path.basename(current_music_file)}")
            music, _ = load_audio(current_music_file, sr)
            
            if music is None:
                print(f"Failed to load music file {current_music_file}, trying next file")
                failed_files.append(current_music_file)
                m_idx += 1
                attempts += 1
            else:
                music_file_usage[current_music_file] += 1
        
        # If we couldn't load any music file, exit
        if music is None:
            print("Error: Couldn't load any music files. Exiting.")
            break
        
        # Calculate how many samples we can extract from this music file
        music_length = len(music)
        max_start_idx = music_length - sample_length
        if max_start_idx <= 0:
            print(f"Music file {os.path.basename(current_music_file)} is too short. Skipping...")
            m_idx += 1
            # Update progress bar for music files tracking
            if max_samples is None:
                pbar.update(1)
            continue
        
        num_samples_possible = max_start_idx // skip_samples + 1
        print(f"Music file length: {music_length/sr:.2f}s, can generate ~{num_samples_possible} samples")
        
        # Create stitched speech and noise arrays to match or exceed music length
        speech_audio = np.array([])
        speech_file_markers = []  # Track which file and time each segment came from
        
        noise_audio = np.array([])
        noise_file_markers = []
        
        # Keep track of what files we've used for this segment
        speech_files_used = []
        noise_files_used = []
        
        # Stitch together speech files
        while len(speech_audio) < music_length:
            current_speech = None
            
            # Try speech files until one loads successfully
            speech_attempts = 0
            max_speech_attempts = len(speech_files)
            
            while current_speech is None and speech_attempts < max_speech_attempts:
                s_idx = s_idx % len(speech_files)
                current_speech_file = speech_files[s_idx]
                
                # Skip if we already know this file failed
                if current_speech_file in failed_files:
                    s_idx += 1
                    speech_attempts += 1
                    continue
                
                current_speech, _ = load_audio(current_speech_file, sr)
                
                if current_speech is None:
                    print(f"Failed to load speech file {current_speech_file}, trying next file")
                    failed_files.append(current_speech_file)
                    s_idx += 1
                    speech_attempts += 1
                else:
                    speech_files_used.append(current_speech_file)
                    speech_file_usage[current_speech_file] += 1
            
            # If we couldn't load any speech file, break
            if current_speech is None:
                print("Error: Couldn't load any speech files. Skipping this music file.")
                break
            
            # Record where this file starts in our stitched audio
            speech_file_markers.append({
                'file': current_speech_file,
                'start_idx': len(speech_audio),
                'length': len(current_speech)
            })
            
            speech_audio = np.concatenate([speech_audio, current_speech])
            s_idx += 1
        
        # If we couldn't stitch speech files, skip to next music file
        if len(speech_audio) == 0:
            m_idx += 1
            continue
        
        # Stitch together noise files
        while len(noise_audio) < music_length:
            current_noise = None
            
            # Try noise files until one loads successfully
            noise_attempts = 0
            max_noise_attempts = len(noise_files)
            
            while current_noise is None and noise_attempts < max_noise_attempts:
                n_idx = n_idx % len(noise_files)
                current_noise_file = noise_files[n_idx]
                
                # Skip if we already know this file failed
                if current_noise_file in failed_files:
                    n_idx += 1
                    noise_attempts += 1
                    continue
                
                current_noise, _ = load_audio(current_noise_file, sr)
                
                if current_noise is None:
                    print(f"Failed to load noise file {current_noise_file}, trying next file")
                    failed_files.append(current_noise_file)
                    n_idx += 1
                    noise_attempts += 1
                else:
                    noise_files_used.append(current_noise_file)
                    noise_file_usage[current_noise_file] += 1
            
            # If we couldn't load any noise file, break
            if current_noise is None:
                print("Error: Couldn't load any noise files. Skipping this music file.")
                break
            
            # Record where this file starts in our stitched audio
            noise_file_markers.append({
                'file': current_noise_file, 
                'start_idx': len(noise_audio),
                'length': len(current_noise)
            })
            
            noise_audio = np.concatenate([noise_audio, current_noise])
            n_idx += 1
        
        # If we couldn't stitch noise files, skip to next music file
        if len(noise_audio) == 0:
            m_idx += 1
            continue
        
        # Now generate samples by sequentially sampling from these audio arrays
        num_samples_to_generate = min(
            num_samples_possible,
            (max_samples - sample_idx) if max_samples is not None else float('inf')
        )
        
        print(f"Generating {num_samples_to_generate} samples from current files...")
        print(f"Used {len(speech_files_used)} speech files and {len(noise_files_used)} noise files")
        
        for i in range(num_samples_to_generate):
            # Calculate the start indices
            m_pos = i * skip_samples
            
            # Extract segments
            music_segment = music[m_pos:m_pos + sample_length]
            speech_segment = speech_audio[m_pos:m_pos + sample_length]
            noise_segment = noise_audio[m_pos:m_pos + sample_length]
            
            # Ensure all segments are the correct length (padding if necessary)
            if len(music_segment) < sample_length:
                padding = np.zeros(sample_length - len(music_segment))
                music_segment = np.concatenate([music_segment, padding])
                music_padded = True
            else:
                music_padded = False
                
            if len(speech_segment) < sample_length:
                padding = np.zeros(sample_length - len(speech_segment))
                speech_segment = np.concatenate([speech_segment, padding])
                speech_padded = True
            else:
                speech_padded = False
                
            if len(noise_segment) < sample_length:
                padding = np.zeros(sample_length - len(noise_segment))
                noise_segment = np.concatenate([noise_segment, padding])
                noise_padded = True
            else:
                noise_padded = False
            
            # Apply SNR scaling
            # Keep speech at original volume, scale noise to achieve desired SNR
            speech_segment_normalized = normalize_audio(speech_segment, headroom_db=6.0)
            noise_scaled = scale_to_snr(noise_segment, speech_segment_normalized, speech_noise_snr)
            speech_with_noise = speech_segment_normalized + noise_scaled
            
            # Random SNR for music mixing
            music_mix_snr = np.random.uniform(*music_mix_snr_range)
            
            # Mix speech+noise with music, keeping speech+noise at original volume
            music_scaled = scale_to_snr(music_segment, speech_with_noise, -music_mix_snr)
            final_mix = speech_with_noise + music_scaled
            
            # Normalize final mix to avoid clipping
            final_mix = normalize_audio(final_mix)
            
            # DON'T normalize individual components, keep speech at original volume
            # Only normalize music and noise if needed to avoid clipping
            if np.max(np.abs(music_scaled)) > 0.99:
                music_scaled = normalize_audio(music_scaled, headroom_db=1.0)
            
            if np.max(np.abs(noise_scaled)) > 0.99:
                noise_scaled = normalize_audio(noise_scaled, headroom_db=1.0)

            # Create sample directory with zero-padded ID
            sample_dir = os.path.join(output_path, f"{sample_idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save all audio files
            mix_path = os.path.join(sample_dir, "mix.wav")
            music_path = os.path.join(sample_dir, "music.wav")
            speech_isolated_path = os.path.join(sample_dir, "speech_isolated.wav")
            noise_path = os.path.join(sample_dir, "noise.wav")
            speech_path = os.path.join(sample_dir, "speech.wav")
            
            save_audio(final_mix, mix_path, sr)
            save_audio(music_scaled, music_path, sr)
            save_audio(speech_segment_normalized, speech_isolated_path, sr)
            save_audio(noise_scaled, noise_path, sr)
            save_audio(speech_with_noise, speech_path, sr)
            
            # Find which original speech and noise files this segment came from
            speech_file_info = find_original_file_info(speech_file_markers, m_pos)
            noise_file_info = find_original_file_info(noise_file_markers, m_pos)
            
            # Record metadata with detailed information
            metadata.append({
                "sample_id": f"{sample_idx:05d}",
                "mix_path": mix_path,
                "music_path": music_path,
                "speech_isolated_path": speech_isolated_path,
                "noise_path": noise_path,
                "speech_path": speech_path,
                "original_music_file": current_music_file,
                "original_speech_file": speech_file_info['file'],
                "original_noise_file": noise_file_info['file'],
                "music_start_time": m_pos / sr,
                "speech_start_time": (m_pos - speech_file_info['start_idx'] + speech_file_info['file_pos']) / sr,
                "noise_start_time": (m_pos - noise_file_info['start_idx'] + noise_file_info['file_pos']) / sr,
                "music_mix_snr": music_mix_snr,
                "speech_noise_snr": speech_noise_snr,
                "random_seed": random_seed,
                "music_padded": music_padded,
                "speech_padded": speech_padded,
                "noise_padded": noise_padded
            })
            
            # Update sample index and progress bar
            sample_idx += 1
            # Only update progress bar for samples if max_samples is set
            if max_samples is not None:
                pbar.update(1)
            
            # Check if we've hit max_samples
            if max_samples is not None and sample_idx >= max_samples:
                break
        
        # Move to the next music file
        m_idx += 1
        
        # Update progress bar for music files if max_samples is not set
        if max_samples is None:
            pbar.update(1)
    
    pbar.close()
    
    # Print final usage statistics
    print("\nFinal file usage:")
    print(f"Music: Used {sum(1 for count in music_file_usage.values() if count > 0)} out of {len(music_files)} files")
    print(f"Speech: Used {sum(1 for count in speech_file_usage.values() if count > 0)} out of {len(speech_files)} files")
    print(f"Noise: Used {sum(1 for count in noise_file_usage.values() if count > 0)} out of {len(noise_files)} files")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_path, "metadata.csv"), index=False)
    
    return len(metadata)

def find_original_file_info(file_markers, position):
    """Find which original file a position in the stitched audio comes from"""
    for i, marker in enumerate(file_markers):
        if position >= marker['start_idx'] and position < marker['start_idx'] + marker['length']:
            # Found the file
            file_pos = position - marker['start_idx']
            return {
                'file': marker['file'],
                'start_idx': marker['start_idx'],
                'file_pos': file_pos
            }
        elif i == len(file_markers) - 1:
            # We're in the last file (might be beyond its length due to padding)
            file_pos = position - marker['start_idx']
            if file_pos >= marker['length']:
                file_pos = marker['length'] - 1  # Clamp to the end of the file
            return {
                'file': marker['file'],
                'start_idx': marker['start_idx'],
                'file_pos': file_pos
            }
    
    # If we couldn't find it (shouldn't happen), return the first file
    return {
        'file': file_markers[0]['file'],
        'start_idx': file_markers[0]['start_idx'],
        'file_pos': 0
    }

def main():
    """Main function to generate the dataset"""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Handle legacy options (backward compatibility)
    if args.sample_length_sec is not None:
        print(f"Using legacy sample_length_sec={args.sample_length_sec} for all splits")
        args.train_sample_length_sec = args.sample_length_sec
        args.val_sample_length_sec = args.sample_length_sec
        args.test_sample_length_sec = args.sample_length_sec
    
    if args.time_skip_sec is not None:
        print(f"Using legacy time_skip_sec={args.time_skip_sec} for all splits")
        args.train_time_skip_sec = args.time_skip_sec
        args.val_time_skip_sec = args.time_skip_sec
        args.test_time_skip_sec = args.time_skip_sec
    
    # Check skip flags
    if args.skip_train and args.skip_val and args.skip_test:
        print("Error: Cannot skip all splits. At least one split must be generated.")
        return
    
    print(f"Using split ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"Using sample rate: {args.sample_rate} Hz")
    print("\nSplit configurations:")
    
    # Determine which splits to generate
    splits_to_generate = []
    if not args.skip_train:
        splits_to_generate.append("train")
        print(f"Train: {args.train_sample_length_sec}s segments, {args.train_time_skip_sec}s stride")
    else:
        print("Train: SKIPPED")
    
    if not args.skip_val:
        splits_to_generate.append("validation")
        print(f"Validation: {args.val_sample_length_sec}s segments, {args.val_time_skip_sec}s stride")
    else:
        print("Validation: SKIPPED")
    
    if not args.skip_test:
        splits_to_generate.append("test")
        print(f"Test: {args.test_sample_length_sec}s segments, {args.test_time_skip_sec}s stride")
    else:
        print("Test: SKIPPED")
    
    print("Scanning directories for audio files...")
    print(f"Music path: {args.music_path}")
    print(f"Speech path: {args.speech_path}")
    print(f"Noise path: {args.noise_path}")
    
    music_files = list_audio_files(args.music_path)
    speech_files = list_audio_files(args.speech_path)
    noise_files = list_audio_files(args.noise_path)
    
    # Filter out files that might cause issues
    if not music_files:
        print("Error: No music files found in the specified directory.")
        return
    if not speech_files:
        print("Error: No speech files found in the specified directory.")
        return
    if not noise_files:
        print("Error: No noise files found in the specified directory.")
        return
    
    print(f"Found {len(music_files)} music files, {len(speech_files)} speech files, and {len(noise_files)} noise files")
    
    # Split files into train, validation, and test sets
    music_splits = split_dataset(music_files, 
                                train_ratio=args.train_ratio, 
                                val_ratio=args.val_ratio, 
                                test_ratio=args.test_ratio, 
                                random_seed=args.random_seed)
    
    speech_splits = split_dataset(speech_files, 
                                 train_ratio=args.train_ratio, 
                                 val_ratio=args.val_ratio, 
                                 test_ratio=args.test_ratio, 
                                 random_seed=args.random_seed)
    
    noise_splits = split_dataset(noise_files, 
                                train_ratio=args.train_ratio, 
                                val_ratio=args.val_ratio, 
                                test_ratio=args.test_ratio, 
                                random_seed=args.random_seed)
    
    # Create output directories for selected splits
    for split in splits_to_generate:
        os.makedirs(os.path.join(args.output_base_path, split), exist_ok=True)
    
    # Generate samples for each split
    total_samples = 0
    
    # Configure split-specific parameters
    split_configs = {
        "train": {
            "sample_length_sec": args.train_sample_length_sec,
            "time_skip_sec": args.train_time_skip_sec
        },
        "validation": {
            "sample_length_sec": args.val_sample_length_sec,
            "time_skip_sec": args.val_time_skip_sec
        },
        "test": {
            "sample_length_sec": args.test_sample_length_sec,
            "time_skip_sec": args.test_time_skip_sec
        }
    }
    
    # Process only the selected splits
    for split in splits_to_generate:
        print(f"\nGenerating {split} samples...")
        
        # Calculate max samples per split if max_samples is specified
        split_max_samples = None
        if args.max_samples is not None:
            # Always use the original ratios regardless of which splits are being generated
            if split == "train":
                split_max_samples = int(args.max_samples * args.train_ratio)
            elif split == "validation":
                split_max_samples = int(args.max_samples * args.val_ratio)
            else:  # test
                split_max_samples = int(args.max_samples * args.test_ratio)
        
        # Generate samples for this split with split-specific parameters
        samples_generated = generate_samples(
            music_splits[split],
            speech_splits[split],
            noise_splits[split],
            os.path.join(args.output_base_path, split),
            sample_length_sec=split_configs[split]["sample_length_sec"],
            time_skip_sec=split_configs[split]["time_skip_sec"],
            max_samples=split_max_samples,
            speech_noise_snr=args.speech_noise_snr,
            sr=args.sample_rate,
            random_seed=args.random_seed
        )
        
        total_samples += samples_generated
        print(f"Generated {samples_generated} {split} samples")
    
    print(f"\nDataset generation complete! Generated {total_samples} total samples across {len(splits_to_generate)} splits.")
    
    # Clarify the actual ratio of samples generated if some splits were skipped
    if args.skip_train or args.skip_val or args.skip_test:
        print("\nNote: Some splits were skipped, but the sample ratios for generated splits")
        print("were maintained according to the specified train/val/test ratios.")

if __name__ == "__main__":
    main() 