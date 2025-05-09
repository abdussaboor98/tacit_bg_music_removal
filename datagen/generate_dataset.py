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
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples (music files) to generate")
    
    # Split control flags
    parser.add_argument("--skip_train", action="store_true", help="Skip generating training set")
    parser.add_argument("--skip_val", action="store_true", help="Skip generating validation set")
    parser.add_argument("--skip_test", action="store_true", help="Skip generating test set")
    
    # Split ratios
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of data for testing")
    
    return parser.parse_args()

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

def generate_samples(music_files, speech_files, noise_files, output_path, 
                    max_samples=None, speech_noise_snr=5.0, 
                    music_mix_snr_range=(-5.0, 5.0), 
                    sr=16000, random_seed=42):
    """Generate mixed audio samples from music, speech, and noise files, processing full music files."""
    
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
    pbar.set_description(f"Processing {pbar_unit} for {os.path.basename(output_path)}")
    
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
            m_idx_looped = m_idx % len(music_files) # Use a looped index for selection
            current_music_file = music_files[m_idx_looped]
            
            # Skip if we already know this file failed
            if current_music_file in failed_files:
                m_idx += 1 # Advance the main progression index
                attempts += 1
                if m_idx >= len(music_files) and max_samples is None: # Exhausted files
                    # If pbar is tracking music files, update for skipped file
                    pbar.update(1)
                continue
                
            print(f"\nLoading music file: {os.path.basename(current_music_file)}")
            music, _ = load_audio(current_music_file, sr)
            
            if music is None:
                print(f"Failed to load music file {current_music_file}, trying next file")
                failed_files.append(current_music_file)
                m_idx += 1 # Advance the main progression index
                attempts += 1
                if m_idx >= len(music_files) and max_samples is None:
                     # If pbar is tracking music files, update for failed file
                    pbar.update(1)

            else:
                music_file_usage[current_music_file] += 1
                # Successfully loaded, break from attempts loop
                break 
        
        # If we couldn't load any music file after trying, exit this loop.
        if music is None:
            print("Error: Couldn't load any suitable music files after trying all available. Exiting generation for this split.")
            break
        
        music_length = len(music)
        if music_length == 0:
            print(f"Music file {os.path.basename(current_music_file)} is empty. Skipping...")
            m_idx += 1
            if max_samples is None: # pbar tracks music files if no max_samples
                pbar.update(1)
            continue
        
        current_processing_length = music_length
        print(f"Processing music file: {os.path.basename(current_music_file)} ({music_length/sr:.2f}s)")

        # Create stitched speech and noise arrays to match or exceed music length
        speech_audio = np.array([])
        speech_file_markers = []  # Track which file and time each segment came from
        
        noise_audio = np.array([])
        noise_file_markers = []
        
        # Keep track of what files we've used for this segment
        speech_files_used = []
        noise_files_used = []
        
        # Stitch together speech files
        temp_s_idx = s_idx # Use a temporary index for stitching this round
        speech_stitched_count = 0
        while len(speech_audio) < current_processing_length and speech_stitched_count < len(speech_files):
            current_speech = None
            speech_load_attempts = 0
            
            while current_speech is None and speech_load_attempts < len(speech_files):
                current_s_idx_looped = temp_s_idx % len(speech_files)
                current_speech_file = speech_files[current_s_idx_looped]
                
                if current_speech_file in failed_files:
                    temp_s_idx += 1
                    speech_load_attempts +=1
                    continue
                
                current_speech, _ = load_audio(current_speech_file, sr)
                
                if current_speech is None:
                    print(f"Failed to load speech file {current_speech_file}, trying next file")
                    failed_files.append(current_speech_file)
                    temp_s_idx += 1
                    speech_load_attempts += 1
                else:
                    speech_files_used.append(current_speech_file)
                    speech_file_usage[current_speech_file] += 1
                    speech_stitched_count +=1 # Count successful stitches
                    break # Loaded successfully

            if current_speech is None : # Could not load any speech file
                 if not speech_audio.any(): # If no speech audio could be gathered at all
                    print("Error: Couldn't load any speech files for stitching. Skipping this music file.")
                    break # Break from speech stitching loop

            if current_speech is not None:
                speech_file_markers.append({
                    'file': current_speech_file,
                    'start_idx': len(speech_audio),
                    'length': len(current_speech)
                })
                speech_audio = np.concatenate([speech_audio, current_speech])
            temp_s_idx += 1 # Move to next speech file for stitching
        
        s_idx = temp_s_idx % len(speech_files) # Update global s_idx for next music file

        if len(speech_audio) == 0:
            print("Error: No speech audio available after stitching. Skipping this music file.")
            m_idx += 1
            if max_samples is None: pbar.update(1)
            continue
        
        # Stitch together noise files
        temp_n_idx = n_idx # Use a temporary index for stitching this round
        noise_stitched_count = 0
        while len(noise_audio) < current_processing_length and noise_stitched_count < len(noise_files):
            current_noise = None
            noise_load_attempts = 0

            while current_noise is None and noise_load_attempts < len(noise_files):
                current_n_idx_looped = temp_n_idx % len(noise_files)
                current_noise_file = noise_files[current_n_idx_looped]

                if current_noise_file in failed_files:
                    temp_n_idx += 1
                    noise_load_attempts += 1
                    continue
                
                current_noise, _ = load_audio(current_noise_file, sr)
                
                if current_noise is None:
                    print(f"Failed to load noise file {current_noise_file}, trying next file")
                    failed_files.append(current_noise_file)
                    temp_n_idx += 1
                    noise_load_attempts += 1
                else:
                    noise_files_used.append(current_noise_file)
                    noise_file_usage[current_noise_file] += 1
                    noise_stitched_count += 1 # Count successful stitches
                    break # Loaded successfully
            
            if current_noise is None: # Could not load any noise file
                if not noise_audio.any(): # If no noise audio could be gathered at all
                    print("Error: Couldn't load any noise files for stitching. Skipping this music file.")
                    break # Break from noise stitching loop
            
            if current_noise is not None:
                noise_file_markers.append({
                    'file': current_noise_file, 
                    'start_idx': len(noise_audio),
                    'length': len(current_noise)
                })
                noise_audio = np.concatenate([noise_audio, current_noise])
            temp_n_idx += 1 # Move to next noise file for stitching

        n_idx = temp_n_idx % len(noise_files) # Update global n_idx for next music file

        if len(noise_audio) == 0:
            print("Error: No noise audio available after stitching. Skipping this music file.")
            m_idx += 1
            if max_samples is None: pbar.update(1)
            continue
        
        # Prepare segments (full length of current music file)
        music_segment = music  # Music is used as is
        music_padded = False   # Music defines the length, so it's not padded

        # Speech segment processing
        if len(speech_audio) >= current_processing_length:
            speech_segment = speech_audio[:current_processing_length]
            speech_padded = False
        else:
            padding = np.zeros(current_processing_length - len(speech_audio))
            speech_segment = np.concatenate([speech_audio, padding])
            speech_padded = True
            print(f"Padded speech segment by {len(padding)/sr:.2f}s")
            
        # Noise segment processing
        if len(noise_audio) >= current_processing_length:
            noise_segment = noise_audio[:current_processing_length]
            noise_padded = False
        else:
            padding = np.zeros(current_processing_length - len(noise_audio))
            noise_segment = np.concatenate([noise_audio, padding])
            noise_padded = True
            print(f"Padded noise segment by {len(padding)/sr:.2f}s")

        print(f"Processing one sample from: Music: {os.path.basename(current_music_file)}, "
              f"Speech files used: {len(speech_files_used)}, Noise files used: {len(noise_files_used)}")
        
        # Apply SNR scaling
        speech_segment_normalized = normalize_audio(speech_segment, headroom_db=6.0)
        noise_scaled = scale_to_snr(noise_segment, speech_segment_normalized, speech_noise_snr)
        speech_with_noise = speech_segment_normalized + noise_scaled
        
        # Random SNR for music mixing
        music_mix_snr = np.random.uniform(*music_mix_snr_range)
        
        music_scaled = scale_to_snr(music_segment, speech_with_noise, -music_mix_snr)
        final_mix = speech_with_noise + music_scaled
        
        final_mix = normalize_audio(final_mix)
        
        if np.max(np.abs(music_scaled)) > 0.99:
            music_scaled = normalize_audio(music_scaled, headroom_db=1.0)
        
        if np.max(np.abs(noise_scaled)) > 0.99:
            noise_scaled = normalize_audio(noise_scaled, headroom_db=1.0)

        sample_dir = os.path.join(output_path, f"{sample_idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        
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
        
        # Find which original speech and noise files this segment (starting at pos 0 of stitched) came from
        # The 'position' argument for find_original_file_info is 0 because we are considering
        # the segment starting from the beginning of the (potentially truncated/padded) stitched audio.
        speech_file_info = find_original_file_info(speech_file_markers, 0) if speech_file_markers else {'file': 'N/A', 'start_idx': 0, 'file_pos': 0}
        noise_file_info = find_original_file_info(noise_file_markers, 0) if noise_file_markers else {'file': 'N/A', 'start_idx': 0, 'file_pos': 0}
        
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
            "music_start_time": 0.0, # Music segment always starts at 0 of the original music file
            "speech_start_time": speech_file_info['file_pos'] / sr if speech_file_info['file'] != 'N/A' else 0.0,
            "noise_start_time": noise_file_info['file_pos'] / sr if noise_file_info['file'] != 'N/A' else 0.0,
            "music_mix_snr": music_mix_snr,
            "speech_noise_snr": speech_noise_snr,
            "random_seed": random_seed,
            "music_padded": music_padded, # Always False as music defines length
            "speech_padded": speech_padded,
            "noise_padded": noise_padded
        })
        
        sample_idx += 1
        if max_samples is not None:
            pbar.update(1)
        
        # Check if we've hit max_samples (already checked at the start of the m_idx loop)
        # This inner check is mostly redundant if outer check is effective
        if max_samples is not None and sample_idx >= max_samples:
            break 
        
        # Move to the next music file
        m_idx += 1
        if max_samples is None: # pbar tracks music files if no max_samples
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
        print(f"Train: Processing full files.")
    else:
        print("Train: SKIPPED")
    
    if not args.skip_val:
        splits_to_generate.append("validation")
        print(f"Validation: Processing full files.")
    else:
        print("Validation: SKIPPED")
    
    if not args.skip_test:
        splits_to_generate.append("test")
        print(f"Test: Processing full files.")
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
        
        # Generate samples
        samples_generated = generate_samples(
            music_splits[split],
            speech_splits[split],
            noise_splits[split],
            os.path.join(args.output_base_path, split),
            max_samples=split_max_samples,
            speech_noise_snr=args.speech_noise_snr,
            music_mix_snr_range=(-5.0, 5.0),
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