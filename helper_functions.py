import os
import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def save_small_filter(filtered_df, h5_path, save_path, save_npy=True):
    """
    Efficient for small filtered subsets (up to ~20% of original data).
    Saves filtered data to:
    - save_path/train.csv
    - save_path/spectrograms.h5
    """
    os.makedirs(save_path, exist_ok=True)
    filtered_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    
    original_indices = filtered_df.index.to_numpy()
    h5_save_path = os.path.join(save_path, 'spectrograms.h5')

    with h5py.File(h5_path, 'r') as src_h5, h5py.File(h5_save_path, 'w') as dst_h5:
        
        src_dset = src_h5['spectrograms']

        if save_npy:
            spectrograms_dict = {}

            if len(filtered_df) > len(src_dset):
                raise ValueError("Filtered DF has more rows than HDF5 dataset!")
            
            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Building spectrogram dict"):
                try:
                    spectrograms_dict[row['samplename']] = src_dset[idx]
                except KeyError:
                    print(f"Warning: Index {idx} or samplename {row['samplename']} missing!")
                    continue

            np.save(os.path.join(save_path, 'spectrograms.npy'), spectrograms_dict)

        else:
            chunks = src_dset.chunks or (1024, *src_dset.shape[1:])
            chunks = (min(chunks[0], len(filtered_df)), *chunks[1:])

            dst_dset = dst_h5.create_dataset(
                'spectrograms',
                shape=(len(filtered_df), *src_dset.shape[1:]),
                dtype=src_dset.dtype,
                chunks=chunks,  # Preserve chunks
                compression=src_dset.compression,
                compression_opts=src_dset.compression_opts
            )
            
            # Process in memory-efficient batches
            batch_size = min(10000, len(original_indices))
            for i in tqdm(range(0, len(original_indices), batch_size)):
                batch_indices = original_indices[i:i+batch_size]
                dst_dset[i:i+len(batch_indices)] = src_dset[batch_indices]

    # return os.path.join(save_path, 'train.csv'), h5_save_path

def save_large_filter(filtered_df, h5_path, save_path, save_npy=False):
    """
    Optimized for large filtered subsets (more than 20% of original data).
    Uses memory-mapping for better performance with large data.
    Saves filtered data to:
    - save_path/train.csv
    - save_path/spectrograms.h5
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save filtered metadata
    filtered_df.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    
    # Create memory-mapped output file
    with h5py.File(h5_path, 'r') as src_h5:
        src_dset = src_h5['spectrograms']

        if save_npy:
            spectrograms_dict = {}

            if len(filtered_df) > len(src_dset):
                raise ValueError("Filtered DF has more rows than HDF5 dataset!")
            
            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Building spectrogram dict"):
                try:
                    spectrograms_dict[row['samplename']] = src_dset[idx]
                except KeyError:
                    print(f"Warning: Index {idx} or samplename {row['samplename']} missing!")
                    continue

            np.save(os.path.join(save_path, 'spectrograms.npy'), spectrograms_dict)

        else:
            with h5py.File(os.path.join(save_path, 'spectrograms.h5'), 'w') as dst_h5:
                # Initialize resizable dataset
                chunks = src_dset.chunks or (1024, *src_dset.shape[1:])
                chunks = (min(chunks[0], len(filtered_df)), *chunks[1:])

                dst_dset = dst_h5.create_dataset(
                    'spectrograms',
                    shape=(0, *src_dset.shape[1:]),
                    maxshape=(None, *src_dset.shape[1:]),
                    dtype=src_dset.dtype,
                    chunks=chunks,
                    compression=src_dset.compression,
                    compression_opts=src_dset.compression_opts
                )
                
                # Process sequentially to minimize memory
                for idx in tqdm(filtered_df.index):
                    dst_dset.resize(dst_dset.shape[0] + 1, axis=0)
                    dst_dset[-1] = src_dset[idx]

    # return os.path.join(save_path, 'train.csv'), os.path.join(save_path, 'spectrograms.h5')

def save_filtered_data(original_df, filtered_df, name, 
                       h5_path = 'processed_data/expanded_spectrograms.h5',
                       base_filepath='processed_data/filtered_pairs', 
                       threshold=0.2, save_npy=True):
    """
    Automatically selects the best saving strategy based on filter ratio.
    Args:
        threshold: Ratio threshold to switch between methods (0.2 = 20%)
    """
    original_size = len(original_df)
    filter_ratio = len(filtered_df) / original_size if original_size else 1.0
    
    save_path = os.path.join(base_filepath, name)

    if filter_ratio <= threshold:
        print('Using small filter')
        save_small_filter(filtered_df, h5_path, save_path, save_npy=save_npy)
    else:
        print('Using large filter')
        save_large_filter(filtered_df, h5_path, save_path, save_npy=save_npy)

    if not save_npy:
        with h5py.File(os.path.join(save_path, 'spectrograms.h5'), 'r') as hf:
            print(f"Filtered HDF5 contains {len(hf['spectrograms'])} spectrograms")
                
            loaded_df = pd.read_csv(os.path.join(save_path, 'train.csv'))
            print(f"Filtered CSV contains {len(loaded_df)} rows")
            assert len(loaded_df) == len(hf['spectrograms'])

    print('No problemo ;)')



def plot_spectrogram(index, h5_path='processed_data/expanded_spectrograms.h5'):
    with h5py.File(h5_path, 'r') as src_h5:
        src_dset = src_h5['spectrograms']
        spec = src_dset[index]

        plt.imshow(spec, interpolation='none')
        plt.show()
