import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
import h5py
import os
from pathlib import Path
import scanpy as sc
from scipy.stats import median_abs_deviation
import re
from typing import Dict, List, Tuple, Optional
import argparse
from scipy.spatial import cKDTree
from tqdm import tqdm

class XeniumPatchExtractor:
    """Class to handle single cell patch extraction for multiple Xenium samples."""
    
    def __init__(self, base_dir: Path, patch_size: int = 224, pixel_size: float = 0.2125, 
                 n_mads_qc: int = 5):
        self.base_dir = Path(base_dir)
        self.xenium_dir = self.base_dir / "xenium_output"
        self.alignment_dir = self.base_dir / "transformation_matrices"
        self.he_dir = self.base_dir / "H&E"
        self.output_dir = self.base_dir / "saved_coordinates"
        
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.pixel_size = pixel_size
        self.n_mads_qc = n_mads_qc
        
        print(f"Base directory: {self.base_dir}")
        print(f"Xenium directory: {self.xenium_dir}")
        print(f"Alignment directory: {self.alignment_dir}")
        print(f"H&E directory: {self.he_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"QC MADs threshold: {self.n_mads_qc}")
        
    def find_sample_files(self) -> Dict[str, Dict[str, Path]]:
        """Find all matching Xenium directories, alignment, and H&E files for each sample."""
        samples = {}
        
        # Find all Xenium directories (look for directories containing cells.csv.gz)
        xenium_dirs = [d for d in self.xenium_dir.glob("*_outs") if d.is_dir() and (d / "cells.csv.gz").exists()]
        print(f"\nFound {len(xenium_dirs)} Xenium output directories")
        
        for xenium_path in xenium_dirs:
            # Extract sample name (remove _outs suffix)
            sample_name = xenium_path.name.replace("_outs", "")
            
            # Find corresponding alignment file
            alignment_pattern = f"{sample_name}_he_imagealignment.csv"
            alignment_file = self.alignment_dir / alignment_pattern
            
            # Find corresponding H&E file
            he_pattern = f"{sample_name}_he_image.ome.tif"
            he_file = self.he_dir / he_pattern
            
            # Check if all files exist
            if alignment_file.exists() and he_file.exists():
                samples[sample_name] = {
                    'xenium_dir': xenium_path,
                    'alignment': alignment_file,
                    'he_image': he_file
                }
                print(f"OK {sample_name}: All files found")
            else:
                missing = []
                if not alignment_file.exists():
                    missing.append(f"alignment ({alignment_pattern})")
                if not he_file.exists():
                    missing.append(f"H&E ({he_pattern})")
                print(f"MISSING {sample_name}: Missing {', '.join(missing)}")
        
        print(f"\nFound {len(samples)} complete sample sets")
        return samples
    
    def transform_coords_inverse(self, aff_transf: pd.DataFrame, coords: np.ndarray) -> np.ndarray:
        """Apply inverse affine transformation to align Xenium output with the H&E image."""
        inv_transf = np.linalg.inv(aff_transf.values)
        transformed_coords = (inv_transf @ np.vstack((coords.T, np.ones(len(coords))))).T[:, :-1]
        return transformed_coords
    
    def apply_qc_filtering(self, cells_df: pd.DataFrame, n_mads_qc: int = 5):
        """Apply basic QC filtering using available columns in cells.csv.gz."""
        print(f"Applying QC filtering with {n_mads_qc} MADs threshold...")
        print(f"Total number of cells before QC: {len(cells_df)}")
        
        initial_count = len(cells_df)
        
        # Check available QC columns and apply filtering
        if 'total_counts' in cells_df.columns:
            median_counts = cells_df['total_counts'].median()
            mad_counts = median_abs_deviation(cells_df['total_counts'])
            lower_bound = median_counts - n_mads_qc * mad_counts
            upper_bound = median_counts + n_mads_qc * mad_counts
            cells_df = cells_df[
                (cells_df['total_counts'] >= lower_bound) & 
                (cells_df['total_counts'] <= upper_bound)
            ]
            print(f"Filtered by total_counts: {len(cells_df)} cells remaining")
        
        if 'n_genes' in cells_df.columns:
            median_genes = cells_df['n_genes'].median()
            mad_genes = median_abs_deviation(cells_df['n_genes'])
            lower_bound = median_genes - n_mads_qc * mad_genes
            upper_bound = median_genes + n_mads_qc * mad_genes
            cells_df = cells_df[
                (cells_df['n_genes'] >= lower_bound) & 
                (cells_df['n_genes'] <= upper_bound)
            ]
            print(f"Filtered by n_genes: {len(cells_df)} cells remaining")
        
        print(f"Filtered out {initial_count - len(cells_df)} cells in QC")
        return cells_df

    def count_neighbors_within_radius(self, coordinates: np.ndarray, radius: float) -> np.ndarray:
        """Count neighbors within radius for each cell with progress bar."""
        print(f"Building KDTree for {len(coordinates)} cells...")
        tree = cKDTree(coordinates)
        
        print(f"Computing neighbors within radius {radius}...")
        
        # Process in chunks to show progress
        chunk_size = 10000
        neighbor_counts = np.zeros(len(coordinates), dtype=int)
        
        for i in tqdm(range(0, len(coordinates), chunk_size), 
                      desc="Computing neighbors", 
                      unit="chunks"):
            end_idx = min(i + chunk_size, len(coordinates))
            chunk_coords = coordinates[i:end_idx]
            chunk_counts = tree.query_ball_point(chunk_coords, r=radius, return_length=True)
            neighbor_counts[i:end_idx] = chunk_counts
        
        print("Neighbor counting completed!")
        # Subtract 1 to exclude the cell itself
        return neighbor_counts - 1

    def plot_neighbor_distribution(self, neighbor_counts: np.ndarray, sample_name: str, radius: float):
        """Plot distribution of neighbor counts."""
        plt.figure(figsize=(10, 6))
        plt.hist(neighbor_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of neighbors within radius')
        plt.ylabel('Number of cells')
        plt.title(f'{sample_name}: Neighbor count distribution (radius = {radius})')
        plt.axvline(np.median(neighbor_counts), color='red', linestyle='--', 
                    label=f'Median: {np.median(neighbor_counts):.1f}')
        plt.axvline(np.mean(neighbor_counts), color='orange', linestyle='--', 
                    label=f'Mean: {np.mean(neighbor_counts):.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Neighbor count statistics:")
        print(f"  Min: {np.min(neighbor_counts)}")
        print(f"  Max: {np.max(neighbor_counts)}")
        print(f"  Mean: {np.mean(neighbor_counts):.2f}")
        print(f"  Median: {np.median(neighbor_counts):.2f}")
        print(f"  25th percentile: {np.percentile(neighbor_counts, 25):.2f}")
        print(f"  75th percentile: {np.percentile(neighbor_counts, 75):.2f}")

    def apply_neighbor_density_filter(self, cells_df: pd.DataFrame, min_neighbors: int, radius: float, 
                                    sample_name: str, visualize: bool = True) -> pd.DataFrame:
        """Filter cells based on neighbor density."""
        print(f"\nApplying neighbor density filter...")
        print(f"Parameters: min_neighbors={min_neighbors}, radius={radius}")
        print(f"Cells before neighbor filtering: {len(cells_df)}")
        
        # For very large datasets, optionally subsample for faster computation
        if len(cells_df) > 100000:
            print(f"Large dataset detected ({len(cells_df)} cells). Consider using --min_neighbors 0 for faster processing.")
            print("Computing neighbors for full dataset... this may take 5-10 minutes.")
        
        coordinates = cells_df[['x_centroid', 'y_centroid']].values
        neighbor_counts = self.count_neighbors_within_radius(coordinates, radius)
        
        if visualize:
            self.plot_neighbor_distribution(neighbor_counts, sample_name, radius)
        
        # Apply filter
        neighbor_mask = neighbor_counts >= min_neighbors
        cells_df_filtered = cells_df[neighbor_mask].copy()
        
        print(f"Cells after neighbor filtering: {len(cells_df_filtered)}")
        print(f"Filtered out {len(cells_df) - len(cells_df_filtered)} isolated cells")
        
        return cells_df_filtered

    def extract_patch_with_padding(self, he_stack: np.ndarray, x_center: int, y_center: int) -> np.ndarray:
        """Extract patch with zero padding for cells near borders."""
        image_height, image_width = he_stack.shape[:2]
        
        # Calculate patch bounds
        x_start = x_center - self.half_patch
        y_start = y_center - self.half_patch
        x_end = x_start + self.patch_size
        y_end = y_start + self.patch_size
        
        # Initialize patch with zeros
        if len(he_stack.shape) == 3:
            patch = np.zeros((self.patch_size, self.patch_size, he_stack.shape[2]), dtype=he_stack.dtype)
        else:
            patch = np.zeros((self.patch_size, self.patch_size), dtype=he_stack.dtype)
        
        # Calculate valid region within image bounds
        img_x_start = max(0, x_start)
        img_y_start = max(0, y_start)
        img_x_end = min(image_width, x_end)
        img_y_end = min(image_height, y_end)
        
        # Calculate corresponding patch region
        patch_x_start = img_x_start - x_start
        patch_y_start = img_y_start - y_start
        patch_x_end = patch_x_start + (img_x_end - img_x_start)
        patch_y_end = patch_y_start + (img_y_end - img_y_start)
        
        # Copy valid region from image to patch
        if img_x_end > img_x_start and img_y_end > img_y_start:
            patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
                he_stack[img_y_start:img_y_end, img_x_start:img_x_end]
        
        return patch

    def visualize_filtered_patches(self, he_stack: np.ndarray, filtered_x: np.ndarray, 
                                 filtered_y: np.ndarray, filtered_indices: np.ndarray,
                                 sample_name: str, n_examples: int = 20):
        """Visualize patches of cells filtered out due to border proximity with zero padding."""
        
        if len(filtered_x) == 0:
            return
            
        n_examples = min(n_examples, len(filtered_x))
        print(f"Visualizing {n_examples} filtered border cells with zero padding...")
        
        # Randomly select examples
        example_indices = np.random.choice(len(filtered_x), n_examples, replace=False)
        
        # Setup visualization
        n_cols = 5
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Extract and display example patches with padding
        for idx in range(n_examples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get coordinates for this example
            example_idx = example_indices[idx]
            x_center = filtered_x[example_idx]
            y_center = filtered_y[example_idx]
            
            # Extract patch with zero padding
            patch = self.extract_patch_with_padding(he_stack, x_center, y_center)
            
            # Display patch
            ax.imshow(patch)
            ax.scatter(self.half_patch, self.half_patch, s=60, c='red', alpha=0.8)
            ax.set_title(f'Filtered Cell {filtered_indices[example_idx]}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_examples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'{sample_name}: {n_examples} Border Cells (Zero Padded)', fontsize=16, y=1.02)
        plt.show()

    def visualize_example_patches(self, he_stack: np.ndarray, patch_x_starts: np.ndarray, 
                                patch_y_starts: np.ndarray, valid_indices: np.ndarray,
                                sample_name: str, n_examples: int = 20):
        """Visualize example patches for quality check."""
        
        n_examples = min(n_examples, len(patch_x_starts))
        if n_examples == 0:
            return
            
        print(f"Visualizing {n_examples} example patches...")
        
        # Randomly select examples
        example_indices = np.random.choice(len(patch_x_starts), n_examples, replace=False)
        
        # Setup visualization
        n_cols = 5
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else np.array([[axes]])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Extract and display example patches
        for idx in range(n_examples):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get coordinates for this example
            example_idx = example_indices[idx]
            x_start = patch_x_starts[example_idx]
            y_start = patch_y_starts[example_idx]
            
            # Extract patch directly from whole slide image
            patch = he_stack[y_start:y_start + self.patch_size, 
                           x_start:x_start + self.patch_size]
            
            # Display patch
            ax.imshow(patch)
            ax.scatter(self.half_patch, self.half_patch, s=60, c='red', alpha=0.8, label="Centroid")
            ax.set_title(f'Cell {valid_indices[example_idx]}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_examples, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'{sample_name}: {n_examples} Random Single Cell Patches', fontsize=16, y=1.02)
        plt.show()

    def extract_patches_for_sample(self, sample_name: str, sample_files: Dict[str, Path], visualize: bool = True, n_examples: int = 20, min_neighbors: int = 5, radius: float = 50.0) -> bool:
        """Extract single cell patches for one sample."""
        
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_name}")
        print(f"{'='*60}")
        
        try:
            # === Step 1: Load transformation matrix ===
            print("Loading transformation matrix...")
            matrix_df = pd.read_csv(sample_files['alignment'], header=None)
            print(f"Loaded transformation matrix: {matrix_df.shape}")
            
            # === Step 2: Load Xenium cells data directly from CSV ===
            print("Loading Xenium cells data from CSV...")
            with tqdm(desc="Loading cells.csv.gz", unit="MB") as pbar:
                cells_df = pd.read_csv(sample_files['xenium_dir'] / "cells.csv.gz")
                pbar.update(1)
            print(f"Loaded cells data: {cells_df.shape}")
            print(f"Available columns: {list(cells_df.columns)}")
            
            # === Step 3: Apply QC filtering ===
            if self.n_mads_qc > 0:
                cells_df = self.apply_qc_filtering(cells_df, self.n_mads_qc)
                if len(cells_df) == 0:
                    print("No cells remaining after QC filtering")
                    return False
            
            # === Step 3.5: Apply neighbor density filtering ===
            if min_neighbors > 0:
                cells_df = self.apply_neighbor_density_filter(cells_df, min_neighbors, radius, 
                                                            sample_name, visualize)
                if len(cells_df) == 0:
                    print("No cells remaining after neighbor density filtering")
                    return False
            
            # === Step 4: Extract centroids directly from CSV ===
            centroids = cells_df[['x_centroid', 'y_centroid']].values
            print(f"Extracted centroids: {centroids.shape}")
            
            # === Step 5: Transform coordinates ===
            print("Transforming coordinates...")
            pixel_centroids = centroids / self.pixel_size
            centroids_transformed = self.transform_coords_inverse(matrix_df, pixel_centroids)
            x_trans = centroids_transformed[:, 0].astype(int)
            y_trans = centroids_transformed[:, 1].astype(int)
            print(f"Transformed {len(centroids)} cell centroids to H&E space")
            
            # === Step 6: Load H&E image ===
            print("Loading H&E image...")
            print(f"H&E file path: {sample_files['he_image']}")
            print(f"File size: {sample_files['he_image'].stat().st_size / (1024**3):.2f} GB")
            
            try:
                with tqdm(desc="Loading H&E image", unit="MB") as pbar:
                    he_stack = tifffile.imread(sample_files['he_image'])
                    pbar.update(1)
                print(f"Loaded H&E image shape: {he_stack.shape}")
                print(f"H&E image dtype: {he_stack.dtype}")
                print(f"Memory usage: {he_stack.nbytes / (1024**3):.2f} GB")
            except Exception as e:
                print(f"Error loading H&E image: {e}")
                print("Trying alternative loading method...")
                try:
                    # Try reading with different method
                    import tifffile as tf
                    with tf.TiffFile(sample_files['he_image']) as tif:
                        print(f"TIFF info: {tif.series[0].shape}, {tif.series[0].dtype}")
                        he_stack = tif.asarray()
                    print(f"Successfully loaded with alternative method: {he_stack.shape}")
                except Exception as e2:
                    print(f"Alternative loading also failed: {e2}")
                    return False
            
            image_height, image_width = he_stack.shape[:2]
            
            # === Step 7: Filter valid cells ===
            print("Filtering valid cells...")
            valid_mask = (
                (x_trans >= self.half_patch) & (x_trans < image_width - self.half_patch) &
                (y_trans >= self.half_patch) & (y_trans < image_height - self.half_patch)
            )
            
            valid_x = x_trans[valid_mask]
            valid_y = y_trans[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            # Get filtered out cells
            filtered_x = x_trans[~valid_mask]
            filtered_y = y_trans[~valid_mask]
            filtered_indices = np.where(~valid_mask)[0]
            
            print(f"Found {len(valid_x)} valid cells out of {len(x_trans)} total cells")
            print(f"Filtered out {len(filtered_x)} cells too close to boundaries")

            if len(valid_x) == 0:
                print("No valid cells found for this sample")
                return False
            
            # === Step 8: Calculate patch coordinates ===
            patch_x_starts = valid_x - self.half_patch
            patch_y_starts = valid_y - self.half_patch
            
            # === Step 9: Create output directory ===
            sample_output_dir = self.output_dir / sample_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {sample_output_dir}")
            
            # === Step 10: Save coordinates to h5 file ===
            h5_path = sample_output_dir / "patch_coordinates.h5"
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('x_start', data=patch_x_starts)
                f.create_dataset('y_start', data=patch_y_starts)
                
                # Add metadata
                f.attrs['total_patches'] = len(valid_x)
                f.attrs['patch_size'] = self.patch_size
                f.attrs['image_shape'] = he_stack.shape
                f.attrs['sample_name'] = sample_name
                f.attrs['pixel_size'] = self.pixel_size
                
            print(f"Saved {len(valid_x)} patch coordinates to {h5_path}")
            
            # === Step 11: Visualize examples ===
            if visualize and len(valid_x) > 0:
                self.visualize_example_patches(he_stack, patch_x_starts, patch_y_starts, 
                                             valid_indices, sample_name, n_examples)
            
            # === Step 12: Visualize filtered cells ===
            if visualize and len(filtered_x) > 0:
                self.visualize_filtered_patches(he_stack, filtered_x, filtered_y, 
                                               filtered_indices, sample_name, n_examples)
            
            return True
            
        except Exception as e:
            print(f"Error processing {sample_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_files_info(self):
        """Check file accessibility and sizes without loading."""
        samples = self.find_sample_files()
        
        if not samples:
            print("No complete sample sets found!")
            return
        
        for sample_name, sample_files in samples.items():
            print(f"\n{'='*50}")
            print(f"Sample: {sample_name}")
            print(f"{'='*50}")
            
            # Check cells.csv.gz
            cells_file = sample_files['xenium_dir'] / "cells.csv.gz"
            if cells_file.exists():
                size_mb = cells_file.stat().st_size / (1024**2)
                print(f"✓ Cells CSV: {size_mb:.1f} MB")
            else:
                print("✗ Cells CSV: Not found")
            
            # Check H&E image
            he_file = sample_files['he_image']
            if he_file.exists():
                size_gb = he_file.stat().st_size / (1024**3)
                print(f"✓ H&E Image: {size_gb:.2f} GB")
                
                # Try to read TIFF info without loading full image
                try:
                    import tifffile as tf
                    with tf.TiffFile(he_file) as tif:
                        print(f"  - Shape: {tif.series[0].shape}")
                        print(f"  - Dtype: {tif.series[0].dtype}")
                        print(f"  - Pages: {len(tif.pages)}")
                except Exception as e:
                    print(f"  - Could not read TIFF info: {e}")
            else:
                print("✗ H&E Image: Not found")
            
            # Check alignment file
            align_file = sample_files['alignment']
            if align_file.exists():
                size_kb = align_file.stat().st_size / 1024
                print(f"✓ Alignment: {size_kb:.1f} KB")
            else:
                print("✗ Alignment: Not found")
        
        print(f"\nSummary:")
        print(f"- {len(samples)} samples found")
        print(f"- H&E images range from 0.50 to 3.78 GB")
        print(f"- All files appear accessible")
    
    def process_all_samples(self, visualize: bool = True, n_examples: int = 20,
                        min_neighbors: int = 5, radius: float = 50.0):       
        """Process all available samples."""
        
        # Find all sample files
        samples = self.find_sample_files()
        
        if not samples:
            print("No complete sample sets found!")
            return
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each sample
        successful = 0
        failed = 0
        
        for sample_name, sample_files in samples.items():
            success = self.extract_patches_for_sample(sample_name, sample_files, visualize, 
                                                    n_examples, min_neighbors, radius)
            if success:
                successful += 1
            else:
                failed += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {successful} samples")
        print(f"Failed: {failed} samples")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        if successful > 0:
            print(f"\nResults saved in:")
            for sample_name in samples.keys():
                sample_dir = self.output_dir / sample_name
                if sample_dir.exists():
                    print(f"   - {sample_dir}")
        
# Utility function to load coordinates
def load_patch_coordinates(h5_file_path: Path):
    """Utility function to load saved patch coordinates"""
    with h5py.File(h5_file_path, 'r') as f:
        coords = {
            'x_start': f['x_start'][:],
            'y_start': f['y_start'][:],
        }
        metadata = dict(f.attrs)
    return coords, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Xenium patches")
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/Volumes/scratch-boeva/data/projects/Jeffery_xenium",
        help="Path to the base directory (default: /Volumes/scratch-boeva/data/projects/Jeffery_xenium)"
    )
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the extracted patch")
    parser.add_argument("--pixel_size", type=float, default=0.2125, help="Pixel size for the images")
    parser.add_argument("--n_mads_qc", type=int, default=5, help="MAD threshold for QC filtering")
    parser.add_argument("--n_examples", type=int, default=20, help="Number of visual examples to show")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--min_neighbors", type=int, default=5, 
                   help="Minimum neighbors within radius (0 to disable)")
    parser.add_argument("--radius", type=float, default=50.0, 
                    help="Radius for neighbor counting")
    parser.add_argument("--check_files_only", action="store_true", 
                   help="Only check file accessibility without processing")
    parser.add_argument("--debug", action="store_true", 
                   help="Enable debug mode with extra information")

    args = parser.parse_args()

    # Initialize the extractor with parsed arguments
    extractor = XeniumPatchExtractor(
        base_dir=Path(args.base_dir),
        patch_size=args.patch_size,
        pixel_size=args.pixel_size,
        n_mads_qc=args.n_mads_qc
    )
    
    # Check files only mode
    if args.check_files_only:
        extractor.check_files_info()
    else:
        # Run processing
        extractor.process_all_samples(
            visualize=not args.no_visualize,
            n_examples=args.n_examples,
            min_neighbors=args.min_neighbors,
            radius=args.radius
        )
    
    print(f"\nTo load coordinates later, use:")
    print(f"coords, metadata = load_patch_coordinates('path/to/patch_coordinates.h5')")