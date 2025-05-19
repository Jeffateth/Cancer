import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, exposure, filters, measure, color
import pandas as pd
from scipy.spatial import KDTree
from scipy import ndimage
import os
import time
import warnings
import random
from tqdm import tqdm  # For progress bars

def extract_xenium_nucleus_centroids_from_spatialdata(sdata):
    """
    Extract nucleus centroids from a SpatialData object containing Xenium data.
    """
    print("Extracting nucleus centroids...")
    available_keys = list(sdata.shapes.keys())
    print(f"Available shape keys: {available_keys}")

    # Try to get nucleus boundaries or cell circles to extract centroids
    if 'nucleus_boundaries' in sdata.shapes:
        print("Using 'nucleus_boundaries' from sdata.shapes")
        nucleus_shapes = sdata.shapes['nucleus_boundaries']
        nucleus_centroids = np.array([
            (geom.centroid.x, geom.centroid.y) 
            for geom in nucleus_shapes.geometry
        ])
        cell_ids = nucleus_shapes.index.values
        print(f"Extracted {nucleus_centroids.shape[0]} nucleus centroids")

    elif 'cell_boundaries' in sdata.shapes:
        print("Using 'cell_boundaries' from sdata.shapes")
        cell_shapes = sdata.shapes['cell_boundaries']
        nucleus_centroids = np.array([
            (geom.centroid.x, geom.centroid.y) 
            for geom in cell_shapes.geometry
        ])
        cell_ids = cell_shapes.index.values
        print(f"Extracted {nucleus_centroids.shape[0]} cell centroids")

    elif 'cell_circles' in sdata.shapes:
        print("Using 'cell_circles' from sdata.shapes")
        cell_circles = sdata.shapes['cell_circles']
        nucleus_centroids = np.array([
            (geom.centroid.x, geom.centroid.y) 
            for geom in cell_circles.geometry
        ])
        cell_ids = cell_circles.index.values
        print(f"Extracted {nucleus_centroids.shape[0]} cell circle centroids")

    else:
        print("No nucleus or cell shapes found. Trying 'spatial' in obsm...")
        if hasattr(sdata.tables, 'table') and hasattr(sdata.tables['table'], 'obsm') and 'spatial' in sdata.tables['table'].obsm:
            nucleus_centroids = sdata.tables['table'].obsm['spatial']
            cell_ids = sdata.tables['table'].obs_names.values
            print(f"Extracted {nucleus_centroids.shape[0]} centroids from obsm['spatial']")
        else:
            raise ValueError("Could not find nucleus/cell centroids in the SpatialData object")

    return nucleus_centroids, cell_ids

def estimate_affine_transform(src_points, dst_points):
    """
    Estimate affine transformation matrix from source to destination points.
    
    Parameters:
    -----------
    src_points : ndarray
        Source points (e.g., Xenium coordinates)
    dst_points : ndarray
        Destination points (e.g., H&E coordinates)
        
    Returns:
    --------
    transform_model : AffineTransform
        Estimated transformation model
    """
    # Ensure we have at least 3 point pairs for a robust estimation
    assert src_points.shape[0] >= 3, "Need at least 3 points to estimate affine transform"
    assert src_points.shape == dst_points.shape, "Source and destination point counts must match"

    # Estimate affine transformation
    transform_model = transform.AffineTransform()
    success = transform_model.estimate(src_points, dst_points)
    
    if not success:
        warnings.warn("Affine transform estimation failed. Using best estimate anyway.")
    
    print(f"Estimated transform matrix:\n{transform_model.params}")
    return transform_model

def apply_inverse_affine_transform(xenium_centroids, transform_matrix):
    """
    Apply inverse affine transformation to transform Xenium coordinates to H&E space.
    """
    he_coordinates = transform_matrix.inverse(xenium_centroids)
    print(f"Transformed {len(he_coordinates)} Xenium centroids to H&E space")
    return he_coordinates

def extract_image_patches_enhanced(he_image, transformed_coords, cell_ids=None,
                                  P=224, kappas=[0.5, 0.75, 1.0, 1.5, 2.0],
                                  resample_order=3, apply_normalization=True):
    """
    Extract and enhance 224×224 patches at zoom levels κ around each point.
    
    Parameters:
    -----------
    he_image : ndarray
        H&E image from which to extract patches
    transformed_coords : ndarray
        Coordinates in H&E space where patches should be centered
    cell_ids : array-like, optional
        Cell IDs corresponding to the coordinates
    P : int
        Patch size (default 224)
    kappas : list
        Zoom levels (default [0.5, 0.75, 1.0, 1.5, 2.0])
    resample_order : int
        Interpolation order for resizing (default 3)
    apply_normalization : bool
        Whether to apply color normalization (default True)
        
    Returns:
    --------
    patches : dict
        Dictionary mapping from index or cell_id to list of patches for each kappa
    """
    H, W = he_image.shape[:2]
    print(f"Extracting patches: image shape=({H},{W}), "
          f"num_points={len(transformed_coords)}, P={P}, kappas={kappas}")
    
    # Check if cell_ids were provided
    if cell_ids is None:
        cell_ids = np.arange(len(transformed_coords))
    
    patches = {}
    debug_info = {}  # Store additional debug info

    for i, ((x, y), cell_id) in enumerate(zip(transformed_coords, cell_ids)):
        xi, yi = int(round(x)), int(round(y))
        
        # Skip out-of-bounds
        if not (0 <= xi < W and 0 <= yi < H):
            continue

        patches[cell_id] = []
        debug_info[cell_id] = {'coords': (xi, yi)}
        
        for κ in kappas:
            # Compute raw patch size to crop
            crop_size = int(np.ceil(P / κ))
            half = crop_size // 2

            # Calculate crop bounds
            x0 = max(0, xi - half)
            y0 = max(0, yi - half)
            x1 = min(W, xi + half)
            y1 = min(H, yi + half)

            # Crop
            crop = he_image[y0:y1, x0:x1]

            # Pad if needed
            pad_h = crop_size - crop.shape[0]
            pad_w = crop_size - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                pad_top = max(0, half - yi)
                pad_left = max(0, half - xi)
                pad_bot = pad_h - pad_top
                pad_right = pad_w - pad_left
                
                crop = np.pad(crop,
                              ((pad_top, pad_bot),
                               (pad_left, pad_right),
                               (0, 0)),
                              mode='constant',
                              constant_values=0)

            # Resize back to P×P
            patch = transform.resize(
                crop,
                (P, P),
                order=resample_order,
                preserve_range=True
            ).astype(np.uint8)

            # Apply color normalization if requested
            if apply_normalization:
                # Skip if patch is completely black
                if np.mean(patch) < 5:
                    patches[cell_id].append(patch)
                    continue
                    
                # Apply contrast enhancement
                p2, p98 = np.percentile(patch, (2, 98), axis=(0, 1))
                patch = exposure.rescale_intensity(patch, in_range=(p2, p98))
                
                # Apply color normalization as described in the paper
                # These values come from the paper: ImageNet mean and std
                means = np.array([0.485, 0.456, 0.406])
                stds = np.array([0.229, 0.224, 0.225])
                
                # Convert to float [0,1]
                normalized = patch.astype(np.float32) / 255.0
                
                # For each channel, normalize
                for c in range(3):
                    normalized[:,:,c] = (normalized[:,:,c] - means[c]) / stds[c]
                
                # Convert back to uint8 for visualization
                normalized = (normalized * stds + means) * 255.0
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)
                
                patches[cell_id].append(normalized)
            else:
                patches[cell_id].append(patch)

    print(f"Extracted patches for {len(patches)} valid coordinates")
    return patches, debug_info

def evaluate_patch_quality(patch, min_content_ratio=0.05, nucleus_size_range=(100, 5000)):
    """
    Evaluate if a patch contains a valid cell/nucleus.
    
    Parameters:
    -----------
    patch : ndarray
        Image patch to evaluate
    min_content_ratio : float
        Minimum ratio of foreground to total pixels
    nucleus_size_range : tuple
        Expected nucleus size range in pixels
        
    Returns:
    --------
    is_valid : bool
        Boolean indicating if patch contains a valid cell
    score : float
        Quality score (higher is better)
    reason : str
        Reason for rejection if not valid
    """
    # Convert to grayscale if it's RGB
    if patch.ndim == 3:
        gray = color.rgb2gray(patch)
    else:
        gray = patch
    
    # Check for completely black patches
    if np.mean(gray) < 0.05:
        return False, 0.0, "Completely black patch"
    
    # Calculate basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # If almost entirely background (very little signal)
    if std_intensity < 0.03:
        return False, 0.1, "Low signal variation"
    
    # Apply Otsu thresholding to separate foreground and background
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh  # H&E stained nuclei are darker
    
    # Calculate ratio of foreground to total pixels
    content_ratio = np.sum(binary) / binary.size
    
    # If too little foreground
    if content_ratio < min_content_ratio:
        return False, 0.2, "Insufficient content"
    
    # Label connected components (potential nuclei)
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    
    # If no regions found
    if len(props) == 0:
        return False, 0.3, "No nuclei detected"
    
    # Check if any region is within the expected nucleus size range
    valid_nuclei = [p for p in props if nucleus_size_range[0] <= p.area <= nucleus_size_range[1]]
    
    # If no valid nuclei
    if len(valid_nuclei) == 0:
        return False, 0.4, "No valid-sized nuclei"
    
    # Calculate centrality - we want nuclei near the center
    center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
    
    # Find the largest nucleus
    largest_nucleus = max(valid_nuclei, key=lambda p: p.area)
    y, x = largest_nucleus.centroid
    
    # Distance from center (normalized)
    dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2) / (gray.shape[0] / 2)
    centrality_score = 1 - min(dist_from_center, 1.0)  # Higher score for more central
    
    # Calculate overall quality score
    quality_score = (centrality_score * 0.6) + (min(content_ratio, 0.5) / 0.5 * 0.4)
    
    return True, quality_score, "Valid"

def apply_quality_control(patches, kappas=[0.5, 0.75, 1.0, 1.5, 2.0]):
    """
    Apply quality control to patches.
    
    Parameters:
    -----------
    patches : dict
        Dictionary mapping from cell_id to list of patches for each kappa
    kappas : list
        Zoom levels
        
    Returns:
    --------
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results for each kappa
    filtered_patches : dict
        Dictionary containing only valid patches
    """
    quality_results = {}
    filtered_patches = {}
    
    print("Applying quality control to patches...")
    for cell_id in patches:
        quality_results[cell_id] = []
        filtered_patches[cell_id] = []
        
        for i, patch in enumerate(patches[cell_id]):
            kappa = kappas[i] if i < len(kappas) else None
            
            # Evaluate quality
            is_valid, score, reason = evaluate_patch_quality(patch)
            
            quality_results[cell_id].append({
                'kappa': kappa,
                'score': score,
                'is_valid': is_valid,
                'reason': reason
            })
            
            # Keep the patch regardless of validity for review
            filtered_patches[cell_id].append(patch)
    
    # Analyze results
    valid_count = 0
    total_count = 0
    kappa_stats = {k: {'valid': 0, 'total': 0} for k in kappas}
    
    for cell_id in quality_results:
        for result in quality_results[cell_id]:
            kappa = result['kappa']
            is_valid = result['is_valid']
            
            kappa_stats[kappa]['total'] += 1
            if is_valid:
                kappa_stats[kappa]['valid'] += 1
                valid_count += 1
            total_count += 1
    
    print(f"Quality control summary: {valid_count}/{total_count} valid patches ({valid_count/total_count*100:.1f}%)")
    for k, stats in kappa_stats.items():
        if stats['total'] > 0:
            valid_percent = stats['valid'] / stats['total'] * 100
            print(f"  κ={k}: {stats['valid']}/{stats['total']} valid ({valid_percent:.1f}%)")
    
    return quality_results, filtered_patches

def remove_black_patches(patches, quality_results, kappas=[0.5, 0.75, 1.0, 1.5, 2.0]):
    """
    Apply paper-specific QC: removing completely black patches at zoom level κ=2.0
    """
    black_patch_count = 0
    
    for cell_id in patches:
        for i, kappa in enumerate(kappas):
            if kappa == 2.0:  # Check specifically at kappa=2.0 as mentioned in the paper
                patch = patches[cell_id][i]
                if np.mean(patch) < 5:  # Very dark patch
                    quality_results[cell_id][i]['is_valid'] = False
                    quality_results[cell_id][i]['reason'] = "Completely black patch"
                    black_patch_count += 1
    
    print(f"Removed {black_patch_count} completely black patches at κ=2.0")
    return patches, quality_results

def sample_cells_by_grid(centroids, grid_size=10, cells_per_grid=2):
    """
    Sample cells evenly across the tissue by dividing into a grid.
    
    Parameters:
    -----------
    centroids : ndarray
        Array of (x, y) coordinates
    grid_size : int
        Number of grid cells in each dimension
    cells_per_grid : int
        Number of cells to sample from each grid cell
        
    Returns:
    --------
    sampled_indices : ndarray
        Indices of sampled cells
    """
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size
    
    sampled_indices = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x_start = x_min + i * x_step
            x_end = x_min + (i + 1) * x_step
            y_start = y_min + j * y_step
            y_end = y_min + (j + 1) * y_step
            
            # Find cells in this grid cell
            mask = ((x_coords >= x_start) & (x_coords < x_end) & 
                    (y_coords >= y_start) & (y_coords < y_end))
            grid_indices = np.where(mask)[0]
            
            # Sample from this grid cell if there are cells
            if len(grid_indices) > 0:
                n_sample = min(cells_per_grid, len(grid_indices))
                sampled = np.random.choice(grid_indices, n_sample, replace=False)
                sampled_indices.extend(sampled)
    
    return np.array(sampled_indices)

def find_best_kappa_by_cell(quality_results, kappas=[0.5, 0.75, 1.0, 1.5, 2.0]):
    """
    Find the best kappa value for each cell based on quality scores.
    
    Parameters:
    -----------
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results for each kappa
    kappas : list
        Zoom levels
        
    Returns:
    --------
    best_kappas : dict
        Dictionary mapping from cell_id to best kappa value and score
    """
    best_kappas = {}
    
    for cell_id, results in quality_results.items():
        valid_results = [(i, r) for i, r in enumerate(results) if r['is_valid']]
        
        if valid_results:
            # Find the best kappa based on score
            best_idx, best_result = max(valid_results, key=lambda x: x[1]['score'])
            best_kappas[cell_id] = {
                'kappa': best_result['kappa'],
                'kappa_idx': best_idx,
                'score': best_result['score']
            }
    
    print(f"Found best kappa values for {len(best_kappas)}/{len(quality_results)} cells")
    return best_kappas

def visualize_overlay(he_image, xenium_coords, he_coords, sample_indices=None, figsize=(15, 10), title=None):
    """
    Visualize the overlay of Xenium nucleus centroids on H&E image.
    
    Parameters:
    -----------
    he_image : ndarray
        H&E image
    xenium_coords : ndarray
        Original Xenium coordinates
    he_coords : ndarray
        Transformed coordinates in H&E space
    sample_indices : array-like, optional
        Indices of points to visualize
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    """
    total_points = len(xenium_coords)
    if sample_indices is None:
        if total_points > 100:
            sample_indices = np.random.choice(total_points, 100, replace=False)
        else:
            sample_indices = range(total_points)

    plt.figure(figsize=figsize)
    plt.imshow(he_image)
    he_x = he_coords[sample_indices, 0]
    he_y = he_coords[sample_indices, 1]
    plt.scatter(he_x, he_y, c='red', s=10, alpha=0.7, label='Transformed Xenium centroids')

    if title is None:
        title = 'Xenium centroids overlaid on H&E image'
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_patches(patches, quality_results, kappas=[0.5, 0.75, 1.0, 1.5, 2.0], 
                      n_cells=5, best_kappas=None, figsize=(15, 15), title=None):
    """
    Visualize patches for multiple cells at different zoom levels.
    
    Parameters:
    -----------
    patches : dict
        Dictionary mapping from cell_id to list of patches for each kappa
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results
    kappas : list
        Zoom levels
    n_cells : int
        Number of cells to visualize
    best_kappas : dict, optional
        Dictionary mapping from cell_id to best kappa value
    figsize : tuple
        Figure size
    title : str, optional
        Figure title
    """
    # Select cells to visualize
    if best_kappas:
        # Take cells with highest quality scores
        top_cells = sorted(
            [(cell_id, best_kappas[cell_id]['score']) for cell_id in best_kappas],
            key=lambda x: x[1],
            reverse=True
        )[:n_cells]
        cell_ids = [cell_id for cell_id, _ in top_cells]
    else:
        # Take random cells
        cell_ids = list(patches.keys())
        if len(cell_ids) > n_cells:
            cell_ids = random.sample(cell_ids, n_cells)
    
    fig, axes = plt.subplots(len(cell_ids), len(kappas), figsize=figsize)
    
    for i, cell_id in enumerate(cell_ids):
        for j, kappa in enumerate(kappas):
            ax = axes[i, j] if len(cell_ids) > 1 else axes[j]
            patch = patches[cell_id][j]
            ax.imshow(patch)
            
            # Show quality score and validity
            result = quality_results[cell_id][j]
            title = f"κ={kappa}"
            if best_kappas and cell_id in best_kappas and best_kappas[cell_id]['kappa'] == kappa:
                title += f" (best, {result['score']:.2f})"
            elif result['is_valid']:
                title += f" ({result['score']:.2f})"
            else:
                title += f" (invalid: {result['reason']})"
            
            ax.set_title(title)
            ax.axis('off')
    
    if title is None:
        title = f"Patches for {len(cell_ids)} cells at different zoom levels"
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_quality_distribution(quality_results, kappas=[0.5, 0.75, 1.0, 1.5, 2.0], figsize=(15, 5)):
    """
    Visualize the distribution of quality scores for different kappa values.
    
    Parameters:
    -----------
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results
    kappas : list
        Zoom levels
    figsize : tuple
        Figure size
    """
    scores_by_kappa = {k: [] for k in kappas}
    
    for cell_id, results in quality_results.items():
        for i, result in enumerate(results):
            if i < len(kappas):
                kappa = kappas[i]
                scores_by_kappa[kappa].append(result['score'])
    
    plt.figure(figsize=figsize)
    for i, kappa in enumerate(kappas):
        plt.subplot(1, len(kappas), i+1)
        plt.hist(scores_by_kappa[kappa], bins=20)
        plt.title(f"κ={kappa}")
        plt.xlabel("Quality Score")
        plt.ylabel("Count")
        plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    
    plt.tight_layout()
    plt.show()

def export_best_patches(patches, best_kappas, output_dir="best_patches"):
    """
    Export the best patches for each cell.
    
    Parameters:
    -----------
    patches : dict
        Dictionary mapping from cell_id to list of patches for each kappa
    best_kappas : dict
        Dictionary mapping from cell_id to best kappa value
    output_dir : str
        Directory to save the patches
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Exporting best patches to {output_dir}...")
    
    # Create a CSV file with metadata
    metadata = []
    
    for cell_id, best in best_kappas.items():
        kappa_idx = best['kappa_idx']
        best_patch = patches[cell_id][kappa_idx]
        
        # Save the patch
        patch_filename = f"{cell_id}_kappa_{best['kappa']}.png"
        patch_path = os.path.join(output_dir, patch_filename)
        plt.imsave(patch_path, best_patch)
        
        # Add to metadata
        metadata.append({
            'cell_id': cell_id,
            'kappa': best['kappa'],
            'score': best['score'],
            'filename': patch_filename
        })
    
    # Save metadata as CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    print(f"Exported {len(metadata)} patches")

def preview_cell_patches(sdata, he_image_path, matching_points_path=None, 
                        n_cells=5, kappas=[0.5, 0.75, 1.0, 1.5, 2.0],
                        apply_normalization=True):
    """
    Generate a preview of cell patches from a few selected cells to test the approach
    before running the full pipeline.
    
    Parameters:
    -----------
    sdata : SpatialData
        SpatialData object containing Xenium data
    he_image_path : str
        Path to H&E image
    matching_points_path : str, optional
        Path to CSV file with matching points
    n_cells : int
        Number of cells to preview (default 5)
    kappas : list
        Zoom levels to test (default [0.5, 0.75, 1.0, 1.5, 2.0])
    apply_normalization : bool
        Whether to apply color normalization (default True)
        
    Returns:
    --------
    patches : dict
        Dictionary mapping from cell_id to list of patches for each kappa
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results
    """
    print(f"Generating preview for {n_cells} cells at {len(kappas)} zoom levels")
    
    # 1. Extract nucleus centroids
    xenium_centroids, cell_ids = extract_xenium_nucleus_centroids_from_spatialdata(sdata)
    
    # 2. Load H&E image
    print(f"Loading H&E image from {he_image_path}")
    he_image = io.imread(he_image_path)
    print(f"H&E image shape: {he_image.shape}")
    
    # 3. Estimate transformation
    if matching_points_path and os.path.exists(matching_points_path):
        print(f"Loading matching points from {matching_points_path}")
        matching_points = pd.read_csv(matching_points_path)
        xenium_points = matching_points[['xenium_x', 'xenium_y']].values
        he_points = matching_points[['he_x', 'he_y']].values
        transform_matrix = estimate_affine_transform(xenium_points, he_points)
    else:
        print("No matching points provided. Using identity transform.")
        transform_matrix = transform.AffineTransform()
    
    # 4. Transform coordinates
    he_coordinates = apply_inverse_affine_transform(xenium_centroids, transform_matrix)
    
    # 5. Visualize overlay with a few sampled points
    sample_indices = np.random.choice(len(he_coordinates), 
                                      min(100, len(he_coordinates)), 
                                      replace=False)
    visualize_overlay(he_image, xenium_centroids, he_coordinates, 
                     sample_indices=sample_indices,
                     title="Preview: Xenium centroids overlaid on H&E image")
    
    # 6. Sample a few cells for preview
    preview_indices = sample_cells_by_grid(xenium_centroids, 
                                         grid_size=int(np.sqrt(n_cells*2)), 
                                         cells_per_grid=1)
    preview_indices = preview_indices[:n_cells]  # Limit to requested number
    preview_cell_ids = [cell_ids[i] for i in preview_indices]
    
    print(f"Selected {len(preview_indices)} cells for preview")
    
    # 7. Extract patches for preview cells
    print("Extracting preview patches...")
    patches, debug_info = extract_image_patches_enhanced(
        he_image, 
        he_coordinates[preview_indices],
        cell_ids=preview_cell_ids,
        apply_normalization=apply_normalization,
        kappas=kappas
    )
    
    # 8. Apply quality control to preview patches
    quality_results, filtered_patches = apply_quality_control(patches, kappas=kappas)
    
    # 9. Find best kappa for each preview cell
    best_kappas = find_best_kappa_by_cell(quality_results, kappas=kappas)
    
    # 10. Visualize results for preview cells
    visualize_patches(filtered_patches, quality_results, kappas=kappas, 
                     n_cells=len(preview_cell_ids), best_kappas=best_kappas,
                     title="Preview: Cell patches at different zoom levels")
    
    # 11. Show quality distribution for preview cells
    visualize_quality_distribution(quality_results, kappas=kappas)
    
    # Create a figure similar to the paper with the preview cells
    create_figure_similar_to_paper(filtered_patches, quality_results, best_kappas,
                                 kappas=kappas, n_cells=min(5, len(preview_cell_ids)),
                                 output_path="preview_cell_patches.png")
    
    print("Preview generation complete")
    
    return filtered_patches, quality_results, best_kappas

def create_figure_similar_to_paper(patches, quality_results, best_kappas, 
                                 kappas=[0.5, 0.75, 1.0, 1.5, 2.0],
                                 n_cells=5, output_path=None):
    """
    Create a figure similar to Figure 4.3 in the paper.
    
    Parameters:
    -----------
    patches : dict
        Dictionary mapping from cell_id to list of patches for each kappa
    quality_results : dict
        Dictionary mapping from cell_id to list of quality results
    best_kappas : dict
        Dictionary mapping from cell_id to best kappa value
    kappas : list
        Zoom levels
    n_cells : int
        Number of cells to visualize
    output_path : str, optional
        Path to save the figure
    """
    # Take cells with highest quality scores
    top_cells = sorted(
        [(cell_id, best_kappas[cell_id]['score']) for cell_id in best_kappas],
        key=lambda x: x[1],
        reverse=True
    )[:n_cells]
    
    cell_ids = [cell_id for cell_id, _ in top_cells]
    
    # Create figure
    fig, axes = plt.subplots(n_cells, len(kappas), figsize=(15, 3*n_cells))
    
    for i, cell_id in enumerate(cell_ids):
        for j, kappa in enumerate(kappas):
            axes[i, j].imshow(patches[cell_id][j])
            axes[i, j].set_title(f"κ={kappa}")
            axes[i, j].axis('off')
            
            # Add a small blue dot at the center to mark the nucleus centroid (like in the paper)
            h, w = patches[cell_id][j].shape[:2]
            axes[i, j].plot(w//2, h//2, 'o', color='cyan', markersize=4)
    
    plt.suptitle("Breast Cancer: 5 random cell IDs at the 5 zoom levels κ ∈ {0.5, 0.75, 1.0, 1.5, 2.0}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def process_xenium_he_data(sdata, he_image_path, matching_points_path=None, 
                         output_dir="xenium_he_output", n_cells=100,
                         apply_normalization=True, visualize=True):
    """
    Full pipeline to process Xenium and H&E data.
    
    Parameters:
    -----------
    sdata : SpatialData
        SpatialData object containing Xenium data
    he_image_path : str
        Path to H&E image
    matching_points_path : str, optional
        Path to CSV file with matching points
    output_dir : str
        Directory to save outputs
    n_cells : int
        Number of cells to process
    apply_normalization : bool
        Whether to apply color normalization
    visualize : bool
        Whether to visualize results
        
    Returns:
    --------
    results : dict
        Dictionary with all results
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    start_time = time.time()
    
    # 1. Extract nucleus centroids
    xenium_centroids, cell_ids = extract_xenium_nucleus_centroids_from_spatialdata(sdata)
    
    # 2. Load H&E image
    print(f"Loading H&E image from {he_image_path}")
    he_image = io.imread(he_image_path)
    print(f"H&E image shape: {he_image.shape}")
    
    # 3. Estimate transformation
    if matching_points_path and os.path.exists(matching_points_path):
        print(f"Loading matching points from {matching_points_path}")
        matching_points = pd.read_csv(matching_points_path)
        xenium_points = matching_points[['xenium_x', 'xenium_y']].values
        he_points = matching_points[['he_x', 'he_y']].values
        transform_matrix = estimate_affine_transform(xenium_points, he_points)
    else:
        print("No matching points provided. Using identity transform.")
        transform_matrix = transform.AffineTransform()
    
    # 4. Transform coordinates
    he_coordinates = apply_inverse_affine_transform(xenium_centroids, transform_matrix)
    
    # 5. Visualize overlay
    if visualize:
        visualize_overlay(he_image, xenium_centroids, he_coordinates)
    
    # 6. Sample cells for processing
    print(f"Sampling {n_cells} cells for processing")
    sample_indices = sample_cells_by_grid(xenium_centroids, grid_size=int(np.sqrt(n_cells)), 
                                         cells_per_grid=5)
    sample_indices = sample_indices[:n_cells]  # Limit to requested number
    sample_cell_ids = [cell_ids[i] for i in sample_indices]
    
    # 7. Extract and enhance patches
    print("Extracting and enhancing patches...")
    patches, debug_info = extract_image_patches_enhanced(
        he_image, 
        he_coordinates[sample_indices],
        cell_ids=sample_cell_ids,
        apply_normalization=apply_normalization
    )
    
    # 8. Apply quality control
    quality_results, filtered_patches = apply_quality_control(patches)
    
    # 9. Apply paper-specific QC (removing black patches)
    filtered_patches, quality_results = remove_black_patches(filtered_patches, quality_results)
    
    # 10. Find best kappa for each cell
    best_kappas = find_best_kappa_by_cell(quality_results)
    
    # 11. Visualize results
    if visualize:
        # Show patches for top cells
        visualize_patches(filtered_patches, quality_results, best_kappas=best_kappas)
        
        # Show quality distribution
        visualize_quality_distribution(quality_results)
        
        # Create a figure similar to the paper
        create_figure_similar_to_paper(filtered_patches, quality_results, best_kappas,
                                     output_path=os.path.join(output_dir, "cell_patches_figure.png"))
    
    # 12. Export best patches
    export_best_patches(filtered_patches, best_kappas, output_dir=os.path.join(output_dir, "best_patches"))
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Return all results
    results = {
        'xenium_centroids': xenium_centroids,
        'cell_ids': cell_ids,
        'he_coordinates': he_coordinates,
        'transform_matrix': transform_matrix,
        'patches': patches,
        'filtered_patches': filtered_patches,
        'quality_results': quality_results,
        'best_kappas': best_kappas
    }
    
    return results