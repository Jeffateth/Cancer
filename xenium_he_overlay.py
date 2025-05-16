import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import pandas as pd
from scipy.spatial import KDTree
import spatialdata as sd
from spatialdata_io import xenium
import geopandas as gpd
from shapely.geometry import Point

print("[DEBUG] Starting debugged Xenium-H&E overlay script")

def extract_xenium_nucleus_centroids_from_spatialdata(sdata):
    """
    Extract nucleus centroids from a SpatialData object containing Xenium data.
    """
    print("[DEBUG] Entering extract_xenium_nucleus_centroids_from_spatialdata")
    print(f"[DEBUG] Available shape keys: {list(sdata.shapes.keys())}")

    # Try to get nucleus boundaries or cell circles to extract centroids
    if 'nucleus_boundaries' in sdata.shapes:
        print("[DEBUG] Using 'nucleus_boundaries' from sdata.shapes")
        nucleus_shapes = sdata.shapes['nucleus_boundaries']
        print(f"[DEBUG] nucleus_shapes contains {len(nucleus_shapes)} geometries")
        nucleus_centroids = np.array([
            (geom.centroid.x, geom.centroid.y) 
            for geom in nucleus_shapes.geometry
        ])
        cell_ids = nucleus_shapes.index.values
        print(f"[DEBUG] Extracted {nucleus_centroids.shape[0]} nucleus centroids")

    elif 'cell_circles' in sdata.shapes:
        print("[DEBUG] Using 'cell_circles' from sdata.shapes")
        cell_circles = sdata.shapes['cell_circles']
        print(f"[DEBUG] cell_circles contains {len(cell_circles)} geometries")
        nucleus_centroids = np.array([
            (geom.centroid.x, geom.centroid.y) 
            for geom in cell_circles.geometry
        ])
        cell_ids = cell_circles.index.values
        print(f"[DEBUG] Extracted {nucleus_centroids.shape[0]} cell circle centroids as nucleus centroids")

    else:
        print("[DEBUG] Neither 'nucleus_boundaries' nor 'cell_circles' found. Trying 'spatial' in obsm...")
        if 'spatial' in sdata.tables['table'].obsm:
            nucleus_centroids = sdata.tables['table'].obsm['spatial']
            cell_ids = sdata.tables['table'].obs_names.values
            print(f"[DEBUG] Extracted {nucleus_centroids.shape[0]} centroids from obsm['spatial']")
        else:
            raise ValueError("Could not find nucleus centroids in the SpatialData object")

    print("[DEBUG] Exiting extract_xenium_nucleus_centroids_from_spatialdata")
    return nucleus_centroids, cell_ids


def estimate_affine_transform(src_points, dst_points):
    """
    Estimate affine transformation matrix from source to destination points.
    """
    print("[DEBUG] Entering estimate_affine_transform")
    print(f"[DEBUG] src_points shape: {src_points.shape}")
    print(f"[DEBUG] dst_points shape: {dst_points.shape}")

    # Ensure we have at least 3 point pairs for a robust estimation
    assert src_points.shape[0] >= 3, "Need at least 3 points to estimate affine transform"
    assert src_points.shape == dst_points.shape, "Source and destination point counts must match"

    # Estimate affine transformation
    transform_model = transform.AffineTransform()
    print("[DEBUG] Calling transform_model.estimate(src_points, dst_points)")
    success = transform_model.estimate(src_points, dst_points)
    print(f"[DEBUG] transform_model.estimate success: {success}")
    print(f"[DEBUG] Estimated transform matrix:\n{transform_model.params}")

    print("[DEBUG] Exiting estimate_affine_transform")
    return transform_model


def apply_inverse_affine_transform(xenium_centroids, transform_matrix):
    """
    Apply inverse affine transformation to transform Xenium coordinates to H&E space.
    """
    print("[DEBUG] Entering apply_inverse_affine_transform")
    print(f"[DEBUG] Number of xenium_centroids: {xenium_centroids.shape[0]}")

    he_coordinates = transform_matrix.inverse(xenium_centroids)
    print(f"[DEBUG] Computed inverse transformation for {he_coordinates.shape[0]} points")

    print("[DEBUG] Exiting apply_inverse_affine_transform")
    return he_coordinates

def extract_image_patches(he_image, transformed_coords,
                          P=224,
                          kappas=[0.5, 0.75, 1.0, 1.5, 2.0],
                          resample_order=3):
    """
    Extract 224×224 patches at zoom levels κ around each point, with detailed debug prints.
    """
    H, W = he_image.shape[:2]
    print(f"[DEBUG] Entering extract_image_patches: image shape=({H},{W}), "
          f"num_points={len(transformed_coords)}, P={P}, kappas={kappas}")
    
    patches = {}

    for i, (x, y) in enumerate(transformed_coords):
        xi, yi = int(round(x)), int(round(y))
        print(f"[DEBUG] Point {i}: rounded coords → (xi, yi)=({xi}, {yi})")
        
        # skip out‐of‐bounds
        if not (0 <= xi < W and 0 <= yi < H):
            print(f"[DEBUG] Point {i} is outside image bounds; skipping.")
            continue

        patches[i] = []
        for κ in kappas:
            # compute raw patch size to crop
            crop_size = int(np.ceil(P / κ))
            half = crop_size // 2
            print(f"[DEBUG]  κ={κ}: crop_size={crop_size}, half={half}")

            x0 = max(0, xi - half)
            y0 = max(0, yi - half)
            x1 = min(W, xi + half)
            y1 = min(H, yi + half)
            print(f"[DEBUG]  crop bounds for κ={κ}: x[{x0}:{x1}], y[{y0}:{y1}]")

            # crop
            crop = he_image[y0:y1, x0:x1]
            print(f"[DEBUG]  cropped shape for κ={κ}: {crop.shape}")

            # pad if needed
            pad_h = crop_size - crop.shape[0]
            pad_w = crop_size - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                pad_top = max(0, half - yi)
                pad_left = max(0, half - xi)
                pad_bot = pad_h - pad_top
                pad_right = pad_w - pad_left
                print(f"[DEBUG]  padding needed for κ={κ}: "
                      f"pad_h={pad_h}, pad_w={pad_w}, "
                      f"top={pad_top}, bottom={pad_bot}, left={pad_left}, right={pad_right}")
                crop = np.pad(crop,
                              ((pad_top, pad_bot),
                               (pad_left, pad_right),
                               (0, 0)),
                              mode='constant',
                              constant_values=0)
                print(f"[DEBUG]  padded shape for κ={κ}: {crop.shape}")

            # resize back to P×P
            patch = transform.resize(
                crop,
                (P, P),
                order=resample_order,
                preserve_range=True
            ).astype(he_image.dtype)
            print(f"[DEBUG]  resized patch shape for κ={κ}: {patch.shape}")

            patches[i].append(patch)

        print(f"[DEBUG] Extracted {len(patches[i])} patches for point {i}")

    print(f"[DEBUG] Exiting extract_image_patches: total valid points = {len(patches)}")
    return patches


def visualize_overlay(he_image, xenium_coords, he_coords, sample_indices=None, figsize=(15, 10)):
    """
    Visualize the overlay of Xenium nucleus centroids on H&E image.
    """
    print("[DEBUG] Entering visualize_overlay")
    total_points = len(xenium_coords)
    if sample_indices is None:
        if total_points > 100:
            sample_indices = np.random.choice(total_points, 100, replace=False)
            print(f"[DEBUG] Randomly selected 100 sample indices from {total_points} points")
        else:
            sample_indices = range(total_points)
            print(f"[DEBUG] Using all {total_points} points for visualization")

    plt.figure(figsize=figsize)
    print(f"[DEBUG] Displaying H&E image and overlaying {len(sample_indices)} points")

    plt.imshow(he_image)
    he_x = he_coords[list(sample_indices), 0]
    he_y = he_coords[list(sample_indices), 1]
    plt.scatter(he_x, he_y, c='red', s=10, alpha=0.7, label='Transformed Xenium centroids')

    plt.title('Xenium centroids overlaid on H&E image')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("[DEBUG] Exiting visualize_overlay")


def main(xenium_data_path=None, he_image_path=None, matching_points_path=None, sdata=None):
    """
    Main function to perform Xenium-H&E overlay.
    """
    print("[DEBUG] Entering main function")
    print(f"[DEBUG] Arguments -> xenium_data_path: {xenium_data_path}, he_image_path: {he_image_path}, matching_points_path: {matching_points_path}, sdata provided: {sdata is not None}")

    # Load nucleus centroids either from SpatialData or from CSV
    if sdata is not None:
        print("[DEBUG] Loading nucleus centroids from SpatialData object")
        xenium_centroids, cell_ids = extract_xenium_nucleus_centroids_from_spatialdata(sdata)
        print(f"[DEBUG] Extracted {len(xenium_centroids)} nucleus centroids from SpatialData object")
    elif xenium_data_path is not None:
        print(f"[DEBUG] Loading Xenium data from CSV at {xenium_data_path}")
        xenium_data = pd.read_csv(xenium_data_path)
        xenium_centroids = xenium_data[['x_centroid', 'y_centroid']].values
        cell_ids = xenium_data['cell_id'].values if 'cell_id' in xenium_data.columns else np.arange(len(xenium_centroids))
        print(f"[DEBUG] Loaded {len(xenium_centroids)} nucleus centroids from CSV file")
    else:
        raise ValueError("Either sdata or xenium_data_path must be provided")

    # Load H&E image
    print(f"[DEBUG] Reading H&E image from {he_image_path}")
    he_image = io.imread(he_image_path)
    print(f"[DEBUG] Loaded H&E image with shape {he_image.shape}")

    # If matching points are provided, use them to estimate the transform
    if matching_points_path:
        print(f"[DEBUG] Loading matching points from {matching_points_path}")
        matching_points = pd.read_csv(matching_points_path)
        xenium_points = matching_points[['xenium_x', 'xenium_y']].values
        he_points = matching_points[['he_x', 'he_y']].values
        transform_matrix = estimate_affine_transform(xenium_points, he_points)
        print("[DEBUG] Estimated affine transformation matrix from matching points")
    else:
        print("[DEBUG] No matching points provided. Using identity transform for demonstration.")
        transform_matrix = transform.AffineTransform()

    # Transform Xenium centroids to H&E space
    he_coordinates = apply_inverse_affine_transform(xenium_centroids, transform_matrix)
    print(f"[DEBUG] Transformed {len(he_coordinates)} Xenium centroids to H&E space")

    # Extract patches at multiple zoom levels
    patches = extract_image_patches(he_image, he_coordinates)
    print(f"[DEBUG] Extracted patches for {len(patches)} valid coordinates")

    # Visualize the overlay
    visualize_overlay(he_image, xenium_centroids, he_coordinates)
    print("[DEBUG] Completed main function")

    return xenium_centroids, he_coordinates, patches, transform_matrix, cell_ids


def example_usage_with_spatialdata():
    """
    Example of how to use the code with a SpatialData object.
    """
    print("[DEBUG] Entering example_usage_with_spatialdata")
    xenium_path = "/Users/jianzhouyao/Cancer/FFPE_Human_Breast_with_Pre-designed_Panel/Xenium_V1_FFPE_Human_Breast_IDC_outs"
    zarr_path = xenium_path + ".zarr"

    try:
        print(f"[DEBUG] Trying to load SpatialData from {zarr_path}")
        sdata = sd.read_zarr(zarr_path)
        print(f"[DEBUG] Loaded SpatialData from {zarr_path}")
    except (FileNotFoundError, ValueError):
        print(f"[DEBUG] Zarr not found or invalid. Reading Xenium data from {xenium_path}")
        sdata = xenium(xenium_path, cells_as_circles=True)
        print(f"[DEBUG] Writing SpatialData to {zarr_path}")
        sdata.write(zarr_path)

    he_image_path = "path/to/he_image.tif"
    matching_points_path = "path/to/matching_points.csv"

    xenium_centroids, he_coordinates, patches, transform_matrix, cell_ids = main(
        he_image_path=he_image_path,
        matching_points_path=matching_points_path,
        sdata=sdata
    )

    print("[DEBUG] Exiting example_usage_with_spatialdata")
    return xenium_centroids, he_coordinates, patches, transform_matrix, cell_ids, sdata

if __name__ == "__main__":
    print("[DEBUG] Script __main__ entry point")
    # Example usages commented out below
    # xenium_centroids, he_coordinates, patches, transform_matrix, cell_ids = main(
    #     xenium_data_path="path/to/xenium_data.csv",
    #     he_image_path="path/to/he_image.tif",
    #     matching_points_path="path/to/matching_points.csv"
    # )

    # Uncomment to run with SpatialData example
    # example_usage_with_spatialdata()
    pass