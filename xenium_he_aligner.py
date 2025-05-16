import os
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile
from skimage import exposure, color, filters
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rescale

# Optional spatialdata support
try:
    import spatialdata as sd
    from spatialdata_io import xenium # Ensure xenium reader is available if you use it
    SPATIALDATA_AVAILABLE = True
except ImportError:
    print("SpatialData not available, proceeding without it.")
    SPATIALDATA_AVAILABLE = False


class XeniumHEAligner:
    def __init__(self, he_path, xenium_path, output_dir=None, scale_factor=1.0):
        self.he_path = Path(he_path)
        self.xenium_path = Path(xenium_path)
        # Assuming xenium_path is a directory like '.../Xenium_V1_FFPE_Human_Breast_IDC_outs'
        # and the Zarr store is named '.../Xenium_V1_FFPE_Human_Breast_IDC_outs.zarr'
        self.zarr_path = self.xenium_path.with_suffix(self.xenium_path.suffix + ".zarr")
        self.output_dir = Path(output_dir or "./output")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.scale_factor = scale_factor

        self.he_image = None
        self.xenium_dapi = None
        self.centroids = None
        self.he_processed = None
        self.xenium_processed = None
        self.transform_matrix = None
        self.transform_name = None
        self.aligned_centroids = None
        self.sdata = None
        self.adata = None

        print(f"Initialized with scale factor {self.scale_factor}\n- H&E image path: {self.he_path}\n- Xenium data path: {self.xenium_path}\n- Zarr path: {self.zarr_path}")

    def load_data(self, sdata_external=None):
        print("Loading H&E image...")
        self.he_image = tifffile.imread(self.he_path)
        # Transpose from (C, H, W) to (H, W, C) if needed for standard image libraries
        if self.he_image.ndim == 3 and self.he_image.shape[0] in [3, 4]: # Check if first dim is channel
             if self.he_image.shape[2] not in [3,4]: # Avoid transposing if already HWC
                self.he_image = self.he_image.transpose(1, 2, 0)
                print("Transposed H&E image to (H, W, C) format")

        # Keep only RGB, remove alpha if present for simplicity in rgb2gray
        if self.he_image.ndim == 3 and self.he_image.shape[2] == 4:
            self.he_image = self.he_image[:, :, :3]
            print("Removed alpha channel from H&E image.")
        print(f"H&E image shape: {self.he_image.shape}")

        if sdata_external is not None:
            self.sdata = sdata_external
            print("Using externally provided SpatialData object.")
        elif SPATIALDATA_AVAILABLE:
            try:
                if self.zarr_path.exists():
                    print(f"Loading Xenium data from existing Zarr store: {self.zarr_path}")
                    self.sdata = sd.read_zarr(self.zarr_path)
                    print("SpatialData object loaded from Zarr.")
                else:
                    print(f"Zarr store not found at {self.zarr_path}. Creating Zarr from: {self.xenium_path}")
                    # Check if the path for xenium reader is the base Xenium output directory
                    if not (self.xenium_path / "experiment.xenium").exists() and not (self.xenium_path / "run_info.json").exists(): # Basic check for Xenium output
                        print(f"Warning: {self.xenium_path} might not be a valid Xenium output directory for spatialdata_io.xenium reader.")
                    
                    # Assuming xenium() can take the base output directory
                    self.sdata = xenium(self.xenium_path) 
                    sd.write_zarr(self.sdata, self.zarr_path)
                    # It's good practice to read back after writing to ensure consistency or if write_zarr doesn't return the sdata object in desired state
                    self.sdata = sd.read_zarr(self.zarr_path) 
                    print("SpatialData object created, written to Zarr, and reloaded.")
                
                print("--- SpatialData Object Structure ---")
                print(self.sdata)
                print("------------------------------------")


            except Exception as e:
                print(f"Error loading or creating Xenium SpatialData: {e}")
                self._load_xenium_fallback()
                return # Important to return after fallback
        else:
            self._load_xenium_fallback()
            return # Important to return after fallback

        # Proceed with sdata if loaded successfully
        if self.sdata is None: # Should be caught by returns above, but as a safeguard
             print("SpatialData object is None, cannot proceed with sdata dependent loading.")
             self._load_xenium_fallback()
             return

        # Load AnnData table for centroids
        if hasattr(self.sdata, 'tables') and 'table' in self.sdata.tables:
            self.adata = self.sdata.tables['table']
            if 'spatial' in self.adata.obsm:
                self.centroids = self.adata.obsm['spatial']
                # AnnData often stores as float32, ensure it's a writeable numpy array for scaling later
                self.centroids = np.array(self.centroids, dtype=np.float64) 
                print(f"Loaded {len(self.centroids)} centroids from AnnData table.")
            else:
                print("Spatial coordinates (obsm['spatial']) not found in AnnData table. Attempting fallback for centroids.")
                # If centroids are not in table, Xenium standard often has them in shapes too.
                # Or, one might load from a separate file if _load_xenium_fallback for centroids is desired.
                # For now, let's assume if not in table, it might be an issue or require fallback for DAPI too.
                self._load_xenium_fallback() # This will try to load centroids from parquet/csv
                # If fallback loads centroids, self.centroids will be updated.
                # Now, try to load DAPI from sdata if available, otherwise fallback already handled DAPI.
                if self.xenium_dapi is None: # If fallback didn't set DAPI, try sdata images
                    self._load_dapi_from_sdata() 
                return # Exit load_data after this path
        else:
            print("No 'table' found in SpatialData.tables. Attempting fallback.")
            self._load_xenium_fallback()
            return

        # Load DAPI from sdata images if not already loaded by a fallback path that also loaded centroids
        if self.xenium_dapi is None: # Check if DAPI needs to be loaded
            self._load_dapi_from_sdata()


    def _load_dapi_from_sdata(self):
        """Helper to load DAPI from self.sdata.images"""
        if not hasattr(self.sdata, 'images') or not self.sdata.images:
            print("No 'images' found in SpatialData or images are empty. Attempting fallback for DAPI.")
            self._load_xenium_fallback_dapi_only() # A more targeted fallback
            return

        selected_image_key = None
        if 'morphology_focus' in self.sdata.images:
            selected_image_key = 'morphology_focus'
        elif 'morphology_mip' in self.sdata.images:
            selected_image_key = 'morphology_mip'
        # Add other potential DAPI keys if necessary, e.g., 'morphology_dapi'
        elif 'morphology_dapi' in self.sdata.images: # As in original code
            selected_image_key = 'morphology_dapi'

        if selected_image_key:
            try:
                print(f"Attempting to load '{selected_image_key}' from SpatialData images...")
                image_sdata_element = self.sdata.images[selected_image_key]  # This is a DataTree

                # Access the highest resolution scale (usually 'scale0')
                # Then access the DataArray within the Dataset (usually named same as selected_image_key)
                # The DataArray is typically under DataTree -> Dataset (e.g. 'scale0') -> DataArray (e.g. 'morphology_focus')
                data_array = image_sdata_element['scale0'][selected_image_key]
                
                numpy_array = data_array.compute().values
                
                # Images are stored as CYX. If it's a single channel DAPI, squeeze the channel dim.
                if numpy_array.ndim == 3 and numpy_array.shape[0] == 1:  # CYX with C=1
                    self.xenium_dapi = numpy_array.squeeze(axis=0)
                elif numpy_array.ndim == 2:  # YX (already squeezed or no channel dim)
                    self.xenium_dapi = numpy_array
                elif numpy_array.ndim == 3 and numpy_array.shape[0] > 1: # Multi-channel CYX
                    print(f"Warning: Loaded {selected_image_key} is multi-channel with shape {numpy_array.shape}. Taking first channel as DAPI.")
                    self.xenium_dapi = numpy_array[0] # Take the first channel
                else:
                    print(f"Warning: Loaded {selected_image_key} has an unexpected shape {numpy_array.shape}. Using as is.")
                    self.xenium_dapi = numpy_array

                print(f"Loaded Xenium morphology image ({selected_image_key}) from sdata. Original DataArray shape: {data_array.shape}, processed to numpy shape: {self.xenium_dapi.shape}")

            except KeyError as e:
                print(f"Error accessing image data for {selected_image_key} in SpatialData (e.g., 'scale0' or DataArray name might be different, or key not found): {e}")
                self._load_xenium_fallback_dapi_only()
            except Exception as e:
                print(f"An unexpected error occurred while loading {selected_image_key} from sdata: {e}")
                self._load_xenium_fallback_dapi_only()
        else:
            print("No suitable morphology image key ('morphology_focus', 'morphology_mip', 'morphology_dapi') found in SpatialData images.")
            self._load_xenium_fallback_dapi_only()

    def _load_xenium_fallback_dapi_only(self):
        """Fallback to load only DAPI if other sdata parts were successful."""
        print("Falling back to direct DAPI image file loading...")
        dapi_candidates = list(self.xenium_path.glob("**/morphology_dapi.ome.tif"))
        if not dapi_candidates:
            dapi_candidates = list(self.xenium_path.glob("**/morphology_focus.ome.tif")) # Common alternative name
        
        if dapi_candidates:
            dapi_file_path = dapi_candidates[0]
            try:
                dapi_img = tifffile.imread(dapi_file_path)
                # Ensure DAPI is 2D (H, W)
                if dapi_img.ndim == 3:
                    if dapi_img.shape[0] == 1: # (1, H, W)
                        dapi_img = dapi_img.squeeze(axis=0)
                    elif dapi_img.shape[-1] == 1: # (H, W, 1)
                        dapi_img = dapi_img.squeeze(axis=-1)
                    else: # Potentially (C, H, W) or (H, W, C) with C > 1
                        print(f"Warning: Fallback DAPI image {dapi_file_path} has shape {dapi_img.shape}. Taking first channel/slice.")
                        dapi_img = dapi_img[0] if dapi_img.shape[0] < dapi_img.shape[-1] else dapi_img[..., 0] # Heuristic
                
                self.xenium_dapi = dapi_img
                print(f"Loaded DAPI image from fallback: {dapi_file_path}, shape: {self.xenium_dapi.shape}")
            except Exception as e:
                print(f"Error loading DAPI image from {dapi_file_path}: {e}")
                self.xenium_dapi = None # Ensure it's None if loading fails
        else:
            print("No DAPI image file (morphology_dapi.ome.tif or morphology_focus.ome.tif) found in fallback paths.")
            self.xenium_dapi = None


    def _load_xenium_fallback(self):
        print("Falling back to direct Xenium file loading (DAPI and centroids)...")
        self._load_xenium_fallback_dapi_only() # Load DAPI using the specific helper

        # Load centroids from parquet/csv
        centroid_candidates = list(self.xenium_path.glob("**/cells.parquet"))
        if not centroid_candidates:
            centroid_candidates = list(self.xenium_path.glob("**/cells.csv"))
        
        if centroid_candidates:
            file_path = centroid_candidates[0]
            try:
                df = pd.read_parquet(file_path) if file_path.suffix == ".parquet" else pd.read_csv(file_path)
                # Common column names for Xenium centroids
                coord_cols = None
                if "x_centroid" in df.columns and "y_centroid" in df.columns:
                    coord_cols = ["x_centroid", "y_centroid"]
                elif "vertex_x" in df.columns and "vertex_y" in df.columns: # For some Xenium outputs (cell boundaries)
                    # If using vertices, one might need to calculate centroids
                    print(f"Found vertex coordinates in {file_path}. Consider calculating centroids if these are not already cell centers.")
                    # For simplicity, if these are the only spatial coords, use them, but note they might not be centroids.
                    # A groupby cell_id and mean of vertices could give centroids.
                    # This example will just take them as is if no 'x_centroid' exists.
                    if "cell_id" in df.columns: # A crude way to get one point per cell
                        df_centroids = df.groupby("cell_id")[["vertex_x", "vertex_y"]].mean().reset_index()
                        self.centroids = df_centroids[["vertex_x", "vertex_y"]].values
                        print(f"Calculated centroids from vertices in {file_path}. Found {len(self.centroids)} centroids.")
                    else: # Cannot easily get centroids from raw vertices without cell_id
                         print(f"Vertex columns found in {file_path} but no cell_id for grouping. Cannot derive centroids easily.")
                         self.centroids = None

                if coord_cols and self.centroids is None: # If not already set by vertex logic
                    self.centroids = df[coord_cols].values
                    print(f"Loaded {len(self.centroids)} centroids using columns {coord_cols} from {file_path}.")
                
                if self.centroids is None and not coord_cols : # If no standard centroid/vertex columns found
                    print(f"Standard centroid/vertex columns not found in {file_path}.")

            except Exception as e:
                print(f"Error loading centroid file {file_path}: {e}")
                self.centroids = None
        else:
            print("No centroid file (cells.parquet or cells.csv) found in fallback paths.")
            self.centroids = None
        
        if self.centroids is not None:
             self.centroids = np.array(self.centroids, dtype=np.float64)


    def preprocess_images(self):
        if self.he_image is None:
            print("Error: H&E image not loaded. Call load_data() first.")
            return None, None
        if self.xenium_dapi is None:
            print("Error: Xenium DAPI image not loaded. Call load_data() first.")
            return None, None

        print(f"Preprocessing images with scale factor {self.scale_factor}...")

        # Convert H&E to grayscale
        # Ensure H&E is 2D or 3D (RGB) before rgb2gray
        if self.he_image.ndim == 3 and self.he_image.shape[-1] == 3: # HWC (RGB)
            he_gray = color.rgb2gray(self.he_image)
        elif self.he_image.ndim == 2: # Already grayscale
            he_gray = self.he_image.astype(float)
            if np.issubdtype(self.he_image.dtype, np.integer): # Rescale if integer
                 he_gray = he_gray / np.iinfo(self.he_image.dtype).max if np.max(he_gray) > 1 else he_gray
        else:
            print(f"Error: H&E image has unexpected shape {self.he_image.shape} for grayscale conversion.")
            return None, None
        
        he_gray = exposure.equalize_adapthist(he_gray) #CLAHE

        # Process Xenium DAPI image (should be 2D (H,W) by now)
        if self.xenium_dapi.ndim != 2:
            print(f"Error: Xenium DAPI image is not 2D (shape: {self.xenium_dapi.shape}) before processing.")
            return None, None

        dapi_gray = self.xenium_dapi.astype(float)
        # Rescale intensity if not already in [0,1] range
        if np.max(dapi_gray) > 1.0 or np.min(dapi_gray) < 0.0:
             dapi_gray = exposure.rescale_intensity(dapi_gray, in_range='image', out_range=(0, 1))
        dapi_gray = exposure.equalize_adapthist(dapi_gray) #CLAHE

        # Median filtering
        # skimage.filters.median is generally robust
        he_gray = filters.median(he_gray)
        dapi_gray = filters.median(dapi_gray)

        # Downsample both images
        # Ensure centroids are also scaled if scale_factor is used for images
        original_centroids = None
        if self.centroids is not None:
            original_centroids = self.centroids.copy() # Keep original if scaling is only for registration image

        if self.scale_factor != 1.0: # Apply scaling only if not 1.0
            print(f"Downsampling images by factor {self.scale_factor}...")
            # cv2.resize expects (width, height) for dsize
            he_target_shape = (int(he_gray.shape[1] * self.scale_factor), int(he_gray.shape[0] * self.scale_factor))
            dapi_target_shape = (int(dapi_gray.shape[1] * self.scale_factor), int(dapi_gray.shape[0] * self.scale_factor))

            he_gray_scaled = cv2.resize(he_gray, he_target_shape, interpolation=cv2.INTER_AREA)
            dapi_gray_scaled = cv2.resize(dapi_gray, dapi_target_shape, interpolation=cv2.INTER_AREA)
            print(f"Downsampled shapes: H&E = {he_gray_scaled.shape}, DAPI = {dapi_gray_scaled.shape}")

            self.he_processed = he_gray_scaled
            self.xenium_processed = dapi_gray_scaled

            # Scale centroids for registration if they exist and scale_factor is used for images
            # These scaled centroids are used for finding the transformation
            if self.centroids is not None:
                self.centroids_scaled_for_registration = original_centroids * self.scale_factor
                print("Scaled centroids for registration to match downsampled image.")
            else:
                self.centroids_scaled_for_registration = None

        else: # No scaling
            self.he_processed = he_gray
            self.xenium_processed = dapi_gray
            if self.centroids is not None:
                self.centroids_scaled_for_registration = original_centroids # Use original if no scaling
            else:
                self.centroids_scaled_for_registration = None
            print(f"Processed image shapes (no downsampling): H&E = {self.he_processed.shape}, DAPI = {self.xenium_processed.shape}")


        return self.he_processed, self.xenium_processed

    def register_feature_based(self):
        if self.he_processed is None or self.xenium_processed is None:
            print("Error: Preprocessed images not available. Call preprocess_images() first.")
            return None

        print("Starting feature-based registration (ORB)...")

        # Ensure images are uint8 for ORB
        img_xenium = cv2.normalize(self.xenium_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_he = cv2.normalize(self.he_processed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_HARRIS_SCORE) # Example: trying HARRIS_SCORE
        kp1, des1 = orb.detectAndCompute(img_xenium, None)
        kp2, des2 = orb.detectAndCompute(img_he, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10: # Increased minimum features
            print("Not enough features detected in one or both images for ORB.")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if not matches:
            print("No matches found between features.")
            return None

        # Sort matches by distance and take top N (e.g., 50 or more if available)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches_count = min(len(matches), 100) # Use up to 100 good matches
        matches_to_use = matches[:good_matches_count]

        if len(matches_to_use) < 4: # Need at least 4 matches for homography
            print(f"Not enough good matches ({len(matches_to_use)}) for homography estimation.")
            return None
        
        print(f"Found {len(matches)} total matches, using top {len(matches_to_use)} for homography.")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_to_use]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_to_use]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        
        if h is None:
            print("Homography estimation failed.")
            return None

        self.transform_matrix = h
        self.transform_name = "homography_ORB" # More specific name

        # Draw matches for visualization
        # Ensure mask is correctly applied if you want to draw only inliers
        # For drawMatches, it usually takes the list of matches directly.
        # If you want to show inliers, you might need to filter 'matches_to_use' based on 'mask'.
        # However, cv2.drawMatches is often used with the raw matches list.
        match_img = cv2.drawMatches(img_xenium, kp1, img_he, kp2, matches_to_use, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(16, 8)) # Increased size for better visibility
        plt.imshow(match_img)
        plt.title(f"ORB Feature Matches (Xenium to H&E) - Top {len(matches_to_use)} Matches")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "orb_matches.png", dpi=150) # Adjust dpi as needed
        plt.close()

        print("Feature-based registration (ORB) complete. Transform matrix stored.")
        return h

    def transform_centroids(self):
        if self.transform_matrix is None:
            print("Error: Transform matrix not available. Run registration first.")
            return None
        if self.centroids is None: # Use the original, unscaled centroids for final transformation
            print("Error: Original centroids not available.")
            return None

        print("Transforming original centroids using the calculated transformation matrix...")
        n = self.centroids.shape[0]
        
        # Centroids are (x, y). Homography expects (x, y, 1).
        # Note: The transformation was found on scaled images.
        # The transformation matrix 'h' maps points from the scaled Xenium space
        # to the scaled H&E space.
        # To apply this to original centroids:
        # 1. Scale original centroids down (if scale_factor != 1.0).
        # 2. Apply homography.
        # 3. Scale transformed centroids back up to original H&E image resolution (if scale_factor != 1.0).
        
        # If self.centroids_scaled_for_registration was used to find 'h':
        # This means 'h' maps from the space of self.centroids_scaled_for_registration
        # to the space of self.he_processed.
        
        # Let P_orig be original Xenium centroids.
        # P_scaled_xen = P_orig * scale_factor
        # H maps P_scaled_xen to P_scaled_he
        # P_scaled_he = H * P_scaled_xen
        # To get P_final_he (centroids on original H&E image):
        # P_final_he = P_scaled_he / scale_factor
        
        # So, if transform_matrix (h) was derived from scaled images:
        # 1. Scale original Xenium centroids by self.scale_factor
        temp_scaled_centroids = self.centroids * self.scale_factor
        
        # 2. Apply the homography matrix
        pts_to_transform = np.hstack([temp_scaled_centroids, np.ones((n, 1))]) # Add homogeneous coordinate
        transformed_pts_homog = (self.transform_matrix @ pts_to_transform.T).T
        
        # Normalize homogeneous coordinates
        transformed_pts_scaled = transformed_pts_homog[:, :2] / transformed_pts_homog[:, 2][:, None]
        
        # 3. Rescale back to the original H&E image's coordinate system
        if self.scale_factor != 1.0:
            self.aligned_centroids = transformed_pts_scaled / self.scale_factor
        else:
            self.aligned_centroids = transformed_pts_scaled

        print(f"Transformed {len(self.aligned_centroids)} centroids.")
        return self.aligned_centroids


    def visualize_alignment(self, n_points=1000, save_only=False, point_size=1, point_alpha=0.5):
        if self.he_image is None:
            print("H&E image not loaded for visualization.")
            return
        if self.aligned_centroids is None:
            print("Aligned centroids not available for visualization.")
            return
            
        # Use the full resolution H&E image for visualization
        he_display_img = self.he_image
        if he_display_img.ndim == 3 and he_display_img.shape[-1] == 3 and np.issubdtype(he_display_img.dtype, np.floating):
            # Potentially normalize if it's float [0,1] for imshow, or let matplotlib handle it
            pass
        elif np.issubdtype(he_display_img.dtype, np.integer):
            # Ensure it's in a displayable range if integer, though matplotlib often handles this.
            pass


        print(f"Visualizing alignment of {n_points} sampled centroids...")
        num_centroids_to_sample = min(n_points, len(self.aligned_centroids))
        if len(self.aligned_centroids) > 0:
            indices = np.random.choice(len(self.aligned_centroids),
                                   num_centroids_to_sample,
                                   replace=False)
            sampled_centroids = self.aligned_centroids[indices]
        else:
            print("No aligned centroids to visualize.")
            sampled_centroids = np.array([])


        plt.figure(figsize=(12, 12)) # Larger figure for detail
        plt.imshow(he_display_img, cmap='gray' if he_display_img.ndim==2 else None)
        
        if sampled_centroids.any():
            plt.scatter(sampled_centroids[:, 0], sampled_centroids[:, 1], 
                        s=point_size, c='red', alpha=point_alpha, edgecolors='none') # No distracting edges

        plt.title(f"Aligned Xenium Cell Centroids on H&E Image (Sampled {num_centroids_to_sample} Points)")
        plt.axis('on') # Show axes to verify coordinates if needed, or 'off' for cleaner image
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.gca().invert_yaxis() # Often needed for image coordinate systems
        plt.tight_layout()

        save_path = self.output_dir / "aligned_centroids_visualization.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved alignment visualization to {save_path}")

        if not save_only:
            plt.show()
        else:
            plt.close()

# Main execution block (if you want to run this script directly)
if __name__ == '__main__':
    # Get USER from environment or default to 'user'
    user = os.getenv('USER', 'user') # For cross-platform compatibility, USER might not always be set.
                                     # Consider getpass.getuser() if available and more robust.
    
    # Example paths - replace with your actual paths
    base_cancer_path = Path(f"/Users/{user}/Cancer/FFPE Human Breast with Pre-designed Panel") # Adjust if path varies
    
    # Check if the base path exists
    if not base_cancer_path.exists():
        print(f"Error: Base data path {base_cancer_path} does not exist. Please check the path.")
        exit()

    he_path = base_cancer_path / "Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif"
    xenium_data_dir = base_cancer_path / "Xenium_V1_FFPE_Human_Breast_IDC_outs" # This is the directory for Xenium outputs

    # Check if specific files/dirs exist
    if not he_path.exists():
        print(f"Error: H&E image not found at {he_path}")
        exit()
    if not xenium_data_dir.exists() or not xenium_data_dir.is_dir():
        print(f"Error: Xenium data directory not found at {xenium_data_dir}")
        exit()

    output_dir = Path("./alignment_output_breast_idc_corrected") # Changed output dir name
    scale_factor_for_registration = 0.1 # Use a descriptive name

    print("Starting Xenium-H&E Alignment Pipeline...")
    aligner = XeniumHEAligner(he_path, xenium_data_dir, output_dir, scale_factor=scale_factor_for_registration)
    
    print("\nStep 1: Loading data...")
    aligner.load_data()

    # Check if data loading was successful before proceeding
    if aligner.he_image is None or aligner.xenium_dapi is None:
        print("\nData loading failed. H&E or Xenium DAPI image is missing. Exiting pipeline.")
    elif aligner.centroids is None:
        print("\nData loading failed or centroids are missing. Exiting pipeline.")
    else:
        print("\nData loading successful.")
        print(f"H&E image loaded with shape: {aligner.he_image.shape}")
        print(f"Xenium DAPI image loaded with shape: {aligner.xenium_dapi.shape}")
        print(f"Centroids loaded: {len(aligner.centroids)} points")

        print("\nStep 2: Preprocessing images...")
        processed_he, processed_dapi = aligner.preprocess_images()
        if processed_he is None or processed_dapi is None:
            print("\nImage preprocessing failed. Exiting pipeline.")
        else:
            print("\nImage preprocessing successful.")
            print(f"Processed H&E shape: {processed_he.shape}")
            print(f"Processed DAPI shape: {processed_dapi.shape}")

            print("\nStep 3: Registering images (feature-based)...")
            transform_matrix = aligner.register_feature_based()
            if transform_matrix is None:
                print("\nImage registration failed. Exiting pipeline.")
            else:
                print("\nImage registration successful. Transform matrix:")
                print(transform_matrix)

                print("\nStep 4: Transforming centroids...")
                aligned_centroids = aligner.transform_centroids()
                if aligned_centroids is None:
                    print("\nCentroid transformation failed.")
                else:
                    print(f"\nCentroid transformation successful. {len(aligned_centroids)} centroids aligned.")
                    
                    print("\nStep 5: Visualizing alignment...")
                    aligner.visualize_alignment(n_points=2000, save_only=False, point_size=2, point_alpha=0.6)
                    print("\nAlignment visualization complete.")
    
    print("\nXenium-H&E Alignment Pipeline Finished.")