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
    from spatialdata_io import xenium
    SPATIALDATA_AVAILABLE = True
except ImportError:
    print("SpatialData not available, proceeding without it.")
    SPATIALDATA_AVAILABLE = False


class XeniumHEAligner:
    def __init__(self, he_path, xenium_path, output_dir=None, scale_factor=1.0):
        self.he_path = Path(he_path)
        self.xenium_path = Path(xenium_path)
        self.zarr_path = Path(xenium_path + ".zarr")  # Define zarr_path
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
        # Transpose from (C, H, W) to (H, W, C) if needed
        if self.he_image.ndim == 3 and self.he_image.shape[0] == 3:
            self.he_image = self.he_image.transpose(1, 2, 0)
            print("Transposed H&E image to (H, W, C) format")

        if self.he_image.ndim == 3 and self.he_image.shape[2] > 3:
            self.he_image = self.he_image[:, :, :3]
        print(f"H&E image shape: {self.he_image.shape}")

        if sdata_external is not None:
            self.sdata = sdata_external
        elif SPATIALDATA_AVAILABLE:
            try:
                if self.zarr_path.exists():
                    print(f"Loading Xenium data from existing Zarr store: {self.zarr_path}")
                    self.sdata = sd.read_zarr(self.zarr_path)
                    print("SpatialData object loaded from Zarr.")
                else:
                    print(f"Zarr store not found. Creating Zarr from: {self.xenium_path}")
                    try:
                         self.sdata = xenium(self.xenium_path)
                         sd.write_zarr(self.sdata, self.zarr_path)
                         self.sdata = sd.read_zarr(self.zarr_path)  # Load after writing
                         print("SpatialData object created and loaded from new Zarr.")
                    except Exception as e:
                         print(f"Error creating zarr file: {e}")
                         self._load_xenium_fallback()
                         return
                print(self.sdata)  # Print sdata for inspection

            except Exception as e:
                print(f"Error loading Xenium data: {e}")
                self._load_xenium_fallback()
                return
        else:
            self._load_xenium_fallback()
            return

        if hasattr(self.sdata, 'tables') and 'table' in self.sdata.tables:
            self.adata = self.sdata.tables['table']
            if 'spatial' in self.adata.obsm:
                self.centroids = self.adata.obsm['spatial']
                print(f"Loaded {len(self.centroids)} centroids from AnnData.")
            else:
                print("Spatial coordinates not found in AnnData.")
                self._load_xenium_fallback()
                return
        else:
            print("No 'table' found in SpatialData.")
            self._load_xenium_fallback()
            return

        if hasattr(self.sdata, 'images'):
            if 'morphology_focus' in self.sdata.images:
                self.xenium_dapi = self.sdata.images['morphology_focus'].compute().values
                print(f"Loaded Xenium morphology image (morphology_focus) with shape: {self.xenium_dapi.shape}")
            elif 'morphology_mip' in self.sdata.images:
                self.xenium_dapi = self.sdata.images['morphology_mip'].compute().values
                print(f"Loaded Xenium morphology image (morphology_mip) with shape: {self.xenium_dapi.shape}")
            elif 'morphology_dapi' in self.sdata.images:
                self.xenium_dapi = self.sdata.images['morphology_dapi'].compute().values
                print(f"Loaded Xenium morphology image (morphology_dapi) with shape: {self.xenium_dapi.shape}")
            else:
                print("No suitable morphology image found in SpatialData images.")
                self._load_xenium_fallback()
                return
        else:
            print("No 'images' found in SpatialData.")
            self._load_xenium_fallback()
            return

    def _load_xenium_fallback(self):
        print("Falling back to direct Xenium data loading...")
        dapi_candidates = list(self.xenium_path.glob("**/morphology_dapi.ome.tif"))
        if not dapi_candidates:
            dapi_candidates = list(self.xenium_path.glob("**/morphology_focus.ome.tif"))
        if dapi_candidates:
            self.xenium_dapi = tifffile.imread(dapi_candidates[0])
            print(f"Loaded DAPI image from: {dapi_candidates[0]}")
        else:
            print("No DAPI image found.")

        centroid_candidates = list(self.xenium_path.glob("**/cells.parquet"))
        if not centroid_candidates:
            centroid_candidates = list(self.xenium_path.glob("**/cells.csv"))
        if centroid_candidates:
            file = centroid_candidates[0]
            df = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
            for x_col in ["x_centroid", "x", "center_x"]:
                for y_col in ["y_centroid", "y", "center_y"]:
                    if x_col in df.columns and y_col in df.columns:
                        self.centroids = df[[x_col, y_col]].values
                        print(f"Using centroid columns: {x_col}, {y_col} from {file}")
                        return
            print(f"Centroid columns not found in {file}")
        else:
            print("No centroid file found.")

    def preprocess_images(self):
        if self.he_image is None or self.xenium_dapi is None:
            print("Error: Images not loaded. Call load_data() first.")
            return None, None

        print(f"Preprocessing images with scale factor {self.scale_factor}...")

        # Convert H&E to grayscale
        he_gray = color.rgb2gray(self.he_image) if self.he_image.ndim == 3 else self.he_image.astype(float)
        he_gray = exposure.equalize_adapthist(he_gray)

        # Process Xenium DAPI image
        dapi_gray = self.xenium_dapi.astype(float)
        dapi_gray = exposure.rescale_intensity(dapi_gray, in_range='image', out_range=(0, 1))
        dapi_gray = exposure.equalize_adapthist(dapi_gray)

        # Median filtering
        he_gray = filters.median(he_gray)
        dapi_gray = filters.median(dapi_gray)

        # Downsample both images
        if self.scale_factor < 1.0:
            he_gray = cv2.resize(he_gray, (0, 0), fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_AREA)
            dapi_gray = cv2.resize(dapi_gray, (0, 0), fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_AREA)
            print(f"Downsampled shapes: H&E = {he_gray.shape}, DAPI = {dapi_gray.shape}")

            # Also scale centroids if needed
            if self.centroids is not None:
                self.centroids *= self.scale_factor
                print("Scaled centroids to match downsampled image.")

        self.he_processed = he_gray
        self.xenium_processed = dapi_gray
        return he_gray, dapi_gray

    def register_feature_based(self):
        if self.he_processed is None or self.xenium_processed is None:
            print("Error: Preprocessed images not available.")
            return None

        print("Starting feature-based registration...")

        img_xenium = (self.xenium_processed * 255).astype(np.uint8)
        img_he = (self.he_processed * 255).astype(np.uint8)

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img_xenium, None)
        kp2, des2 = orb.detectAndCompute(img_he, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print("Not enough features detected.")
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(matches) < 4:
            print("Not enough matches for homography.")
            return None

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        self.transform_matrix = h
        self.transform_name = "homography"

        match_img = cv2.drawMatches(img_xenium, kp1, img_he, kp2, matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(match_img)
        plt.title("ORB Feature Matches (Xenium to H&E)")
        plt.axis('off')
        plt.savefig(self.output_dir / "orb_matches.png", dpi=300)
        plt.close()

        print("Feature-based registration complete.")
        return h

    def transform_centroids(self):
        if self.transform_matrix is None or self.centroids is None:
            print("Error: Missing transform or centroids.")
            return None

        print("Transforming centroids...")
        n = self.centroids.shape[0]
        pts = np.hstack([self.centroids, np.ones((n, 1))])
        transformed = (self.transform_matrix @ pts.T).T
        transformed /= transformed[:, 2][:, None]
        self.aligned_centroids = transformed[:, :2]
        return self.aligned_centroids

    def visualize_alignment(self, n_points=1000, save_only=False):
        if self.he_image is None or self.aligned_centroids is None:
            print("Missing data for visualization.")
            return

        print("Visualizing alignment...")
        indices = np.random.choice(len(self.aligned_centroids),
                                   min(n_points, len(self.aligned_centroids)),
                                   replace=False)
        sampled = self.aligned_centroids[indices]

        plt.figure(figsize=(10, 10))
        plt.imshow(self.he_image if self.he_image.ndim == 3 else self.he_image, cmap='gray')
        plt.scatter(sampled[:, 0], sampled[:, 1], s=5, c='red', alpha=0.5)
        plt.title("Aligned Xenium Cell Centroids on H&E")
        plt.axis('off')

        # Save the figure
        save_path = self.output_dir / "aligned_centroids.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved alignment visualization to {save_path}")

        if not save_only:
            plt.show()
        else:
            plt.close()


if __name__ == '__main__':
    he_path = "/Users/jianzhouyao/Cancer/FFPE Human Breast with Pre-designed Panel/Xenium_V1_FFPE_Human_Breast_IDC_he_image.ome.tif"
    xenium_path = "/Users/jianzhouyao/Cancer/FFPE Human Breast with Pre-designed Panel/Xenium_V1_FFPE_Human_Breast_IDC_outs"
    output_dir = "./output_alignment"
    scale_factor = 0.1

    aligner = XeniumHEAligner(he_path, xenium_path, output_dir, scale_factor)
    aligner.load_data()

    if aligner.he_image is not None and aligner.xenium_dapi is not None:
        aligner.preprocess_images()
        transform = aligner.register_feature_based()
        if transform is not None and aligner.centroids is not None:
            aligned_centroids = aligner.transform_centroids()
            aligner.visualize_alignment(save_only=False)
