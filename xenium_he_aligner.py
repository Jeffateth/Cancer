import os
import numpy as np
import napari
from spatialdata.models import ShapesModel
from spatialdata.transformations import (
    Identity, 
    get_transformation_between_landmarks, 
    align_elements_using_landmarks
)

def align_xenium_to_he(sdata, he_image, centroids_um, psx, psy, xenium_dir):
    """
    Align Xenium coordinates to H&E image using an existing transformation 
    or manual landmark selection if needed.
    
    Parameters:
        sdata: SpatialData object containing Xenium data
        he_image: H&E image as ndarray (H, W, C)
        centroids_um: Cell centroids in microns
        psx, psy: Pixel size in microns
        xenium_dir: Directory containing Xenium data
        
    Returns:
        coords_px: Aligned coordinates in pixels
        affine: The computed affine transformation
    """
    # Define the transformation matrix path
    transform_path = os.path.join(xenium_dir, "he_imagealignment.csv")
    
    # Check if transformation matrix already exists
    if os.path.exists(transform_path):
        print(f"✅ Found existing transformation matrix at {transform_path}")
        print("→ Loading and applying existing transformation...")
        
        # Load the transformation matrix
        transform_matrix = np.loadtxt(transform_path, delimiter=",")
        
        # Apply the transformation to centroids
        aligned_centroids_um = apply_transform(centroids_um, transform_matrix)
        
        # Convert microns → pixels
        coords_px = np.zeros_like(aligned_centroids_um, dtype=int)
        coords_px[:, 0] = np.round(aligned_centroids_um[:, 0] / psx).astype(int)
        coords_px[:, 1] = np.round(aligned_centroids_um[:, 1] / psy).astype(int)
        
        print("✅ Applied existing transformation to coordinates")
        
        # Create an affine transformation object for compatibility
        from spatialdata.transformations import Affine
        affine = Affine(transform_matrix)
        
        return coords_px, affine
    
    # If no transformation exists, proceed with manual landmark selection
    print("ℹ️ No existing transformation found. Starting manual landmark selection...")
    return select_and_align_with_landmarks(sdata, he_image, centroids_um, psx, psy, xenium_dir)

def apply_transform(points, transform):
    """Apply affine transformation to points."""
    # Ensure points is a 2D array
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Add homogeneous coordinate (1) to each point
    h_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply transformation
    transformed = h_points @ transform.T
    
    # Convert back to original coordinate space
    return transformed[:, :-1]

def select_and_align_with_landmarks(sdata, he_image, centroids_um, psx, psy, xenium_dir):
    """
    Open Napari for manual landmark selection and align Xenium data to H&E image.
    
    This function:
    1. Opens Napari with both the H&E image and Xenium visualization
    2. Lets the user manually select corresponding landmarks in both images
    3. Saves those landmarks to the SpatialData object
    4. Computes and applies the transformation
    5. Saves the transformation matrix to the specified location
    
    Parameters:
        sdata: SpatialData object containing Xenium data
        he_image: H&E image as ndarray (H, W, C)
        centroids_um: Cell centroids in microns
        psx, psy: Pixel size in microns
        xenium_dir: Directory to save the transformation matrix
        
    Returns:
        coords_px: Aligned coordinates in pixels
        affine: The computed affine transformation
    """
    print("🧭 Opening Napari for manual landmark selection...")
    
    # Create a napari viewer
    viewer = napari.Viewer()
    
    # Add H&E image to viewer
    he_layer = viewer.add_image(
        he_image, 
        name="H&E Image", 
        rgb=True if he_image.shape[-1] == 3 else False
    )
    
    # Add a representation of Xenium data to the viewer
    # Find an appropriate element to visualize (e.g., morphology image or cell boundaries)
    xenium_element_name = None
    
    # Try to find morphology image first
    for key in sdata.images.keys():
        if "morphology" in key.lower():
            xenium_element_name = key
            break
    
    # If no morphology found, try to use cell boundaries
    if xenium_element_name is None and 'cell_boundaries' in sdata.shapes:
        xenium_element_name = 'cell_boundaries'
        # Show cell boundaries from Xenium
        viewer.add_shapes(
            data=sdata.shapes['cell_boundaries'].geometry,
            name="Xenium Cell Boundaries",
            edge_color="cyan",
            face_color="transparent"
        )
    elif xenium_element_name is not None:
        # Show Xenium morphology image
        viewer.add_image(
            sdata.images[xenium_element_name].data,
            name="Xenium Morphology",
            blending="additive"
        )
    
    # Create points layers for the landmarks
    he_landmarks_layer = viewer.add_points(
        name="H&E Landmarks", 
        size=20,
        face_color="red",
        symbol="cross"
    )
    
    xenium_landmarks_layer = viewer.add_points(
        name="Xenium Landmarks", 
        size=20,
        face_color="green",
        symbol="cross"
    )
    
    # Display instructions to the user
    instructions = """
    INSTRUCTIONS:
    1. Add landmarks to "H&E Landmarks" layer by selecting it and clicking on distinct features
    2. Add CORRESPONDING landmarks to "Xenium Landmarks" in the SAME ORDER
    3. Add at least 3 pairs of landmarks for accurate alignment
    4. Close the Napari window when finished to proceed with alignment
    """
    
    print(instructions)
    viewer.text_overlay.text = instructions
    viewer.text_overlay.visible = True
    
    # This blocks until the window is closed
    napari.run()
    
    # Once window is closed, retrieve the landmark points
    he_landmark_points = he_landmarks_layer.data
    xenium_landmark_points = xenium_landmarks_layer.data
    
    # Validate we have enough landmarks and they match
    if len(he_landmark_points) < 3 or len(xenium_landmark_points) < 3:
        raise ValueError("❌ Need at least 3 landmark pairs for accurate alignment")
    
    if len(he_landmark_points) != len(xenium_landmark_points):
        raise ValueError(f"❌ Mismatched landmarks: {len(he_landmark_points)} (H&E) vs {len(xenium_landmark_points)} (Xenium)")
    
    print(f"✅ Successfully collected {len(he_landmark_points)} landmark pairs")
    
    # Convert landmarks to SpatialData ShapesModel and add to the SpatialData object
    # For H&E landmarks
    he_landmarks = ShapesModel.parse(
        he_landmark_points, 
        geometry=0,  # Points geometry
        radius=20,   # Visual size for landmarks
        transformations={"global": Identity()}  # Assuming H&E is in "global" coordinate system
    )
    
    # For Xenium landmarks
    xenium_landmarks = ShapesModel.parse(
        xenium_landmark_points, 
        geometry=0,  # Points geometry
        radius=20,   # Visual size for landmarks
        transformations={"global": Identity()}  # Assuming Xenium is in "global" coordinate system
    )
    
    # Add landmarks to SpatialData object
    sdata["he_landmarks"] = he_landmarks
    sdata["xenium_landmarks"] = xenium_landmarks
    
    # Save landmarks to disk
    sdata.write_element("he_landmarks")
    sdata.write_element("xenium_landmarks")
    print("✅ Landmarks saved to SpatialData object")
    
    # Compute the transformation using align_elements_using_landmarks
    if xenium_element_name:
        # If we have a xenium element to use as reference
        he_coord_system = "global"
        xenium_coord_system = "global"  # Change if your Xenium data is in a different coord system
        
        affine = align_elements_using_landmarks(
            references_coords=sdata["he_landmarks"],
            moving_coords=sdata["xenium_landmarks"],
            reference_element=he_landmarks,  # Using H&E landmarks as reference element
            moving_element=sdata[xenium_element_name],
            reference_coordinate_system=he_coord_system,
            moving_coordinate_system=xenium_coord_system,
            new_coordinate_system="aligned"
        )
    else:
        # If we don't have a specific element, just get the transformation
        affine = get_transformation_between_landmarks(
            references_coords=sdata["he_landmarks"],
            moving_coords=sdata["xenium_landmarks"]
        )
    
    # Print the transformation details
    print("✅ Computed alignment transformation:")
    print(affine)
    
    # Get the transformation matrix (handle different types of transformations)
    if hasattr(affine, 'to_matrix'):
        transform_matrix = affine.to_matrix()
    elif hasattr(affine, 'affine_matrix'):
        transform_matrix = affine.affine_matrix
    else:
        # This handles sequences or other transformation types
        # For demonstration, we'll extract the matrix part
        transform_matrix = affine.to_dict()['matrix'] if hasattr(affine, 'to_dict') else affine

    # Apply transformation to centroids
    aligned_centroids_um = apply_transform(centroids_um, transform_matrix)
    
    # Convert microns → pixels
    coords_px = np.zeros_like(aligned_centroids_um, dtype=int)
    coords_px[:, 0] = np.round(aligned_centroids_um[:, 0] / psx).astype(int)
    coords_px[:, 1] = np.round(aligned_centroids_um[:, 1] / psy).astype(int)
    
    # Save the transformation matrix in the specified location
    transform_path = os.path.join(xenium_dir, "he_imagealignment.csv")
    np.savetxt(transform_path, transform_matrix, delimiter=",")
    print(f"✅ Transformation matrix saved to {transform_path}")
    
    # Also save aligned coordinates for reference
    np.save(os.path.join(xenium_dir, "aligned_centroids_um.npy"), aligned_centroids_um)
    
    # Write transformations to disk
    sdata.write_transformations()
    print("✅ Transformations saved to SpatialData object")
    
    return coords_px, affine