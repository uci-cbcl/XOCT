import matplotlib as mpl
# Set matplotlib to use the Agg backend (offline mode)
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import shutil

# Define paths and structure
folders = ["results_octa_6M", "results_octa_3M"]
sub_folders = ["FULL", "ILM_OPL", "Mean", "OPL_BM"]
norm_folder = "norm_True"  # Assuming this is consistent

# plot method order: [0] ground truth, [1] unet3d2dsplit3stem_3d2d_perceptual_gn_3M, [2] transprostem_gn_3M, [3] unet3d2dallstem_3d2d_perceptual_gn_3M, [4] unet3dmsffstem_scale_cam_res_gn_3M, [5] bbdm_3d, [6] unet3dstem_gn_3M

# Ground truth image location template
gt_template = "octa-500/OCT2OCTA{}_3D/Projection_Maps_from_npy/OCTA/norm_True/{}/{}.bmp"

# Ground truth 3D volume location template
gt_3d_template = "octa-500/OCT2OCTA{}_3D/test/A/{}.npy"

# Map folder names to their corresponding parts in the ground truth path
folder_to_gt_map = {
    "results_octa_3M": "3M",
    "results_octa_6M": "6M"
}

# Desired plot method order (may need to adjust based on actual method names available in each folder)
desired_method_order = [
    "unet3d2dmsffallstem_3d2d_perceptual_scale_cam_res_gn_3M",
    "transprostem_gn_3M",
    "bbdm_3d",
    "unet3dstem_gn_3M",
]

method_to_title = {
    "transprostem_gn_3M": "TransPro",
    "unet3dstem_gn_3M": "Pix2Pix3D",
    "bbdm_3d": "BBDM3D",
    "unet3d2dmsffallstem_3d2d_perceptual_scale_cam_res_gn_3M": "XOCT",
    "transprostem_gn_6M": "TransPro",
    "unet3dstem_gn_6M": "Pix2Pix3D",
    "unet3d2dmsffallstem_3d2d_perceptual_scale_cam_res_gn_6M": "XOCT"
}

# Function to get ordered method list for a specific folder
def get_ordered_methods(folder):
    if folder == "results_octa_6M":
        # Adjust method names for 6M by replacing _3M with _6M
        adjusted_order = [m.replace("_3M", "_6M") for m in desired_method_order]
    else:
        adjusted_order = desired_method_order
    return adjusted_order

# Get methods for each folder separately and order them
folder_methods = {}
for folder in folders:
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist")
        continue
    
    # Get method names (direct subdirectories in each folder)
    available_methods = [d for d in os.listdir(folder) 
                        if os.path.isdir(os.path.join(folder, d))]
    
    # Order methods according to desired order
    folder_methods[folder] = get_ordered_methods(folder)
    
    print(f"Found {len(folder_methods[folder])} methods in {folder} (ordered): {folder_methods[folder]}")

# Function to get all patient IDs from a specific folder/method/subfolder
def get_patient_ids(folder, method, sub_folder):
    path = os.path.join(folder, method, norm_folder, sub_folder)
    if not os.path.exists(path):
        return []
    
    # Get all image files (assuming .bmp extension)
    return [f for f in os.listdir(path) if f.endswith('.bmp')]

# Get common patients within each folder (across methods and subfolders)
def get_common_patients_per_folder():
    common_patients_by_folder = {}
    
    for folder in folders:
        if folder not in folder_methods:
            continue
            
        methods = folder_methods[folder]
        folder_patients = set()
        first = True
        
        for method in methods:
            for sub_folder in sub_folders:
                patients = set(get_patient_ids(folder, method, sub_folder))
                
                if first and patients:
                    folder_patients = patients
                    first = False
                elif patients:
                    folder_patients = folder_patients.intersection(patients)
        
        common_patients_by_folder[folder] = sorted(list(folder_patients))
        print(f"Found {len(common_patients_by_folder[folder])} common patients in {folder}")
    
    return common_patients_by_folder

common_patients_by_folder = get_common_patients_per_folder()

# Create output directory if it doesn't exist
os.makedirs("quality_plots", exist_ok=True)

# Visualize images for each patient
def visualize_patient(patient_id, folder):
    methods = folder_methods[folder]
    
    # Skip if no methods to visualize
    if not methods:
        print(f"No desired methods found for {folder}, skipping patient {patient_id}")
        return
    
    for sub_folder in sub_folders:
        # Two rows: first for images, second for differences
        fig, axs = plt.subplots(2, len(methods) + 1, figsize=(30, 10))
        fig.suptitle(f"Patient {patient_id} - {sub_folder} - {folder}", fontsize=16)
        
        # Get ground truth image first
        gt_image = None
        gt_suffix = folder_to_gt_map.get(folder, "")
        if not gt_suffix:
            print(f"Warning: No ground truth mapping for folder {folder}")
            axs[0, 0].text(0.5, 0.5, "GT mapping error",
                       horizontalalignment='center', verticalalignment='center')
        else:
            gt_path = gt_template.format(gt_suffix, sub_folder, patient_id.split('.')[0])
            if os.path.exists(gt_path):
                gt_image = cv2.imread(gt_path)
                if gt_image is not None:
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                    axs[0, 0].imshow(gt_image)
                else:
                    axs[0, 0].text(0.5, 0.5, "GT load error",
                               horizontalalignment='center', verticalalignment='center')
            else:
                axs[0, 0].text(0.5, 0.5, f"GT not found\n{gt_path}",
                           horizontalalignment='center', verticalalignment='center')
        
        axs[0, 0].set_title("Ground Truth")
        axs[0, 0].axis('off')
        
        # Empty plot in the difference row for ground truth position
        axs[1, 0].text(0.5, 0.5, "No difference\n(Ground Truth)", 
                    horizontalalignment='center', verticalalignment='center')
        axs[1, 0].axis('off')
        
        # Add the method results in the ordered sequence
        for i, method in enumerate(methods):
            img_path = os.path.join(folder, method, norm_folder, sub_folder, patient_id)
            method_image = None
            
            # First row: method image
            if os.path.exists(img_path):
                method_image = cv2.imread(img_path)
                if method_image is not None:
                    method_image = cv2.cvtColor(method_image, cv2.COLOR_BGR2RGB)
                    axs[0, i+1].imshow(method_image)
                else:
                    axs[0, i+1].text(0.5, 0.5, "Image load error", 
                                  horizontalalignment='center', verticalalignment='center')
            else:
                axs[0, i+1].text(0.5, 0.5, "Not available", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Use the method_to_title mapping for better titles
            method_title = method_to_title.get(method, method)
            axs[0, i+1].set_title(method_title, fontsize=14)
            
            # Second row: difference image
            if gt_image is not None and method_image is not None:
                # Ensure both images have the same dimensions
                if gt_image.shape == method_image.shape:
                    # Calculate absolute difference
                    diff_image = cv2.absdiff(gt_image, method_image)
                    
                    # Enhance difference visibility
                    diff_image_enhanced = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Display the difference
                    axs[1, i+1].imshow(diff_image_enhanced)
                    
                    # Calculate and display mean absolute error with the mapped method title
                    mae = np.mean(np.abs(gt_image.astype(float) - method_image.astype(float)))
                    axs[1, i+1].set_title(f"{method_title} Diff (MAE: {mae:.2f})", fontsize=12)
                else:
                    axs[1, i+1].text(0.5, 0.5, f"Size mismatch\nGT: {gt_image.shape}\nMethod: {method_image.shape}", 
                                  horizontalalignment='center', verticalalignment='center')
            else:
                axs[1, i+1].text(0.5, 0.5, "Cannot calculate difference", 
                              horizontalalignment='center', verticalalignment='center')
            
            axs[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Give more space to the suptitle
        plt.savefig(f"quality_plots/patient_{patient_id}_{sub_folder}_{folder}.png", dpi=300)
        plt.close(fig)

# Function to extract all method images for a specific patient
def extract_patient_images(patient_id, output_dir=None, volume_slice=106):
    """
    Extract all images related to a specific patient from all methods and save them in a structured folder.
    Also extracts slices from 3D volume data.
    
    Args:
        patient_id (str): The patient ID to extract (with or without extension)
        output_dir (str): The output directory. If None, uses "extracted_patients/{patient_id}"
        volume_slice (int): The slice number to extract from 3D volumes (default: 106)
    
    Returns:
        str: The path to the output directory
    """
    # Handle patient_id with or without extension
    patient_base = patient_id.split('.')[0]
    if not output_dir:
        output_dir = os.path.join("extracted_patients", patient_base)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting images for patient {patient_base} to {output_dir}")
    
    # Track what we've extracted
    extracted_count = 0
    
    # Extract ground truth images
    for folder in folders:
        gt_suffix = folder_to_gt_map.get(folder, "")
        if gt_suffix:
            for sub_folder in sub_folders:
                # Ground truth path
                gt_path = gt_template.format(gt_suffix, sub_folder, patient_base)
                if os.path.exists(gt_path):
                    # Create subfolders
                    gt_out_dir = os.path.join(output_dir, "ground_truth", gt_suffix, sub_folder)
                    os.makedirs(gt_out_dir, exist_ok=True)
                    
                    # Copy the file
                    gt_out_path = os.path.join(gt_out_dir, os.path.basename(gt_path))
                    shutil.copy2(gt_path, gt_out_path)
                    extracted_count += 1
                    print(f"Extracted GT: {gt_path} -> {gt_out_path}")
    
    # Extract ground truth 3D volumes
    for suffix in ["3M", "6M"]:
        gt_3d_path = gt_3d_template.format(suffix, patient_base)
        if os.path.exists(gt_3d_path):
            try:
                # Load the 3D volume
                gt_volume = np.load(gt_3d_path).squeeze()
                
                # Extract the specified slice
                if gt_volume.ndim >= 3 and gt_volume.shape[0] > volume_slice:
                    slice_data = gt_volume[volume_slice, :, :]
                    
                    # Create output directory
                    slice_dir = os.path.join(output_dir, "3d_slices", "ground_truth", suffix)
                    os.makedirs(slice_dir, exist_ok=True)
                    
                    # Save as image
                    slice_filename = f"{patient_base}_A_slice{volume_slice}.png"
                    slice_path = os.path.join(slice_dir, slice_filename)
                    
                    # Normalize to 0-255 for better visualization
                    normalized_slice = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Save the slice
                    cv2.imwrite(slice_path, normalized_slice.astype(np.uint8))
                    extracted_count += 1
                    print(f"Extracted GT 3D slice: {gt_3d_path}[{volume_slice},:,:] -> {slice_path}")
                    
                    # Also save a colormap version for better visualization
                    plt.figure(figsize=(8, 8))
                    plt.imshow(slice_data, cmap='viridis')
                    plt.colorbar(label='Intensity')
                    plt.title(f"Ground Truth {suffix} - Slice {volume_slice}")
                    plt.axis('off')
                    colormap_path = os.path.join(slice_dir, f"{patient_base}_slice{volume_slice}_colormap.png")
                    plt.savefig(colormap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    extracted_count += 1
                else:
                    print(f"Warning: GT Volume shape incompatible for slice {volume_slice}: {gt_volume.shape}")
                    
            except Exception as e:
                print(f"Error processing GT 3D volume {gt_3d_path}: {str(e)}")
    
    # Extract method images and 3D volumes
    for folder in folders:
        if folder in folder_methods:
            methods = folder_methods[folder]
            
            for method in methods:
                # Extract 2D projections
                # for sub_folder in sub_folders:
                #     # Try both with and without extension
                #     for patient_variation in [patient_base, f"{patient_base}.bmp"]:
                #         img_path = os.path.join(folder, method, norm_folder, sub_folder, patient_variation)
                #         if os.path.exists(img_path):
                #             # Create subfolders
                #             method_title = method_to_title.get(method, method)
                #             method_out_dir = os.path.join(output_dir, "methods", folder, method_title, sub_folder)
                #             os.makedirs(method_out_dir, exist_ok=True)
                            
                #             # Copy the file
                #             method_out_path = os.path.join(method_out_dir, os.path.basename(img_path))
                #             shutil.copy2(img_path, method_out_path)
                #             extracted_count += 1
                #             print(f"Extracted method: {img_path} -> {method_out_path}")
                #             break  # Found one variation, no need to try the other
                
                # Extract 3D volume slice
                volume_dir = os.path.join(folder, method, "test_latest", "fake_B_3d")
                if os.path.exists(volume_dir):
                    volume_files = [f for f in os.listdir(volume_dir) 
                                    if f.startswith(patient_base) and f.endswith('.npy')]
                    
                    for vol_file in volume_files:
                        vol_path = os.path.join(volume_dir, vol_file)
                        try:
                            # Load the 3D volume
                            volume = np.load(vol_path).squeeze()
                            
                            # Extract the specified slice
                            if volume.ndim >= 3 and volume.shape[0] > volume_slice:
                                slice_data = volume[volume_slice, :, :]
                                
                                # Create output directory
                                method_title = method_to_title.get(method, method)
                                slice_dir = os.path.join(output_dir, "3d_slices", folder, method_title)
                                os.makedirs(slice_dir, exist_ok=True)
                                
                                # Save as image
                                slice_filename = f"{patient_base}_slice{volume_slice}.png"
                                slice_path = os.path.join(slice_dir, slice_filename)
                                
                                # Normalize to 0-255 for better visualization
                                normalized_slice = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX)
                                
                                # Save the slice
                                cv2.imwrite(slice_path, normalized_slice.astype(np.uint8))
                                extracted_count += 1
                                print(f"Extracted 3D slice: {vol_path}[{volume_slice},:,:] -> {slice_path}")
                                
                                # Also save a colormap version for better visualization
                                plt.figure(figsize=(8, 8))
                                plt.imshow(slice_data, cmap='viridis')
                                plt.colorbar(label='Intensity')
                                plt.title(f"{method_title} - Slice {volume_slice}")
                                plt.axis('off')
                                colormap_path = os.path.join(slice_dir, f"{patient_base}_slice{volume_slice}_colormap.png")
                                plt.savefig(colormap_path, dpi=300, bbox_inches='tight')
                                plt.close()
                                extracted_count += 1
                            else:
                                print(f"Warning: Volume shape incompatible for slice {volume_slice}: {volume.shape}")
                                
                        except Exception as e:
                            print(f"Error processing 3D volume {vol_path}: {str(e)}")
    
    # Create a summary figure showing all volume slices side by side (GT vs methods)
    try:
        # Collect all the extracted 3D slice images
        gt_slices = {}
        method_slices = {}
        
        # Find GT slices
        for suffix in ["3M", "6M"]:
            gt_slice_path = os.path.join(output_dir, "3d_slices", "ground_truth", suffix, 
                                       f"{patient_base}_slice{volume_slice}.png")
            if os.path.exists(gt_slice_path):
                gt_slices[suffix] = cv2.imread(gt_slice_path)
        
        # Find method slices for each folder
        for folder in folders:
            if folder in folder_methods:
                for method in folder_methods[folder]:
                    method_title = method_to_title.get(method, method)
                    slice_path = os.path.join(output_dir, "3d_slices", folder, method_title,
                                           f"{patient_base}_slice{volume_slice}.png")
                    if os.path.exists(slice_path):
                        key = f"{folder}_{method}"
                        method_slices[key] = {
                            'image': cv2.imread(slice_path),
                            'title': f"{method_title} ({folder.split('_')[-1]})"
                        }
        
        # If we found any slices, create a comparison figure
        if gt_slices or method_slices:
            # Determine grid size
            n_cols = max(1, len(method_slices))
            n_rows = len(gt_slices) + 1 if gt_slices else 1
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            
            # Make axs always a 2D array for consistent indexing
            if n_rows == 1 and n_cols == 1:
                axs = np.array([[axs]])
            elif n_rows == 1:
                axs = np.array([axs])
            elif n_cols == 1:
                axs = np.array([[ax] for ax in axs])
            
            # Plot ground truth in first row
            for i, (suffix, img) in enumerate(gt_slices.items()):
                if i < n_cols:
                    axs[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axs[0, i].set_title(f"Ground Truth {suffix}")
                    axs[0, i].axis('off')
            
            # Fill any empty spots in first row
            for i in range(len(gt_slices), n_cols):
                axs[0, i].axis('off')
            
            # Plot methods in second row
            for i, (key, data) in enumerate(method_slices.items()):
                if i < n_cols:
                    row = 1 if len(gt_slices) > 0 else 0
                    axs[row, i].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB), vmin=img.min(), vmax=img.max())
                    axs[row, i].set_title(data['title'])
                    axs[row, i].axis('off')
            
            plt.tight_layout()
            summary_path = os.path.join(output_dir, f"{patient_base}_3d_slice{volume_slice}_comparison.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            extracted_count += 1
            print(f"Created 3D slice comparison summary: {summary_path}")
        
    except Exception as e:
        print(f"Error creating summary figure: {str(e)}")
    
    if extracted_count == 0:
        print(f"Warning: No images found for patient {patient_base}")
    else:
        print(f"Successfully extracted {extracted_count} images for patient {patient_base}")
    
    return output_dir

# Example usage
# extract_patient_images("10231.bmp")  # Uncomment to use

# # Example usage: Visualize the first 3 patients from each folder
# for folder, patients in common_patients_by_folder.items():
#     num_to_show = len(patients)
#     print(f"Visualizing {num_to_show} patients from {folder}")
    
#     for patient_id in patients[:num_to_show]:
#         visualize_patient(patient_id, folder)

# print("All visualizations have been saved to the 'quality_plots' directory.")