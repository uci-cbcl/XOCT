import os
import sys
from visualize import extract_patient_images
import argparse

def main():
    """
    Utility script to extract images for one or more patients.
    """
    parser = argparse.ArgumentParser(description="Extract patient images and 3D volume slices")
    parser.add_argument("patient_ids", nargs="+", help="One or more patient IDs to extract")
    parser.add_argument("--slice", "-s", type=int, default=106, 
                        help="Slice number to extract from 3D volumes (default: 106)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Custom output directory (default: ./extracted_patients/patient_id)")
    
    args = parser.parse_args()
    
    # Extract each patient specified
    for patient_id in args.patient_ids:
        for slice_num in range(90, 155, 5):
            extract_patient_images(patient_id, output_dir=args.output, volume_slice=slice_num)
        # extract_patient_images(patient_id, output_dir=args.output, volume_slice=args.slice)
    
    print(f"Extraction complete for {len(args.patient_ids)} patients.")

if __name__ == "__main__":
    main()
