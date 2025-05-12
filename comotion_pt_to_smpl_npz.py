import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Default values, similar to extract_joints_to_trc.py for consistency
DEFAULT_PT_FILE = "Test2.pt" # Should be specified by user ideally
DEFAULT_OUTPUT_DIR = "smpl_npz_output"
DEFAULT_FPS = 30.0
DEFAULT_GENDER = "neutral" # SMPL model gender

def load_pt_data(pt_path: str | Path, weights_only: bool = False) -> dict:
    """
    Loads a PyTorch .pt file and checks for required keys.
    (Adapted from extract_joints_to_trc.py)
    """
    if not weights_only:
        print(f"Warning: Loading .pt file '{pt_path}' with weights_only=False. "
              f"Ensure the file is from a trusted source.")
    data = torch.load(pt_path, map_location="cpu", weights_only=weights_only)
    print(f"Loaded from {pt_path}. Keys: {list(data.keys())}")

    required_keys = ["id", "frame_idx", "pose", "betas", "trans"]
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(f"Missing required keys in {pt_path}: {missing_keys}")

    if data['pose'].shape[-1] != 72:
        print(f"Warning: Expected pose dimension to be 72 (SMPL global_orient + body_pose), but got {data['pose'].shape[-1]}.")
    # Add other checks if necessary, e.g., betas shape (10)
    return data

def main(
    pt_file: str | Path,
    output_dir: str | Path,
    fps: float = DEFAULT_FPS,
    gender: str = DEFAULT_GENDER,
    load_weights_only: bool = False # For torch.load, safer if True but depends on .pt content
):
    """
    Converts SMPL parameters from a CoMotion .pt file to individual .npz files
    for each person, compatible with SMPL2AddBiomechanics.
    """
    pt_file_path = Path(pt_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting conversion: {pt_file_path}")
    print(f"Output directory: {output_path}")
    print(f"FPS: {fps}, Gender: {gender}")

    full_data = load_pt_data(pt_file_path, weights_only=load_weights_only)

    # Using pandas to group by person ID and sort by frame_idx, similar to extract_joints_to_trc.py
    df_data_list = []
    for i in range(full_data['id'].shape[0]):
        df_data_list.append({
            "id": full_data["id"][i].item(),
            "frame_idx": full_data["frame_idx"][i].item(),
            "pose": full_data["pose"][i].cpu().numpy(), # Convert to numpy
            "trans": full_data["trans"][i].cpu().numpy(), # Convert to numpy
            "betas": full_data["betas"][i].cpu().numpy() # Convert to numpy
        })
    df = pd.DataFrame(df_data_list)

    unique_ids = sorted(df["id"].unique())
    print(f"Found {len(unique_ids)} unique person IDs: {unique_ids}")

    for person_id in unique_ids:
        print(f"\nProcessing data for person ID: {person_id}")
        person_df = df[df["id"] == person_id].sort_values(by="frame_idx")

        if person_df.empty:
            print(f"  No data found for person ID: {person_id}. Skipping.")
            continue
        # Reconstruct numpy arrays for this person
        person_poses_np = np.stack(person_df["pose"].tolist())
        person_trans_np = np.stack(person_df["trans"].tolist())

        # # Apply 180-degree rotation around X-axis to global orientation
        # num_frames_pose = person_poses_np.shape[0]
        # for i in range(num_frames_pose):
        #     global_orient_aa = person_poses_np[i, 0:3]
            
        #     # Skip if global_orient_aa is zero vector to avoid issues with Rotation.from_rotvec
        #     if np.all(global_orient_aa == 0):
        #         # If orientation is zero, it remains zero after rotation
        #         person_poses_np[i, 0:3] = global_orient_aa
        #         continue

        #     r_orig = R.from_rotvec(global_orient_aa)
        #     r_flip_x = R.from_euler('x', 180, degrees=True)
            
        #     # Post-multiply to rotate in the world frame
        #     # If we want to rotate in the local frame of the object: r_new = r_orig * r_flip_x
        #     # For flipping an already oriented object in world space: r_new = r_flip_x * r_orig
        #     r_new = r_flip_x * r_orig 
            
        #     person_poses_np[i, 0:3] = r_new.as_rotvec()
        # print(f"  Applied 180-degree X-axis rotation to global orientation for person {person_id}.")

        # Use betas from the first frame, assuming they are constant for the sequence
        # The SMPL2AddBiomechanics loader handles expanding this if it's (10,)
        person_betas_np = person_df["betas"].iloc[0]
        if person_betas_np.shape[0] > 10:
             print(f"  Warning: Person {person_id} has betas with shape {person_betas_np.shape}. Using first 10.")
             person_betas_np = person_betas_np[:10]
        elif person_betas_np.shape[0] < 10:
            raise ValueError(f"Person {person_id} has betas with shape {person_betas_np.shape}. Expected at least 10.")


        num_frames = person_poses_np.shape[0]
        print(f"  Person {person_id}: {num_frames} frames.")
        print(f"  Poses shape: {person_poses_np.shape}, Trans shape: {person_trans_np.shape}, Betas shape: {person_betas_np.shape}")

        # Data for .npz file
        npz_data_dict = {
            "poses": person_poses_np,  # (num_frames, 72)
            "trans": person_trans_np,  # (num_frames, 3)
            "betas": person_betas_np,  # (10,)
            "gender": str(gender),     # string
            "mocap_framerate": float(fps) # float scalar
        }

        # Define output filename
        # e.g., if pt_file is "results/Test2.pt", output is "smpl_npz_output/Test2_P1.npz"
        output_npz_filename = f"{pt_file_path.stem}_P{person_id}.npz"
        output_npz_filepath = output_path / output_npz_filename

        try:
            np.savez(output_npz_filepath, **npz_data_dict)
            print(f"  ✓ Saved .npz file for person {person_id} to {output_npz_filepath}")
        except Exception as e:
            print(f"  ✗ Error saving .npz file for person {person_id}: {e}")

    print("\nConversion complete.")

if __name__ == "__main__":
    # --- Configuration --- 
    # !!! EDIT THESE VALUES BEFORE RUNNING THE SCRIPT !!!
    pt_file_to_process = "results/Test2.pt"  # Path to the input CoMotion .pt file
    output_directory = "results/smpl_npz_output" # Directory to save the output .npz files
    motion_fps = 30.0  # Frame rate of the motion sequence
    model_gender = "neutral"  # Gender of the SMPL model ('male', 'female', 'neutral')
    use_load_weights_only = False # Use weights_only=True when loading .pt file (safer if True)
    # ---------------------

    print("--- Running comotion_pt_to_smpl_npz.py ---")
    print(f"Input .pt file: {pt_file_to_process}")
    print(f"Output directory for .npz files: {output_directory}")
    print(f"Motion FPS: {motion_fps}")
    print(f"SMPL Gender: {model_gender}")
    print(f"Load .pt with weights_only: {use_load_weights_only}")
    print("--------------------------------------------")

    # Ensure the output directory for the .npz files exists, relative to the script or workspace
    # Path(output_directory).mkdir(parents=True, exist_ok=True)
    # This is already handled in main()

    main(
        pt_file=pt_file_to_process,
        output_dir=output_directory,
        fps=motion_fps,
        gender=model_gender,
        load_weights_only=use_load_weights_only
    ) 
