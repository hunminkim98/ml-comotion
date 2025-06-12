import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tkinter import filedialog

# === SMPL Model Constants ===
SMPL_POSE_DIMENSIONS: int = 72
SMPL_BETAS_DIMENSIONS: int = 10
SMPL_REQUIRED_KEYS: list[str] = ["id", "frame_idx", "pose", "betas", "trans"]
SMPL_OUTPUT_FORMAT_JOINTS: str = "mesh" # Options: "joints", "joints_face", "joints_coco", "mesh"

# === Unit Conversion Constants ===
MM_TO_M_SCALE: float = 1000.0

# === Default Configuration Values ===
DEFAULT_PT_FILE: str = "Test.pt"
DEFAULT_FPS: float = 30.0
DEFAULT_UNITS_MM: bool = False

# === Coordinate System Transformation Constants ===
# CoMotion/SMPL: X=right, Y=up, Z=forward
# OpenSim: X=forward, Y=up, Z=left
# Transformation: Input X -> OpenSim -Z, Input Y -> OpenSim +Y,
# Input Z -> OpenSim +X
# Current setting maintains SMPL X,Y,Z order with Y-axis sign flip
# OpenSim X = Input X, OpenSim Y = -Input Y, OpenSim Z = Input Z
DEFAULT_AXES_SWAP: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (0, 1, 2), (1, -1, 1)
)
DEFAULT_ZERO_PELVIS: bool = False
DEFAULT_FORCE_ZERO_PELVIS: bool = False

# === SMPL Joint Names ===
SMPL_STANDARD_JOINT_NAMES: list[str] = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

# === Automatic Path Detection and Import Setup ===
def _setup_imports() -> None:
    """Setup import paths and handle import errors gracefully."""
    script_dir = Path(__file__).parent.absolute()
    src_dir = script_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Initialize import setup
_setup_imports()

# Import with clear error handling
try:
    from comotion_demo.utils.smpl_kinematics import SMPLKinematics
except ImportError as e:
    print("ERROR: Failed to import SMPLKinematics module.")
    print(f"Import error: {e}")
    print("\nPossible solutions:")
    print("1. Ensure you're running this script from the project root directory")
    print("2. Verify that the 'src/comotion_demo' directory exists")
    print("3. Check that all required dependencies are installed")
    print("4. Make sure the SMPL model file is properly placed in src/comotion_demo/data/smpl/")
    sys.exit(1)

def load_pt(pt_path: str | Path, weights_only: bool = False) -> dict:
    """
    Load PyTorch .pt file and validate required keys.

    Args:
        pt_path: Path to the .pt file.
        weights_only: torch.load weights_only parameter. True recommended
                     for security.

    Returns:
        Loaded data dictionary.

    Raises:
        KeyError: If required keys ('pose', 'betas', 'trans', 'id',
                 'frame_idx') are missing.
    """
    if not weights_only:
        print(f"Warning: Loading .pt file '{pt_path}' with weights_only=False. "
              f"Ensure the file is from a trusted source.")

    data = torch.load(pt_path, map_location="cpu", weights_only=weights_only)
    print(f"Loaded from {pt_path}. Keys: {list(data.keys())}")

    missing_keys = [k for k in SMPL_REQUIRED_KEYS if k not in data]
    if missing_keys:
        raise KeyError(f"Missing required keys in {pt_path}: {missing_keys}")

    if data['pose'].shape[-1] != SMPL_POSE_DIMENSIONS:
        print(f"Warning: Expected pose dimension to be {SMPL_POSE_DIMENSIONS}, "
              f"but got {data['pose'].shape[-1]}. SMPLKinematics expects full "
              f"{SMPL_POSE_DIMENSIONS}-dim pose.")
    return data

def smpl_kinematics_to_joints(data: dict,
                              smpl_model: SMPLKinematics) -> torch.Tensor:
    """
    Compute joint coordinates from SMPL parameters using SMPLKinematics model.

    Args:
        data: Dictionary containing 'betas', 'pose', 'trans' keys.
              betas can be either (num_frames, 10) or (10,) shape.
              If (10,), it will be expanded to match num_frames.
        smpl_model: Initialized SMPLKinematics model instance.

    Returns:
        Computed joint coordinates as (N, 24, 3) PyTorch tensor in meters.
    """
    with torch.no_grad():
        pose = torch.as_tensor(data["pose"], dtype=torch.float32,
                              device="cpu")
        trans = torch.as_tensor(data["trans"], dtype=torch.float32,
                               device="cpu")
        betas_input = torch.as_tensor(data["betas"], dtype=torch.float32,
                                     device="cpu")

        num_frames = pose.shape[0]

        # Handle different betas input shapes
        if (betas_input.ndim == 1 and
            betas_input.shape[0] == SMPL_BETAS_DIMENSIONS):
            # Expand (10,) to (num_frames, 10)
            betas = betas_input.unsqueeze(0).repeat(num_frames, 1)
        elif (betas_input.ndim == 2 and betas_input.shape[0] == num_frames
              and betas_input.shape[1] == SMPL_BETAS_DIMENSIONS):
            betas = betas_input
        else:
            raise ValueError(f"Betas must be of shape ({SMPL_BETAS_DIMENSIONS},) "
                           f"or (num_frames, {SMPL_BETAS_DIMENSIONS}), "
                           f"got {betas_input.shape}")

        joints = smpl_model(
            betas=betas,
            pose=pose,
            trans=trans,
            output_format=SMPL_OUTPUT_FORMAT_JOINTS
        )
    return joints

def write_trc(
    joints: torch.Tensor,
    out_path: str | Path,
    fps: float,
    units_mm: bool = False,
    marker_names: list[str] | None = None,
    axes_swap: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    zero_pelvis: bool = True,
    force_zero_pelvis: bool = False
) -> None:
    """
    Save joint coordinates to .trc file format.

    Args:
        joints: (N, M, 3) shaped PyTorch tensor of joint coordinates.
        out_path: Output .trc file path.
        fps: Frames per second.
        units_mm: If True, use millimeters; if False, use meters.
        marker_names: List of marker names. If None, auto-generate as
                     J00, J01, etc.
        axes_swap: Tuple for axis transformation ((idx_tuple), (sign_tuple)).
                  Example: ((0,2,1),(1,1,-1)) -> (X, Z, -Y).
        zero_pelvis: If True, move first frame pelvis to origin and apply
                    same offset to all joints in all frames.
        force_zero_pelvis: If True, force pelvis to origin for each frame
                          individually. Applied after zero_pelvis.
                          Mainly for visualization and pelvis motion removal.
    """
    # Clone to avoid modifying original tensor
    processed_joints = joints.clone()

    # Apply pelvis zeroing before coordinate transformation
    if zero_pelvis:
        if processed_joints.shape[0] > 0 and processed_joints.shape[1] > 0:
            # Get first frame pelvis position (first joint)
            pelvis_origin = processed_joints[0:1, 0:1, :].clone()
            # Subtract offset from all joints in all frames
            processed_joints -= pelvis_origin
            print(f"  Pelvis zeroed based on first frame. Original "
                  f"pelvis[0,0]: {pelvis_origin.numpy().squeeze()}")
        else:
            print("  Warning: Cannot apply zero_pelvis, joints tensor is "
                  "empty or has no joints.")

    # Force pelvis to origin for all frames (after zero_pelvis)
    if force_zero_pelvis:
        if processed_joints.shape[0] > 0 and processed_joints.shape[1] > 0:
            num_frames = processed_joints.shape[0]
            for frame_idx in range(num_frames):
                # Get current frame pelvis position (joint 0)
                current_pelvis_pos = processed_joints[frame_idx:frame_idx+1,
                                                    0:1, :].clone()
                # Subtract pelvis position from all joints in current frame
                processed_joints[frame_idx, :, :] = (
                    processed_joints[frame_idx, :, :] - current_pelvis_pos
                )
            print(f"  Pelvis forced to origin for all {num_frames} frames.")
        else:
            print("  Warning: Cannot apply force_zero_pelvis, joints tensor "
                  "is empty or has no joints.")

    # Convert to numpy
    joint_positions_np = processed_joints.detach().cpu().numpy()

    # Apply coordinate system transformation
    if axes_swap:
        idx, sign = axes_swap
        assert (len(idx) == 3 and all(isinstance(i, int) for i in idx)), \
            "axes_swap idx_tuple must contain 3 integers."
        assert (len(sign) == 3 and
                all(isinstance(s, (int, float)) for s in sign)), \
            "axes_swap sign_tuple must contain 3 numbers."

        # Use temporary variable to avoid in-place modification issues
        joint_swapped = np.empty_like(joint_positions_np)
        for axis_idx in range(3):
            joint_swapped[..., axis_idx] = (joint_positions_np[..., idx[axis_idx]]
                                          * sign[axis_idx])
        joint_positions_np = joint_swapped

    # Apply unit conversion
    scale = MM_TO_M_SCALE if units_mm else 1.0
    joint_positions_np *= scale
    joint_positions_np = joint_positions_np.astype(np.float64, copy=False)

    num_frames, num_joints, _ = joint_positions_np.shape
    actual_marker_names = (marker_names if marker_names and
                          len(marker_names) == num_joints
                          else [f"J{i:02d}" for i in range(num_joints)])

    if marker_names and len(marker_names) != num_joints:
        print(f"Warning: Provided marker_names count ({len(marker_names)}) "
              f"does not match number of joints ({num_joints}). Using "
              f"default names J00-J{num_joints-1}.")

    output_units = "mm" if units_mm else "m"
    out_path_obj = Path(out_path)

    # Create TRC file header
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{out_path_obj.name}",
        ("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
         "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"),
        (f"{fps:.1f}\t{fps:.1f}\t{num_frames}\t{num_joints}\t"
         f"{output_units}\t{fps:.1f}\t1\t{num_frames}"),
        "Frame#\tTime\t" + "\t\t\t".join(actual_marker_names),
        ("\t\t" + "\t".join([ax + str(i+1) for i in range(num_joints)
                            for ax in ("X", "Y", "Z")]))
    ]

    # Write TRC file
    with open(out_path_obj, "w") as f:
        for header_line in header:
            f.write(header_line + "\n")
        for frame_idx in range(num_frames):
            time_stamp = frame_idx / fps
            row_data = [str(frame_idx + 1), f"{time_stamp:.6f}"]
            row_data += [f"{coord:.6f}" for coord in
                        joint_positions_np[frame_idx].reshape(-1)]
            f.write("\t".join(row_data) + "\n")

    print(f"✓ TRC file saved to {out_path_obj} ({num_frames} frames, "
          f"{num_joints} markers, {output_units})")

def main(
    pt_file: str | Path,
    trc_file_template: str,
    fps: float = DEFAULT_FPS,
    units_mm: bool = DEFAULT_UNITS_MM,
    axes_swap: tuple[tuple[int, int, int], tuple[int, int, int]] | None = DEFAULT_AXES_SWAP,
    zero_pelvis: bool = DEFAULT_ZERO_PELVIS,
    force_zero_pelvis: bool = DEFAULT_FORCE_ZERO_PELVIS,
    custom_marker_names: list[str] | None = None,
    load_weights_only: bool = False
):
    """
    Convert SMPL parameter .pt file to individual .trc files per person ID.

    Args:
        pt_file: Path to input .pt file containing SMPL parameters.
        trc_file_template: Template for output .trc filenames with {person_id}.
        fps: Frames per second for the motion data.
        units_mm: If True, output in millimeters; if False, in meters.
        axes_swap: Coordinate system transformation parameters.
        zero_pelvis: If True, move each person's first frame pelvis to origin.
        force_zero_pelvis: If True, force pelvis to (0,0,0) for all frames
                          of each person. Applied after zero_pelvis.
        custom_marker_names: Custom marker names list. If None, uses
                           SMPL_STANDARD_JOINT_NAMES.
        load_weights_only: Security parameter for torch.load.
    """
    print(f"Starting multi-person conversion: {pt_file}")
    full_data = load_pt(pt_file, weights_only=load_weights_only)

    smpl_model = SMPLKinematics().eval()
    smpl_model.to("cpu")
    

    # Wrap data in Pandas DataFrame for easier sorting and grouping
    # While working with PyTorch tensors directly might be more efficient,
    # Pandas provides convenient sorting and grouping functionality
    dataframe_records = []
    for i in range(full_data['id'].shape[0]):
        dataframe_records.append({
            "id": full_data["id"][i].item(),
            "frame_idx": full_data["frame_idx"][i].item(),
            "pose": full_data["pose"][i],
            "trans": full_data["trans"][i],
            "betas": full_data["betas"][i]
        })
    df = pd.DataFrame(dataframe_records)

    unique_ids = sorted(df["id"].unique())
    print(f"Found {len(unique_ids)} unique person IDs: {unique_ids}")

    for person_id in unique_ids:
        print(f"\nProcessing data for person ID: {person_id}")
        person_df = df[df["id"] == person_id].sort_values(by="frame_idx")

        if person_df.empty:
            print(f"No data found for person ID: {person_id}. Skipping.")
            continue

        # Reconstruct tensors from sorted data
        person_pose = torch.stack(person_df["pose"].tolist())
        person_trans = torch.stack(person_df["trans"].tolist())
        # Use first frame betas (assumed constant across all frames for a person)
        person_betas_first_frame = person_df["betas"].iloc[0]

        person_data_for_smpl = {
            "pose": person_pose,
            "trans": person_trans,
            "betas": person_betas_first_frame  # (10,) shaped tensor
        }

        print(f"  Person {person_id}: {person_pose.shape[0]} frames.")
        print(f"  Pose shape: {person_pose.shape}, Trans shape: "
              f"{person_trans.shape}, Betas (first frame): "
              f"{person_betas_first_frame.shape}")

        joints = smpl_kinematics_to_joints(person_data_for_smpl, smpl_model)

        # Generate output filename for each person ID
        base_trc_path = Path(trc_file_template)
        # Convert person_id to string (may be integer)
        actual_trc_file = base_trc_path.with_name(
            base_trc_path.name.format(person_id=str(person_id))
        )

        # Use custom marker names if provided, otherwise use SMPL standard names
        marker_names_to_use = (custom_marker_names if custom_marker_names
                              is not None else SMPL_STANDARD_JOINT_NAMES)

        write_trc(
            joints,
            actual_trc_file,
            fps,
            units_mm=units_mm,
            marker_names=marker_names_to_use,
            axes_swap=axes_swap,
            zero_pelvis=zero_pelvis,
            force_zero_pelvis=force_zero_pelvis
        )

    print("\nMulti-person conversion complete.")

if __name__ == "__main__":
    # Select file using GUI dialog
    file_path = filedialog.askopenfilename(
        title="Select SMPL .pt file",
        filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
    )

    pt_file = Path(file_path) if file_path else Path(DEFAULT_PT_FILE)
    trc_template = f"{pt_file.stem}_OpenSim_P{{person_id}}.trc"

    main(pt_file=pt_file, trc_file_template=trc_template)