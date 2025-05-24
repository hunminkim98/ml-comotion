# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Demo CoMotion with a video file or a directory of images."""

import logging
import os
import shutil
import tempfile
import pickle
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

from comotion_demo.models import comotion
from comotion_demo.utils import dataloading, helper
from comotion_demo.utils import track as track_utils
from comotion_demo.utils import smpl_kinematics

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.billboard import Billboard
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera

    comotion_model_dir = Path(comotion.__file__).parent
    CONFIG.smplx_models = os.path.join(comotion_model_dir, "../data")
    CONFIG.window_type = "pyqt6"
    aitviewer_available = True

except ModuleNotFoundError:
    print(
        "WARNING: Skipped aitviewer import, ensure it is installed to run visualization."
    )
    aitviewer_available = False

from vispy import scene
from vispy.app import Timer
from vispy.scene.visuals import Mesh as VispyMesh
from vispy.geometry import create_plane

# Load SMPL face indices for mesh visualization
smpl_dir = os.path.join(os.path.dirname(smpl_kinematics.__file__), "../data/smpl")
smpl_model_path = f"{smpl_dir}/SMPL_NEUTRAL.pkl"
with open(smpl_model_path, "rb") as f:
    smpl_model = pickle.load(f, encoding="latin1")
    smpl_faces = smpl_model['f']  # Triangle faces for mesh visualization

# SMPL joint mapping for height calculation
SMPL_JOINT_NAMES = {
    0: 'pelvis', 1: 'left_hip', 2: 'right_hip', 3: 'spine1',
    4: 'left_knee', 5: 'right_knee', 6: 'spine2', 7: 'left_ankle',
    8: 'right_ankle', 9: 'spine3', 10: 'left_foot', 11: 'right_foot',
    12: 'neck', 13: 'left_collar', 14: 'right_collar', 15: 'head',
    16: 'left_shoulder', 17: 'right_shoulder', 18: 'left_elbow',
    19: 'right_elbow', 20: 'left_wrist', 21: 'right_wrist',
    22: 'left_hand', 23: 'right_hand'
}

# Height calculation pairs based on SMPL structure
HEIGHT_PAIRS = [
    # Lower body segments (ankle to hip)
    ('left_ankle', 'left_knee'), ('left_knee', 'left_hip'),
    ('right_ankle', 'right_knee'), ('right_knee', 'right_hip'),
    # Spine segments (hip to shoulder)  
    ('pelvis', 'spine1'), ('spine1', 'spine2'), ('spine2', 'spine3'),
    # Upper body (spine to head)
    ('spine3', 'neck'), ('neck', 'head'),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global device declaration - will be overridden in main() function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_mps = torch.mps.is_available()


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return torch.norm(point2 - point1, dim=-1)


def calculate_person_height_from_joints(joints_3d, valid_mask=None):
    """
    Calculate person height from 3D joints using SMPL structure.
    
    Args:
        joints_3d: torch.Tensor of shape [24, 3] or [N, 24, 3] - SMPL joint positions
        valid_mask: Optional mask for valid joints
    
    Returns:
        height: Estimated height in meters
    """
    if joints_3d.dim() == 3:
        joints_3d = joints_3d[0]  # Take first person if multiple
    
    # Create joint name to index mapping
    name_to_idx = {v: k for k, v in SMPL_JOINT_NAMES.items()}
    
    # Calculate segment lengths
    segment_lengths = []
    for joint1_name, joint2_name in HEIGHT_PAIRS:
        if joint1_name in name_to_idx and joint2_name in name_to_idx:
            idx1, idx2 = name_to_idx[joint1_name], name_to_idx[joint2_name]
            
            if valid_mask is None or (valid_mask[idx1] and valid_mask[idx2]):
                segment_length = euclidean_distance(joints_3d[idx1], joints_3d[idx2])
                segment_lengths.append(segment_length)
    
    if not segment_lengths:
        return None
    
    # Sum all segments for total height
    total_height = sum(segment_lengths)
    
    # Add approximate foot height (10cm) and head top (15cm from neck)
    foot_height = 0.10  # 10 cm
    head_extension = 0.15  # 15 cm from neck to head top
    
    estimated_height = total_height + foot_height + head_extension
    return estimated_height.item()


def calculate_person_height_from_smpl(smpl_kinematics_model, betas, pose, trans=None):
    """
    Calculate person height from SMPL parameters.
    
    Args:
        smpl_kinematics_model: SMPLKinematics model
        betas: Body shape parameters
        pose: Body pose parameters  
        trans: Optional translation
    
    Returns:
        height: Estimated height in meters
    """
    try:
        # Generate joints from SMPL parameters
        with torch.no_grad():
            device = smpl_kinematics_model.joint_template.device
            betas_tensor = betas.unsqueeze(0).to(device) if betas.dim() == 1 else betas.to(device)
            pose_tensor = pose.unsqueeze(0).to(device) if pose.dim() == 1 else pose.to(device)
            
            joints = smpl_kinematics_model.forward(
                betas_tensor,
                pose_tensor,
                trans=trans,
                output_format="joints"
            )
            
            # Calculate height from joints
            height = calculate_person_height_from_joints(joints.cpu())
            return height
            
    except Exception as e:
        logging.warning(f"Failed to calculate height from SMPL: {e}")
        return None


def apply_height_based_depth_correction(detections, user_height_m, smpl_kinematics_model, 
                                       previous_scale=None, smooth_factor=0.8):
    """
    Apply height-based depth correction to detections.
    
    Args:
        detections: Detection results containing betas, pose, trans
        user_height_m: User-specified height in meters
        smpl_kinematics_model: SMPL model for height calculation
        previous_scale: Previous scale factor for smoothing
        smooth_factor: Smoothing factor for temporal consistency
    
    Returns:
        corrected_detections: Corrected detection results
        current_scale: Current scale factor
    """
    corrected_detections = detections.copy()
    
    if not all(key in detections for key in ["betas", "pose", "trans"]):
        return corrected_detections, 1.0
    
    try:
        # Extract parameters for first person
        betas = detections["betas"]
        pose = detections["pose"] 
        trans = detections["trans"]
        
        # Handle different tensor dimensions
        if betas.ndim == 3 and betas.shape[0] > 0 and betas.shape[1] > 0:
            betas = betas[0, 0].cpu()
            pose = pose[0, 0].cpu()
            trans = trans[0, 0].cpu()
        elif betas.ndim == 2 and betas.shape[0] > 0:
            betas = betas[0].cpu()
            pose = pose[0].cpu() 
            trans = trans[0].cpu()
        else:
            return corrected_detections, 1.0
        
        # Calculate estimated height from SMPL
        estimated_height = calculate_person_height_from_smpl(
            smpl_kinematics_model, betas, pose
        )
        
        if estimated_height is None or estimated_height <= 0:
            return corrected_detections, 1.0
        
        # Calculate scale factor
        scale_factor = user_height_m / estimated_height
        
        # Apply temporal smoothing
        if previous_scale is not None:
            scale_factor = smooth_factor * previous_scale + (1 - smooth_factor) * scale_factor
        
        # Apply depth correction by scaling translation Z component
        corrected_trans = trans.clone()
        corrected_trans[2] *= scale_factor  # Scale depth (Z-axis)
        
        # Update detections with corrected translation
        original_shape = detections["trans"].shape
        if len(original_shape) == 3:  # [batch, person, 3]
            corrected_detections["trans"] = detections["trans"].clone()
            corrected_detections["trans"][0, 0] = corrected_trans.to(detections["trans"].device)
        elif len(original_shape) == 2:  # [person, 3]  
            corrected_detections["trans"] = detections["trans"].clone()
            corrected_detections["trans"][0] = corrected_trans.to(detections["trans"].device)
        
        # logging.info(f"Height correction: est={estimated_height:.2f}m → target={user_height_m:.2f}m (scale={scale_factor:.3f})")
        
        return corrected_detections, scale_factor
        
    except Exception as e:
        logging.warning(f"Height-based depth correction failed: {e}")
        return corrected_detections, 1.0


def prepare_scene(viewer, width, height, K, image_paths, fps=30):
    """Prepare the scene for AITViewer rendering."""
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    extrinsics = np.eye(4)[:3]

    # Initialize camera
    cam = OpenCVCamera(K, extrinsics, cols=width, rows=height, viewer=viewer)
    viewer.scene.add(cam)
    viewer.scene.camera.position = [0, 0, -5]
    viewer.scene.camera.target = [0, 0, 10]
    viewer.auto_set_camera_target = False
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps

    # "billboard" display for video frames
    billboard = Billboard.from_camera_and_distance(
        cam, 100.0, cols=width, rows=height, textures=image_paths
    )
    viewer.scene.add(billboard)


def add_pose_to_scene(
    viewer,
    smpl_layer,
    betas,
    pose,
    trans,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    color_ref=None,
):
    """Add estimated poses to the rendered scene."""
    if betas.ndim == 2:
        betas = betas[None]
        pose = pose[None]
        trans = trans[None]

    poses_root = pose[..., :3]
    poses_body = pose[..., 3:]
    max_people = pose.shape[1]

    if (betas != 0).any():
        for person_idx in range(max_people):
            if color_ref is None:
                person_color = color
            else:
                person_color = color_ref[person_idx % len(color_ref)] * 0.4 + 0.3
            person_color = [c_ for c_ in person_color] + [alpha]

            valid_vals = (betas[:, person_idx] != 0).any(-1)
            idx_range = valid_vals.nonzero()
            if len(idx_range) > 0:
                trans[~valid_vals][..., 2] = -10000
                viewer.scene.add(
                    SMPLSequence(
                        smpl_layer=smpl_layer,
                        betas=betas[:, person_idx],
                        poses_root=poses_root[:, person_idx],
                        poses_body=poses_body[:, person_idx],
                        trans=trans[:, person_idx],
                        color=person_color,
                    )
                )


def visualize_poses(
    input_path,
    cache_path,
    video_path,
    start_frame,
    num_frames,
    frameskip=1,
    color=(0.6, 0.6, 0.6),
    alpha=1,
    fps=30,
):
    """Visualize SMPL poses."""
    logging.info(f"Rendering SMPL video: {input_path}")

    # Prepare temporary directory with saved images
    tmp_vis_dir = Path(tempfile.mkdtemp())

    frame_idx = 0
    image_paths = []
    for image, K in dataloading.yield_image_and_K(
        input_path, start_frame, num_frames, frameskip
    ):
        image_height, image_width = image.shape[-2:]
        image = dataloading.convert_tensor_to_image(image)
        image_paths.append(f"{tmp_vis_dir}/{frame_idx:06d}.jpg")
        Image.fromarray(image).save(image_paths[-1])
        frame_idx += 1

    # Initialize viewer
    viewer = HeadlessRenderer(size=(image_width, image_height))

    if dataloading.is_a_video(input_path):
        fps = int(dataloading.get_input_video_fps(input_path))

    prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), image_paths, fps)

    # Prepare SMPL poses
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    if not cache_path.exists():
        logging.warning("No detections found.")
    else:
        preds = torch.load(cache_path, weights_only=False, map_location="cpu")
        track_subset = track_utils.query_range(preds, 0, frame_idx - 1)
        id_lookup = track_subset["id"].max(0)[0]
        color_ref = helper.color_ref[id_lookup % len(helper.color_ref)]
        if len(id_lookup) == 1:
            color_ref = [color_ref]

        betas = track_subset["betas"]
        pose = track_subset["pose"]
        trans = track_subset["trans"]

        add_pose_to_scene(
            viewer, smpl_layer, betas, pose, trans, color, alpha, color_ref
        )

    # Save rendered scene
    viewer.save_video(
        video_dir=str(video_path),
        output_fps=fps,
        ensure_no_overwrite=False,
    )

    # Remove temporary directory
    shutil.rmtree(tmp_vis_dir)


def run_detection(input_path, cache_path, skip_visualization=False, model=None, user_height_m=None):
    """Run model and visualize detections on single image."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    # Load image
    image = np.array(Image.open(input_path))
    image = dataloading.convert_image_to_tensor(image)
    K = dataloading.get_default_K(image)
    cropped_image, cropped_K = dataloading.prepare_network_inputs(image, K, device)

    # Get detections
    detections = model.detection_model(cropped_image, cropped_K)
    detections = comotion.detect.decode_network_outputs(
        K.to(device),
        model.smpl_decoder,
        detections,
        std=0.15,  # Adjust NMS sensitivity
        conf_thr=0.25,  # Adjust confidence cutoff
    )

    detections = {k: v[0].cpu() for k, v in detections.items()}
    
    # Apply height-based depth correction if user height is provided
    if user_height_m is not None:
        smpl_kinematics_model = smpl_kinematics.SMPLKinematics()
        if torch.cuda.is_available():
            smpl_kinematics_model = smpl_kinematics_model.cuda()
        smpl_kinematics_model.eval()
        
        detections, _ = apply_height_based_depth_correction(
            detections, user_height_m, smpl_kinematics_model
        )
    
    torch.save(detections, cache_path)

    if not skip_visualization:
        # Initialize viewer
        image_height, image_width = image.shape[-2:]
        viewer = HeadlessRenderer(size=(image_width, image_height))
        prepare_scene(
            viewer, image_width, image_height, K.cpu().numpy(), [str(input_path)]
        )

        # Prepare SMPL poses
        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
        add_pose_to_scene(
            viewer,
            smpl_layer,
            detections["betas"],
            detections["pose"],
            detections["trans"],
        )

        # Save rendered scene
        viewer.save_frame(str(cache_path).replace(".pt", ".png"))


def track_poses(
    input_path, cache_path, start_frame, num_frames, frameskip=1, model=None
):
    """Track poses over a video or a directory of images."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    detections = []
    tracks = []

    initialized = False
    for image, K in tqdm(
        dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip),
        desc="Running CoMotion",
    ):
        if not initialized:
            image_res = image.shape[-2:]
            model.init_tracks(image_res)
            initialized = True

        detection, track = model(image, K, use_mps=use_mps)
        detection = {k: v.cpu() for k, v in detection.items()}
        track = track.cpu()
        detections.append(detection)
        tracks.append(track)

    detections = {k: [d[k] for d in detections] for k in detections[0].keys()}
    tracks = torch.stack(tracks, 1)
    tracks = {k: getattr(tracks, k) for k in ["id", "pose", "trans", "betas"]}

    track_ref = track_utils.cleanup_tracks(
        {"detections": detections, "tracks": tracks},
        K,
        model.smpl_decoder.cpu(),
        min_matched_frames=1,
    )
    if track_ref:
        frame_idxs, track_idxs = track_utils.convert_to_idxs(
            track_ref, tracks["id"][0].squeeze(-1).long()
        )
        preds = {k: v[0, frame_idxs, track_idxs] for k, v in tracks.items()}
        preds["id"] = preds["id"].squeeze(-1).long()
        preds["frame_idx"] = frame_idxs
        torch.save(preds, cache_path)

        # Save bounding box tracks in MOT format
        bboxes = track_utils.bboxes_from_smpl(
            model.smpl_decoder,
            {k: preds[k] for k in ["betas", "pose", "trans"]},
            image_res,
            K,
        )
        with open(str(cache_path).replace(".pt", ".txt"), "w") as f:
            f.write(track_utils.convert_to_mot(preds["id"], preds["frame_idx"], bboxes))


@click.command()
@click.option(
    "-i",
    "--input-path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the input video, a directory of images, or a single input image.",
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(exists=False, path_type=Path),
    help="Path to the output directory.",
)
@click.option(
    "--webcam",
    is_flag=True,
    help="Use webcam for real-time processing and visualization.",
)
@click.option(
    "-s",
    "--start-frame",
    default=0,
    type=int,
    help="Frame to start with.",
)
@click.option(
    "-n",
    "--num-frames",
    default=1_000_000_000,
    type=int,
    help="Number of frames to process.",
)
@click.option(
    "--skip-visualization",
    is_flag=True,
    help="Whether to skip rendering the output SMPL meshes.",
)
@click.option(
    "--frameskip",
    default=1,
    type=int,
    help="Subsample video frames (e.g. frameskip=2 processes every other frame).",
)
@click.option(
    "--user-height",
    type=float,
    help="User height in meters for depth correction (e.g. 1.75 for 175cm).",
)
def main(
    input_path, output_dir, webcam, start_frame, num_frames, skip_visualization, frameskip, user_height
):
    """Demo entry point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = input_path.stem if input_path else "webcam"
    skip_visualization = skip_visualization | (not aitviewer_available)

    cache_path = output_dir / f"{input_name}.pt"
    
    # Define the device here so it's available for all code paths
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if webcam:
        logging.info("Using webcam for real-time processing.")
        cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

        if not cap.isOpened():
            logging.error("Cannot open webcam.")
            return
        
        # Set webcam resolution for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual webcam resolution
        webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # logging.info(f"Webcam resolution: {webcam_width}x{webcam_height}")

        # Initialize Vispy Canvas for 3D visualization (720p for high quality recording)
        canvas_width, canvas_height = 1280, 720  # 720p resolution for 3D recording
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(canvas_width, canvas_height))
        canvas.native.activateWindow()  # 창에 포커스 강제 설정
        canvas.native.raise_()          # 창을 맨 앞으로 가져오기
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'  # Use a turntable camera for 3D navigation
        view.camera.fov = 45
        view.camera.distance = 12  # Optimal distance to see 15x15 grid and person
        view.camera.center = (0, 0, 0.5)  # Slightly above ground level
        view.camera.elevation = 20  # Look down at a slight angle to see the grid
        
        # logging.info(f"3D visualization resolution: {canvas_width}x{canvas_height}")

        # Add a Markers visual for 3D keypoints
        # Initialize with a single dummy point, it will be updated
        keypoints_3d_vis = scene.visuals.Markers(parent=view.scene)
        # Add multiple Mesh visuals for SMPL body meshes (support up to 6 people)
        max_people = 10
        smpl_mesh_visuals = []
        smpl_vertices_visuals = []
        for i in range(max_people):
            mesh_vis = VispyMesh(parent=view.scene)
            mesh_vis.visible = False  # Start with hidden mesh
            smpl_mesh_visuals.append(mesh_vis)
            
            vertices_vis = scene.visuals.Markers(parent=view.scene)
            vertices_vis.visible = False  # Start with hidden vertices
            smpl_vertices_visuals.append(vertices_vis)
        
        # Initialize height-related variables first
        user_height_m = user_height if user_height else 1.75  # Default height in meters (175cm)
        height_correction_enabled = user_height is not None
        previous_scale_factor = None
        
        # Pelvis centering control - disabled when height correction is active
        pelvis_centering_enabled = not height_correction_enabled
        
        # Add grid floor for biomechanical feel
        grid_size = 15.0  # 15x15 meter grid
        grid_divisions = 38  # 38x38 divisions (40cm spacing)
        
        # Create grid lines
        grid_lines = []
        step = grid_size / grid_divisions
        
        # Grid height offset - 키의 절반만큼 아래로 (골반에서 발까지)
        grid_z_offset = (-user_height_m / 2) - 0.15 # 사용자 키의 절반만큼 아래에 그리드 배치 + 15cm 아래로 (foot 높이)
        
        # Horizontal lines
        for i in range(grid_divisions + 1):
            y = -grid_size/2 + i * step
            grid_lines.extend([
                [-grid_size/2, y, grid_z_offset],
                [grid_size/2, y, grid_z_offset]
            ])
        
        # Vertical lines  
        for i in range(grid_divisions + 1):
            x = -grid_size/2 + i * step
            grid_lines.extend([
                [x, -grid_size/2, grid_z_offset],
                [x, grid_size/2, grid_z_offset]
            ])
        
        # Convert to numpy array and create line segments
        grid_lines = np.array(grid_lines, dtype=np.float32)
        
        # Create line visual for grid with biomechanical styling
        grid_visual = scene.visuals.Line(
            pos=grid_lines,
            color=(0.2, 0.4, 0.6, 0.7),  # Blue-gray for biomechanical feel
            width=1.0,
            parent=view.scene,
            connect='segments'
        )
        
        # Add thicker main grid lines every 2 meters
        main_grid_lines = []
        for i in range(0, grid_divisions + 1, 5):  # Every 5th line (2 meters)
            y = -grid_size/2 + i * step
            x = -grid_size/2 + i * step
            # Horizontal main lines
            main_grid_lines.extend([
                [-grid_size/2, y, grid_z_offset],
                [grid_size/2, y, grid_z_offset]
            ])
            # Vertical main lines
            main_grid_lines.extend([
                [x, -grid_size/2, grid_z_offset],
                [x, grid_size/2, grid_z_offset]
            ])
        
        main_grid_lines = np.array(main_grid_lines, dtype=np.float32)
        main_grid_visual = scene.visuals.Line(
            pos=main_grid_lines,
            color=(0.3, 0.5, 0.7, 0.9),  # Brighter blue for main lines
            width=2.0,
            parent=view.scene,
            connect='segments'
        )
        
        # Add coordinate axes for reference at grid level (scaled for 15x15m grid)
        axes_lines = np.array([
            # X-axis (red) - 0.4m long
            [0, 0, grid_z_offset], [0.4, 0, grid_z_offset],
            # Y-axis (green) - 0.4m long
            [0, 0, grid_z_offset], [0, 0.4, grid_z_offset],
            # Z-axis (blue) - 0.4m long
            [0, 0, grid_z_offset], [0, 0, grid_z_offset + 0.4]
        ], dtype=np.float32)
        
        axes_colors = np.array([
            [1, 0, 0, 0.8], [1, 0, 0, 0.8],  # Red for X (좌우)
            [0, 0, 1, 0.8], [0, 0, 1, 0.8],  # Blue for Y (위아래, 수직축)
            [0, 1, 0, 0.8], [0, 1, 0, 0.8]   # Green for Z (앞뒤)
        ], dtype=np.float32)
        
        axes_visual = scene.visuals.Line(
            pos=axes_lines,
            color=axes_colors,
            width=8.0,  # Thicker lines for better visibility in large grid
            parent=view.scene,
            connect='segments'
        )

        # Initialize SMPLKinematics model for mesh generation
        smpl_kinematics_model = smpl_kinematics.SMPLKinematics()
        if torch.cuda.is_available():
            smpl_kinematics_model = smpl_kinematics_model.cuda()
        smpl_kinematics_model.eval()  # Set to evaluation mode
        
        model = comotion.CoMotion(use_coreml=use_mps)
        model.to(device).eval()
        
        # Track current view mode: 0=keypoints, 1=mesh, 2=vertices
        current_view_mode = 1  # Start with mesh view
        
        # Store current vertex data for visualization
        current_vertices_data = None
        
        # Grid and axes visibility control
        grid_visible = True
        axes_visible = True
        
        # Session recording variables
        session_detections = []  # Store all detections
        session_frames_2d = []   # Store 2D visualization frames
        session_frames_3d = []   # Store 3D visualization frames
        frame_count = 0
        
        # FPS calculation variables
        import time
        session_start_time = time.time()
        last_fps_calc_time = session_start_time
        fps_calc_frame_count = 0
        current_fps = 30.0  # Initial estimate

        # Initialize model tracks once with image resolution
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to get first frame for track initialization.")
            cap.release()
            cv2.destroyAllWindows()
            canvas.close()
            return
        temp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_image_tensor = dataloading.convert_image_to_tensor(temp_image)
        model.init_tracks(temp_image_tensor.shape[-2:])

        # Exit flag for clean shutdown
        should_exit = False
        
        # Add key controls
        @canvas.events.key_press.connect
        def on_key_press(event):
            nonlocal current_view_mode, current_vertices_data, user_height_m, height_correction_enabled, pelvis_centering_enabled, grid_visible, axes_visible, should_exit
            if event.key == 'space' or event.key == '1':  # 스페이스바 또는 1키
                # 키포인트로 전환
                current_view_mode = 0
            elif event.key == '2':  # 2키
                # 메시로 전환
                current_view_mode = 1
            elif event.key == '3':  # 3키
                # 버텍스로 전환
                current_view_mode = 2
            elif event.key == 'h':  # H키로 키 보정 토글
                height_correction_enabled = not height_correction_enabled
                # Auto-disable pelvis centering when height correction is first enabled
                if height_correction_enabled:
                    pelvis_centering_enabled = False
                    print(f"Height correction: ON (pelvis centering auto-disabled)")
                else:
                    print(f"Height correction: OFF")
                print(f"Pelvis centering: {'ON' if pelvis_centering_enabled else 'OFF'}")
                update_text_display()
            elif event.key == '=' or event.key == '+':  # 키 증가 (+ 키)
                user_height_m += 0.05
                print(f"User height: {user_height_m:.2f}m")
                update_grid_position()
                update_text_display()
            elif event.key == '-':  # 키 감소 (- 키)
                user_height_m = max(1.0, user_height_m - 0.05)  # 최소 1m
                print(f"User height: {user_height_m:.2f}m")
                update_grid_position()
                update_text_display()
            elif event.key == 'p':  # P키로 골반 중심화 토글
                pelvis_centering_enabled = not pelvis_centering_enabled
                print(f"Pelvis centering: {'ON' if pelvis_centering_enabled else 'OFF'}")
                update_text_display()
            elif event.key == 'g':  # G키로 그리드 토글
                grid_visible = not grid_visible
                grid_visual.visible = grid_visible
                main_grid_visual.visible = grid_visible
                print(f"Grid: {'ON' if grid_visible else 'OFF'}")
                update_text_display()
            elif event.key == 'a':  # A키로 축 토글
                axes_visible = not axes_visible
                axes_visual.visible = axes_visible
                print(f"Axes: {'ON' if axes_visible else 'OFF'}")
                update_text_display()
            elif event.key == 'q' or event.key == 'Escape':  # Q키 또는 ESC키로 종료
                print("Exit requested from 3D window")
                should_exit = True
        
        # Add instruction text to the screen
        height_status = "ON" if height_correction_enabled else "OFF"
        pelvis_status = "ON" if pelvis_centering_enabled else "OFF"
        
        # Display current mode
        if height_correction_enabled and not pelvis_centering_enabled:
            mode_info = "Mode: Height correction (Real 3D positions)"
        elif not height_correction_enabled and pelvis_centering_enabled:
            mode_info = "Mode: Traditional visualization (Pelvis centered)"
        elif height_correction_enabled and pelvis_centering_enabled:
            mode_info = "Mode: Height correction priority"
        else:
            mode_info = "Mode: Raw 3D data (No position corrections)"
        
        grid_status = "ON" if grid_visible else "OFF"
        axes_status = "ON" if axes_visible else "OFF"
        text_vis = scene.visuals.Text(
            text=f'1=Keypoints 2=Mesh 3=Vertices | G:{grid_status} A:{axes_status}\n'
                 f'Height:{height_status} Pelvis:{pelvis_status} | {mode_info}\n'
                 f'H/P=toggle +/-=height({user_height_m:.2f}m) Q=exit',
            color='white',
            pos=(20, 100),  # 아래쪽으로 이동
            anchor_x='left',
            anchor_y='top',
            font_size=10,  # 폰트 크기 조정
            parent=canvas.scene
        )
        
        def update_view_mode():
            """Update visibility based on current view mode"""
            nonlocal current_vertices_data
            if current_view_mode == 0:  # Keypoints
                keypoints_3d_vis.visible = True
                for mesh_vis in smpl_mesh_visuals:
                    mesh_vis.visible = False
                for vertices_vis in smpl_vertices_visuals:
                    vertices_vis.visible = False
            elif current_view_mode == 1:  # Mesh
                keypoints_3d_vis.visible = False
                for i, mesh_vis in enumerate(smpl_mesh_visuals):
                    mesh_vis.visible = i < len(current_vertices_data) if current_vertices_data else False
                for vertices_vis in smpl_vertices_visuals:
                    vertices_vis.visible = False
            elif current_view_mode == 2:  # Vertices
                keypoints_3d_vis.visible = False
                for mesh_vis in smpl_mesh_visuals:
                    mesh_vis.visible = False
                for i, vertices_vis in enumerate(smpl_vertices_visuals):
                    if current_vertices_data and i < len(current_vertices_data):
                        vertices_vis.visible = True
                        # Update vertices visualization with current data
                        person_color = person_colors[i % len(person_colors)]
                        vertices_vis.set_data(
                            current_vertices_data[i],
                            edge_color='white',  # White edge for better visibility
                            face_color=person_color + (1.0,),  # Person-specific color with full alpha
                            size=4.0,  # Much larger point size for better visibility
                            edge_width=0.1  # Thicker edge
                        )
                    else:
                        vertices_vis.visible = False
        
        def update_grid_position():
            """Update grid position based on current user height"""
            nonlocal grid_visual, main_grid_visual, axes_visual
            new_grid_z_offset = (-user_height_m * 0.2)  # 키의 20% 높이
            
            # Create new grid lines with updated position
            grid_lines = []
            # Horizontal lines
            for i in range(grid_divisions + 1):
                y = -grid_size/2 + i * step
                grid_lines.extend([
                    [-grid_size/2, y, new_grid_z_offset],
                    [grid_size/2, y, new_grid_z_offset]
                ])
            
            # Vertical lines  
            for i in range(grid_divisions + 1):
                x = -grid_size/2 + i * step
                grid_lines.extend([
                    [x, -grid_size/2, new_grid_z_offset],
                    [x, grid_size/2, new_grid_z_offset]
                ])
            
            grid_lines = np.array(grid_lines, dtype=np.float32)
            grid_visual.set_data(pos=grid_lines)
            
            # Update main grid lines
            main_grid_lines = []
            for i in range(0, grid_divisions + 1, 5):  # Every 5th line (2 meters)
                y = -grid_size/2 + i * step
                x = -grid_size/2 + i * step
                # Horizontal main lines
                main_grid_lines.extend([
                    [-grid_size/2, y, new_grid_z_offset],
                    [grid_size/2, y, new_grid_z_offset]
                ])
                # Vertical main lines
                main_grid_lines.extend([
                    [x, -grid_size/2, new_grid_z_offset],
                    [x, grid_size/2, new_grid_z_offset]
                ])
            
            main_grid_lines = np.array(main_grid_lines, dtype=np.float32)
            main_grid_visual.set_data(pos=main_grid_lines)
            
            # Update axes position to match grid level (scaled for 15x15m grid)
            new_axes_lines = np.array([
                # X-axis (red) - 0.4m long
                [0, 0, new_grid_z_offset], [0.4, 0, new_grid_z_offset],
                # Y-axis (green) - 0.4m long
                [0, 0, new_grid_z_offset], [0, 0.4, new_grid_z_offset],
                # Z-axis (blue) - 0.4m long
                [0, 0, new_grid_z_offset], [0, 0, new_grid_z_offset + 0.4]
            ], dtype=np.float32)
            axes_visual.set_data(pos=new_axes_lines)
        
        def update_text_display():
            """Update the instruction text with current height"""
            height_status = "ON" if height_correction_enabled else "OFF"
            pelvis_status = "ON" if pelvis_centering_enabled else "OFF"
            
            # Display current mode
            if height_correction_enabled and not pelvis_centering_enabled:
                mode_info = "Mode: Height correction (Real 3D positions)"
            elif not height_correction_enabled and pelvis_centering_enabled:
                mode_info = "Mode: Traditional visualization (Pelvis centered)"
            elif height_correction_enabled and pelvis_centering_enabled:
                mode_info = "Mode: Height correction priority"
            else:
                mode_info = "Mode: Raw 3D data (No position corrections)"
            
            grid_status = "ON" if grid_visible else "OFF"
            axes_status = "ON" if axes_visible else "OFF"
            
            # Update text content - more compact
            new_text = (f'1=Keypoints 2=Mesh 3=Vertices | G:{grid_status} A:{axes_status}\n'
                       f'Height:{height_status} Pelvis:{pelvis_status} | {mode_info}\n'
                       f'H/P=toggle +/-=height({user_height_m:.2f}m) Q/ESC=exit')
            text_vis.text = new_text
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Cannot receive frame (stream end?). Exiting ...")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_tensor = dataloading.convert_image_to_tensor(image)
            K = dataloading.get_default_K(image_tensor)

            cropped_image, cropped_K = dataloading.prepare_network_inputs(image_tensor, K, device)
            
            detection_output = model.detection_model(cropped_image, cropped_K)
            detections = comotion.detect.decode_network_outputs(
                K.to(device),
                model.smpl_decoder,
                detection_output,
                std=0.15,
                conf_thr=0.25,
            )
            
            # Apply height-based depth correction if enabled
            if height_correction_enabled and detections:
                detections, previous_scale_factor = apply_height_based_depth_correction(
                    detections, user_height_m, smpl_kinematics_model, 
                    previous_scale_factor, smooth_factor=0.8
                )
            
            # Store detections for session recording
            if detections:
                # Convert to CPU and store
                detection_to_store = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in detections.items()}
                detection_to_store['frame_idx'] = frame_count
                session_detections.append(detection_to_store)

            processed_frame = frame.copy()
            if detections and "pred_2d" in detections:
                if detections["pred_2d"].ndim == 4:
                    keypoints_2d = detections["pred_2d"][0].cpu().numpy()
                elif detections["pred_2d"].ndim == 3:
                    keypoints_2d = detections["pred_2d"].cpu().numpy()
                else:
                    keypoints_2d = None

                if keypoints_2d is not None:
                    for person_keypoints in keypoints_2d:
                        for kp in person_keypoints:
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(processed_frame, (x, y), 3, (0, 255, 0), -1)
            
            cv2.imshow('Webcam Feed', processed_frame)
            
            # Store 2D visualization frame
            session_frames_2d.append(processed_frame.copy())

            # Update Vispy 3D visualization
            if detections and "pred_3d" in detections:
                all_points_3d_data = [] # 모든 사람의 데이터
                pred_3d_tensor = detections["pred_3d"]

                if pred_3d_tensor.ndim == 4 and pred_3d_tensor.shape[0] > 0 and pred_3d_tensor.shape[1] > 0: # Check batch and people dimensions
                    # Multiple people in batch
                    for person_idx in range(pred_3d_tensor.shape[1]):
                        person_data = pred_3d_tensor[0, person_idx].cpu().numpy()
                        if not np.allclose(person_data, 0):  # Skip empty detections
                            all_points_3d_data.append(person_data)
                elif pred_3d_tensor.ndim == 3: # No batch dimension
                    if pred_3d_tensor.shape[0] > 0: # Check if people/keypoint sets are detected (dim 0)
                        if pred_3d_tensor.shape[1] == smpl_kinematics.NUM_PARTS_AUX and pred_3d_tensor.shape[2] == 3:
                            # (num_people, num_joints, 3)
                            for person_idx in range(pred_3d_tensor.shape[0]):
                                person_data = pred_3d_tensor[person_idx].cpu().numpy()
                                if not np.allclose(person_data, 0):  # Skip empty detections
                                    all_points_3d_data.append(person_data)
                        elif pred_3d_tensor.shape[0] == smpl_kinematics.NUM_PARTS_AUX and pred_3d_tensor.shape[1] == 3:
                            # Single person case
                            person_data = pred_3d_tensor.cpu().numpy()
                            if not np.allclose(person_data, 0):
                                all_points_3d_data.append(person_data)

                # Process all detected people
                all_points_to_visualize = []
                person_colors = [
                    (1.0, 1.0, 1.0),    # White
                    (1.0, 1.0, 0.0),    # Yellow
                    (1.0, 0.5, 0.0),    # Orange
                    (0.0, 1.0, 1.0),    # Cyan (sky blue)
                    (1.0, 0.0, 1.0),    # Magenta (bright pink)
                    (0.5, 1.0, 0.0),    # Lime green
                    (1.0, 0.0, 0.0),    # Bright red
                    (0.0, 1.0, 0.0),    # Bright green
                    (1.0, 1.0, 0.5),    # Light yellow
                    (1.0, 0.7, 1.0),    # Light pink
                    (0.7, 1.0, 1.0),    # Light cyan
                    (1.0, 0.8, 0.0),    # Gold
                ]  # Very bright, vivid colors for maximum visibility
                
                for person_idx, points_3d_data_original in enumerate(all_points_3d_data):
                    if points_3d_data_original.ndim == 2 and \
                       points_3d_data_original.shape[0] == smpl_kinematics.NUM_PARTS_AUX and \
                       points_3d_data_original.shape[1] == 3:

                        points_3d_to_visualize = points_3d_data_original.copy()
                        pelvis_index = 0  # SMPL에서 골반은 보통 0번 관절
                        
                        # Case 1: 키 보정 ON, 골반 중심화 OFF - 실제 3D 위치 유지
                        if height_correction_enabled and not pelvis_centering_enabled:
                            # 원시 데이터 유지, 회전만 적용
                            pass
                        
                        # Case 2: 키 보정 OFF, 골반 중심화 ON - 전통적 시각화
                        elif not height_correction_enabled and pelvis_centering_enabled:
                            if points_3d_data_original.shape[0] > pelvis_index:
                                pelvis_position = points_3d_data_original[pelvis_index:pelvis_index+1, :].copy()
                                points_3d_to_visualize = points_3d_data_original - pelvis_position
                        
                        # Case 3: 둘 다 ON - 실제 3D 위치 우선 (키 보정 적용, 골반 중심화 안함)
                        elif height_correction_enabled and pelvis_centering_enabled:
                            # 키 보정이 우선, 골반 중심화 무시
                            pass
                        
                        # Case 4: 둘 다 OFF - 원시 데이터 (위치 처리만 안함)
                        elif not height_correction_enabled and not pelvis_centering_enabled:
                            # 원시 3D 데이터를 그대로 사용 (키 보정, 골반 중심화 안함)
                            pass

                        # 회전 적용 (시각화를 위해 항상 적용)
                        angle = -np.pi / 2
                        rotation_matrix_x = np.array([
                            [1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]
                        ])
                        points_3d_to_visualize = np.dot(points_3d_to_visualize, rotation_matrix_x.T)
                        
                        # Store processed points with color info
                        person_color = person_colors[person_idx % len(person_colors)]
                        all_points_to_visualize.append((points_3d_to_visualize, person_color))
                
                # Update keypoints visualization with multiple people
                if all_points_to_visualize:
                    # Combine all keypoints for visualization
                    all_keypoints = []
                    all_colors = []
                    for points, color in all_points_to_visualize:
                        all_keypoints.append(points)
                        # Create color array for all keypoints of this person with alpha channel
                        person_colors_array = np.tile(list(color) + [1.0], (points.shape[0], 1))  # Add alpha=1.0
                        all_colors.append(person_colors_array)
                    
                    combined_keypoints = np.vstack(all_keypoints)
                    combined_colors = np.vstack(all_colors)
                    keypoints_3d_vis.set_data(
                        combined_keypoints, 
                        edge_color='white',  # White edge for better visibility
                        face_color=combined_colors, 
                        size=6.0,  # Much larger keypoints
                        edge_width=0.1  # Thicker edge
                    )
                else:
                    keypoints_3d_vis.set_data(np.array([[0,0,0]]), edge_color=None, face_color=(1, 1, 1, 0.5), size=0.01)
                
                # Generate and visualize SMPL mesh for all people
                if "betas" in detections and "pose" in detections and "trans" in detections:
                    # Extract SMPL parameters for all people
                    all_people_params = []
                    
                    betas_tensor = detections["betas"]
                    pose_tensor = detections["pose"]
                    trans_tensor = detections["trans"]
                    
                    if betas_tensor is not None and isinstance(betas_tensor, torch.Tensor):
                        if betas_tensor.ndim == 3 and betas_tensor.shape[0] > 0:  # [batch, person, dim]
                            num_people = betas_tensor.shape[1]
                            for person_idx in range(num_people):
                                betas = betas_tensor[0, person_idx].cpu()
                                pose = pose_tensor[0, person_idx].cpu()
                                trans = trans_tensor[0, person_idx].cpu()
                                
                                # Check if parameters are valid (not all zeros)
                                if not (torch.allclose(betas, torch.zeros_like(betas)) and 
                                       torch.allclose(pose, torch.zeros_like(pose))):
                                    all_people_params.append((betas, pose, trans, person_idx))
                                    
                        elif betas_tensor.ndim == 2 and betas_tensor.shape[0] > 0:  # [person, dim]
                            num_people = betas_tensor.shape[0]
                            for person_idx in range(num_people):
                                betas = betas_tensor[person_idx].cpu()
                                pose = pose_tensor[person_idx].cpu()
                                trans = trans_tensor[person_idx].cpu()
                                
                                # Check if parameters are valid (not all zeros)
                                if not (torch.allclose(betas, torch.zeros_like(betas)) and 
                                       torch.allclose(pose, torch.zeros_like(pose))):
                                    all_people_params.append((betas, pose, trans, person_idx))
                    
                    # Process all detected people for mesh generation
                    current_vertices_data = []
                    
                    for person_idx, (betas, pose, trans, original_person_idx) in enumerate(all_people_params):
                        # Generate mesh vertices using SMPLKinematics
                        with torch.no_grad():
                            try:
                                # Ensure parameters are on the same device as the model
                                device = smpl_kinematics_model.joint_template.device
                                betas_tensor = betas.unsqueeze(0).to(device)
                                pose_tensor = pose.unsqueeze(0).to(device)
                                trans_tensor = trans.unsqueeze(0).to(device)
                                
                                vertices = smpl_kinematics_model.forward(
                                    betas_tensor,
                                    pose_tensor,
                                    trans_tensor,
                                    output_format="mesh"
                                )
                                
                                # Move result back to CPU for numpy conversion
                                vertices = vertices.squeeze(0).cpu().numpy()
                                
                                if vertices is not None and vertices.shape[0] > 0:
                                    vertices_processed = vertices.copy()
                                    
                                    # Apply same logic as keypoints
                                    # Get corresponding keypoints for this person
                                    person_keypoints = None
                                    if person_idx < len(all_points_3d_data):
                                        person_keypoints = all_points_3d_data[person_idx]
                                    
                                    # Case 1: 키 보정 ON, 골반 중심화 OFF - 실제 3D 위치 유지
                                    if height_correction_enabled and not pelvis_centering_enabled:
                                        pass
                                    
                                    # Case 2: 키 보정 OFF, 골반 중심화 ON - 전통적 시각화
                                    elif not height_correction_enabled and pelvis_centering_enabled:
                                        if person_keypoints is not None and person_keypoints.shape[0] > 0:
                                            pelvis_position = person_keypoints[0:1, :].copy()
                                            vertices_processed = vertices - pelvis_position
                                    
                                    # Case 3: 둘 다 ON - 키 보정 우선
                                    elif height_correction_enabled and pelvis_centering_enabled:
                                        pass
                                    
                                    # Case 4: 둘 다 OFF - 원시 데이터 (위치 처리만 안함)
                                    elif not height_correction_enabled and not pelvis_centering_enabled:
                                        pass
                                    
                                    # Apply rotation (시각화를 위해 항상 적용)
                                    angle = -np.pi / 2
                                    rotation_matrix_x = np.array([
                                        [1, 0, 0],
                                        [0, np.cos(angle), -np.sin(angle)],
                                        [0, np.sin(angle), np.cos(angle)]
                                    ])
                                    vertices_rotated = np.dot(vertices_processed, rotation_matrix_x.T)
                                    
                                    # Store vertex data for this person
                                    current_vertices_data.append(vertices_rotated)
                                    
                                    # Update mesh visual for this person
                                    if person_idx < len(smpl_mesh_visuals):
                                        person_color = person_colors[person_idx % len(person_colors)]
                                        smpl_mesh_visuals[person_idx].set_data(
                                            vertices=vertices_rotated,
                                            faces=smpl_faces,
                                            color=person_color + (0.9,)  # Person-specific color with less transparency
                                        )
                                
                            except Exception as e:
                                logging.warning(f"Failed to generate mesh for person {person_idx}: {e}")
                                continue
                    
                    # Hide unused mesh visuals
                    for i in range(len(current_vertices_data), len(smpl_mesh_visuals)):
                        smpl_mesh_visuals[i].visible = False
                        smpl_vertices_visuals[i].visible = False
                    
                    # Update view based on current mode
                    update_view_mode()
                else:
                    current_vertices_data = []  # Clear vertex data if no detections
                    # Hide all mesh visuals
                    for mesh_vis in smpl_mesh_visuals:
                        mesh_vis.visible = False
                    for vertices_vis in smpl_vertices_visuals:
                        vertices_vis.visible = False
                    update_view_mode()
            else:
                keypoints_3d_vis.set_data(np.array([[0,0,0]]), edge_color=None, face_color=(1, 1, 1, 0.5), size=0.01)
                current_vertices_data = []  # Clear vertex data when no data available
                # Hide all mesh visuals
                for mesh_vis in smpl_mesh_visuals:
                    mesh_vis.visible = False
                for vertices_vis in smpl_vertices_visuals:
                    vertices_vis.visible = False
                update_view_mode()
            
            canvas.update()
            canvas.app.process_events() # Process Vispy events
            
            # Capture 3D visualization frame (720p)
            try:
                # Render and capture the 3D scene at full canvas resolution
                canvas.render()
                img_3d = canvas.render(alpha=False)  # Render without alpha for better quality
                if img_3d is not None:
                    # Ensure we have the right format and size
                    if img_3d.shape[2] == 4:  # RGBA
                        img_3d_bgr = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
                    elif img_3d.shape[2] == 3:  # RGB
                        img_3d_bgr = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
                    else:
                        img_3d_bgr = img_3d
                    
                    # Ensure it's 720p
                    if img_3d_bgr.shape[:2] != (720, 1280):
                        img_3d_bgr = cv2.resize(img_3d_bgr, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
                    
                    session_frames_3d.append(img_3d_bgr)
                else:
                    # If render fails, store a black 720p frame
                    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    session_frames_3d.append(black_frame)
            except Exception as e:
                logging.warning(f"Failed to capture 3D frame: {e}")
                black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                session_frames_3d.append(black_frame)
            
            frame_count += 1
            fps_calc_frame_count += 1
            
            # Calculate actual FPS every 30 frames
            current_time = time.time()
            if fps_calc_frame_count >= 30:
                elapsed_time = current_time - last_fps_calc_time
                if elapsed_time > 0:
                    current_fps = fps_calc_frame_count / elapsed_time
                    print(f"Current processing FPS: {current_fps:.2f}")
                last_fps_calc_time = current_time
                fps_calc_frame_count = 0

            # Check for exit from multiple sources
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27 or should_exit:  # q, ESC, or 3D window exit
                print("Exiting...")
                break

        # Save session data
        logging.info("Saving session data...")
        
        # Calculate final session FPS
        session_end_time = time.time()
        total_session_time = session_end_time - session_start_time
        if total_session_time > 0 and frame_count > 0:
            final_fps = frame_count / total_session_time
            logging.info(f"Session statistics: {frame_count} frames in {total_session_time:.2f}s = {final_fps:.2f} FPS")
        else:
            final_fps = 30.0  # Fallback
            logging.warning("Could not calculate FPS, using 30 FPS as fallback")
        
        # Save detections
        if session_detections:
            detections_path = output_dir / "webcam_session_detections.pt"
            torch.save(session_detections, detections_path)
            logging.info(f"Saved {len(session_detections)} detection frames to {detections_path}")
        
        # Save 2D visualization video (webcam resolution)
        if session_frames_2d:
            video_2d_path = output_dir / "webcam_session_2d.mp4"
            height, width, _ = session_frames_2d[0].shape
            
            # Try multiple codecs for Windows compatibility
            codecs_to_try = [
                ('mp4v', 'MP4V'),  # MPEG-4 Part 2 - widely supported
                ('MJPG', 'MJPG'),  # Motion JPEG - very compatible
                ('XVID', 'XVID'),  # Xvid MPEG-4 - good quality
                ('H264', 'H264'),  # H.264 - best quality but may have issues
            ]
            
            out_2d = None
            used_codec = None
            
            for codec_fourcc, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                    out_2d = cv2.VideoWriter(str(video_2d_path), fourcc, final_fps, (width, height))
                    
                    # Test if the writer was successfully initialized
                    if out_2d.isOpened():
                        used_codec = codec_name
                        logging.info(f"Using {codec_name} codec for 2D video")
                        break
                    else:
                        out_2d.release()
                        out_2d = None
                except Exception as e:
                    logging.warning(f"Failed to initialize {codec_name} codec: {e}")
                    if out_2d:
                        out_2d.release()
                        out_2d = None
            
            if out_2d and out_2d.isOpened():
                for frame in session_frames_2d:
                    out_2d.write(frame)
                out_2d.release()
                logging.info(f"Saved 2D visualization video ({width}x{height}) using {used_codec} to {video_2d_path}")
            else:
                logging.error("Failed to initialize any video codec for 2D video")
        
        # Save 3D visualization video (720p)
        if session_frames_3d:
            video_3d_path = output_dir / "webcam_session_3d.mp4"
            # Force 720p resolution for 3D video
            target_width, target_height = 1280, 720
            
            # Try multiple codecs for Windows compatibility
            codecs_to_try = [
                ('mp4v', 'MP4V'),  # MPEG-4 Part 2 - widely supported
                ('MJPG', 'MJPG'),  # Motion JPEG - very compatible
                ('XVID', 'XVID'),  # Xvid MPEG-4 - good quality
                ('H264', 'H264'),  # H.264 - best quality but may have issues
            ]
            
            out_3d = None
            used_codec = None
            
            for codec_fourcc, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                    out_3d = cv2.VideoWriter(str(video_3d_path), fourcc, final_fps, (target_width, target_height))
                    
                    # Test if the writer was successfully initialized
                    if out_3d.isOpened():
                        used_codec = codec_name
                        logging.info(f"Using {codec_name} codec for 3D video")
                        break
                    else:
                        out_3d.release()
                        out_3d = None
                except Exception as e:
                    logging.warning(f"Failed to initialize {codec_name} codec: {e}")
                    if out_3d:
                        out_3d.release()
                        out_3d = None
            
            if out_3d and out_3d.isOpened():
                for frame in session_frames_3d:
                    # Resize frame to 720p if necessary
                    if frame.shape[:2] != (target_height, target_width):
                        frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                        out_3d.write(frame_resized)
                    else:
                        out_3d.write(frame)
                out_3d.release()
                logging.info(f"Saved 3D visualization video (1280x720) using {used_codec} to {video_3d_path}")
            else:
                logging.error("Failed to initialize any video codec for 3D video")
        
        # Save session summary
        summary_path = output_dir / "webcam_session_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"WebCam Session Summary\n")
            f.write(f"======================\n")
            f.write(f"Total frames processed: {frame_count}\n")
            f.write(f"Session duration: {total_session_time:.2f}s\n")
            f.write(f"Average FPS: {final_fps:.2f}\n")
            f.write(f"Frames with detections: {len(session_detections)}\n")
            f.write(f"User height setting: {user_height_m:.2f}m\n")
            f.write(f"Height correction: {'ON' if height_correction_enabled else 'OFF'}\n")
            f.write(f"Pelvis centering: {'ON' if pelvis_centering_enabled else 'OFF'}\n")
            f.write(f"Grid visible: {'ON' if grid_visible else 'OFF'}\n")
            f.write(f"Axes visible: {'ON' if axes_visible else 'OFF'}\n")
            f.write(f"\nOutput files:\n")
            f.write(f"- Detections: webcam_session_detections.pt\n")
            f.write(f"- 2D visualization: webcam_session_2d.mp4 ({final_fps:.2f} FPS)\n")
            f.write(f"- 3D visualization: webcam_session_3d.mp4 ({final_fps:.2f} FPS)\n")
        
        logging.info(f"Session summary saved to {summary_path}")
        logging.info("Session recording completed!")

        cap.release()
        cv2.destroyAllWindows()
        canvas.close() # Close Vispy canvas when exiting

    elif input_path and input_path.suffix.lower() in dataloading.IMAGE_EXTENSIONS:
        # Run and visualize detections for a single image
        run_detection(input_path, cache_path, skip_visualization, user_height_m=user_height)
    else:
        # Run unrolled tracking on a full video
        track_poses(input_path, cache_path, start_frame, num_frames, frameskip)
        if not skip_visualization:
            video_path = output_dir / f"{input_name}.mp4"
            visualize_poses(
                input_path, cache_path, video_path, start_frame, num_frames, frameskip
            )


if __name__ == "__main__":
    main()
