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


def save_video_frames(frames_list, output_path, fps, frame_size, logging_prefix="Video"):
    """Helper function to save a list of frames as a video."""
    if not frames_list:
        logging.warning(f"{logging_prefix}: No frames to save for {output_path}.")
        return

    expected_height, expected_width = frame_size
    
    # cv2.VideoWriter expects (width, height)
    output_width, output_height = expected_width, expected_height 

    # Try multiple codecs for Windows compatibility
    codecs_to_try = [
        ('mp4v', 'MP4V'),  # MPEG-4 Part 2 - widely supported
        ('MJPG', 'MJPG'),  # Motion JPEG - very compatible
        ('XVID', 'XVID'),  # Xvid MPEG-4 - good quality
        ('H264', 'H264'),  # H.264 - best quality but may have issues
    ]
    
    out_video = None
    used_codec = None
    
    for codec_fourcc, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            # Use output_width, output_height for VideoWriter
            out_video = cv2.VideoWriter(str(output_path), fourcc, float(fps), (output_width, output_height))
            
            if out_video.isOpened():
                used_codec = codec_name
                logging.info(f"{logging_prefix}: Using {codec_name} codec for video {output_path} ({output_width}x{output_height} @ {fps:.2f} FPS)")
                break
            else:
                # Ensure release even if not opened, to be safe
                if out_video: out_video.release()
                out_video = None
        except Exception as e:
            logging.warning(f"{logging_prefix}: Failed to initialize {codec_name} codec for {output_path}: {e}")
            if out_video:
                out_video.release()
                out_video = None
    
    if out_video and out_video.isOpened():
        for frame_idx, frame in enumerate(frames_list):
            # Ensure frame matches output dimensions
            if frame.shape[0] != output_height or frame.shape[1] != output_width:
                # logging.debug(f"{logging_prefix}: Resizing frame {frame_idx} from {frame.shape[1]}x{frame.shape[0]} to {output_width}x{output_height} for {output_path}")
                frame_resized = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                out_video.write(frame_resized)
            else:
                out_video.write(frame)
        out_video.release()
        logging.info(f"{logging_prefix}: Saved video ({output_width}x{output_height}) using {used_codec} to {output_path}")
    else:
        logging.error(f"{logging_prefix}: Failed to initialize any video codec for {output_path}. Video not saved.")


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
        if not preds or not preds.get("id", None): # Handle empty predictions from cache
            logging.warning(f"No valid predictions found in cache {cache_path} or 'id' key missing. Skipping pose addition.")
        else:
            track_subset = track_utils.query_range(preds, 0, frame_idx - 1)
            if not track_subset or not track_subset.get("id", None): # Check if query_range returns empty or invalid
                 logging.warning(f"Querying range in predictions for {cache_path} did not yield valid tracks. Skipping pose addition.")
            else:
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
    input_path, cache_path, start_frame, num_frames, frameskip=1, model=None,
    output_2d_frames_list=None, user_height_m=None
):
    """Track poses over a video or a directory of images."""
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    smpl_kinematics_model = None
    previous_scale_factor = None
    if user_height_m is not None: # Initialize if user_height_m is provided
        smpl_kinematics_model = smpl_kinematics.SMPLKinematics()
        if torch.cuda.is_available(): # Use CUDA if available for SMPL model
            smpl_kinematics_model = smpl_kinematics_model.cuda()
        else: # Fallback to CPU for SMPL model
            smpl_kinematics_model = smpl_kinematics_model.cpu()
        smpl_kinematics_model.eval()


    detections_history = []
    tracks_history = []
    image_res_for_saving = None # Store image resolution for MOT format saving

    initialized = False
    processed_frame_count = 0
    last_K_for_saving = None # Store K for MOT format saving


    for image_tensor, K in tqdm(
        dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip),
        desc="Running CoMotion & Preparing 2D Video" if output_2d_frames_list is not None else "Running CoMotion",
    ):
        if not initialized:
            image_res = image_tensor.shape[-2:]
            image_res_for_saving = image_res # Store for later use
            model.init_tracks(image_res)
            initialized = True
        
        last_K_for_saving = K.cpu() # Store for later, ensure it's on CPU

        detection, track = model(image_tensor, K, use_mps=use_mps)

        # Apply height-based depth correction if enabled
        if user_height_m is not None and smpl_kinematics_model is not None and detection:
            required_keys = ["betas", "pose", "trans"]
            if all(key in detection and detection[key] is not None for key in required_keys):
                # Pass detection directly, it contains tensors on the 'device'
                detection_corrected, scale_factor = apply_height_based_depth_correction(
                    detection, user_height_m, smpl_kinematics_model,
                    previous_scale_factor, smooth_factor=0.8
                )
                if scale_factor != 1.0: 
                    detection = detection_corrected 
                    previous_scale_factor = scale_factor
            # else:
                # logging.debug("Skipping height correction due to missing keys in detection.")
        
        detection_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in detection.items()}
        track_cpu = track.cpu() if isinstance(track, torch.Tensor) else track
        
        detections_history.append(detection_cpu)
        tracks_history.append(track_cpu)

        if output_2d_frames_list is not None:
            processed_frame = dataloading.convert_tensor_to_image(image_tensor) # HWC, uint8
            
            if detection_cpu and "pred_2d" in detection_cpu and detection_cpu["pred_2d"] is not None:
                pred_2d_tensor = detection_cpu["pred_2d"]
                # pred_2d_tensor should be on CPU already from detection_cpu
                # Expected shape from model: (1, num_people, num_keypoints, 2)
                
                keypoints_2d_batch = None
                if pred_2d_tensor.ndim == 4 and pred_2d_tensor.shape[0] == 1: # Batch size 1
                    keypoints_2d_batch = pred_2d_tensor[0].numpy() # (num_people, num_keypoints, 2)
                elif pred_2d_tensor.ndim == 3: # Potentially (num_people, num_keypoints, 2)
                    keypoints_2d_batch = pred_2d_tensor.numpy()
                
                if keypoints_2d_batch is not None:
                    for person_keypoints in keypoints_2d_batch: # Iterate over people
                        for kp_idx in range(person_keypoints.shape[0]): # Iterate over keypoints
                            kp = person_keypoints[kp_idx]
                            if kp.shape[0] == 2: 
                                x, y = int(kp[0]), int(kp[1])
                                if 0 <= x < processed_frame.shape[1] and 0 <= y < processed_frame.shape[0]:
                                    cv2.circle(processed_frame, (x, y), 3, (0, 255, 0), -1)
            
            output_2d_frames_list.append(processed_frame.copy())
        processed_frame_count +=1

    if not detections_history:
        logging.warning("No detections were made during tracking. Saving empty cache.")
        torch.save({}, cache_path)
        return processed_frame_count # Return count even if no detections

    detections_processed = {k: [d[k] for d in detections_history if k in d] for k in detections_history[0].keys()}
    tracks_stacked = torch.stack(tracks_history, 1)
    tracks_processed = {k: getattr(tracks_stacked, k) for k in ["id", "pose", "trans", "betas"]}

    # Ensure K for cleanup is on CPU
    k_for_cleanup = last_K_for_saving if last_K_for_saving is not None else dataloading.get_default_K(image_tensor).cpu()


    track_ref = track_utils.cleanup_tracks(
        {"detections": detections_processed, "tracks": tracks_processed},
        k_for_cleanup, 
        model.smpl_decoder.cpu(),
        min_matched_frames=1,
    )
    if track_ref and track_ref.get("id") is not None and len(track_ref["id"]) > 0 :
        frame_idxs, track_idxs = track_utils.convert_to_idxs(
            track_ref, tracks_processed["id"][0].squeeze(-1).long()
        )
        preds = {k: v[0, frame_idxs, track_idxs] for k, v in tracks_processed.items()}
        preds["id"] = preds["id"].squeeze(-1).long()
        preds["frame_idx"] = frame_idxs
        torch.save(preds, cache_path)

        if image_res_for_saving:
            # Ensure K for bboxes_from_smpl is on CPU
            bboxes_k = k_for_cleanup
            bboxes = track_utils.bboxes_from_smpl(
                model.smpl_decoder.cpu(), # Ensure decoder is on CPU
                {k: preds[k].cpu() if isinstance(preds[k], torch.Tensor) else preds[k] for k in ["betas", "pose", "trans"]}, # Ensure params are on CPU
                image_res_for_saving,
                bboxes_k,
            )
            with open(str(cache_path).replace(".pt", ".txt"), "w") as f:
                f.write(track_utils.convert_to_mot(preds["id"], preds["frame_idx"], bboxes))
    else:
        logging.warning("Track reference is empty or invalid after cleanup. Saving empty cache.")
        torch.save({}, cache_path)
    
    return processed_frame_count

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
    default=1_000_000_000, # A large number to signify processing all frames by default
    type=int,
    help="Number of frames to process.",
)
@click.option(
    "--skip-visualization",
    is_flag=True,
    help="Whether to skip rendering output visualizations (AITViewer for sequences, Vispy for webcam/video).",
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
    
    # Note: skip_visualization now affects both AITViewer and the Vispy-based live visualization
    # If aitviewer_available is False, Vispy might still run if not skipped.
    # The prompt implies that for video, it should work like webcam, so Vispy would be prioritized if vis is not skipped.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # global use_mps is defined at the script level

    is_video_input = input_path and dataloading.is_a_video(input_path)

    # Determine if we should enter the interactive (Vispy-based) visualization block
    # This block will be used for webcam, or for video if visualization is requested.
    run_interactive_vispy_mode = False
    if webcam:
        run_interactive_vispy_mode = True
    elif is_video_input and not skip_visualization:
        run_interactive_vispy_mode = True

    if run_interactive_vispy_mode:
        if webcam:
            logging.info("Using webcam for real-time Vispy processing and visualization.")
        else: # Must be is_video_input and not skip_visualization
            logging.info(f"Using video '{input_path.name}' for real-time Vispy processing and visualization.")

        # Initialize frame source (webcam or video)
        frame_source_fn = None
        frame_source_resource = None # To hold 'cap' or 'iterator' for cleanup
        actual_image_width, actual_image_height = 1280, 720 # Default, will be updated
        output_target_fps = 30 # Default FPS for saving videos

        # --- Model Initialization (common for webcam/video in this block) ---
        # This model instance is local to this interactive block
        interactive_model = comotion.CoMotion(use_coreml=use_mps)
        interactive_model.to(device).eval()
        
        # --- SMPL Kinematics Model (common for webcam/video in this block) ---
        smpl_kinematics_instance = smpl_kinematics.SMPLKinematics()
        if torch.cuda.is_available(): # Use global device
            smpl_kinematics_instance = smpl_kinematics_instance.to(device)
        smpl_kinematics_instance.eval()

        if webcam:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Cannot open webcam.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            actual_image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Webcam resolution: {actual_image_width}x{actual_image_height}")
            frame_source_resource = cap
            output_target_fps = 30 # Typical for webcam

            def _get_webcam_frame():
                nonlocal frame_source_resource # cap
                ret, frame = frame_source_resource.read()
                if not ret: return None, None
                # For webcam, K is often a default based on resolution
                k_for_frame = dataloading.get_default_K_for_image_res((actual_image_height, actual_image_width), device="cpu")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), k_for_frame
            frame_source_fn = _get_webcam_frame
        
        elif is_video_input: # Already checked: is_video_input and not skip_visualization
            # Get video properties for initialization
            try:
                # Peek first frame for dimensions (K comes with each frame from iterator)
                temp_iter = dataloading.yield_image_and_K(input_path, start_frame, 1, frameskip)
                first_img_tensor, _ = next(temp_iter) # K will be handled per frame
                actual_image_height, actual_image_width = first_img_tensor.shape[-2:]
                del temp_iter, first_img_tensor 
            except StopIteration:
                logging.error(f"Video {input_path} is empty or cannot be read at start_frame {start_frame}.")
                return
            
            video_frame_iterator = dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip)
            frame_source_resource = video_frame_iterator # For conceptual similarity, though iterators don't need explicit release
            output_target_fps = int(dataloading.get_input_video_fps(input_path))
            logging.info(f"Video input: {input_path.name}, Resolution: {actual_image_width}x{actual_image_height}, Output FPS: {output_target_fps}")

            processed_frames_count = 0 # For respecting num_frames with video iterator
            def _get_video_frame():
                nonlocal frame_source_resource, processed_frames_count, num_frames # video_frame_iterator
                if num_frames != 1_000_000_000 and processed_frames_count >= num_frames:
                    return None, None
                try:
                    image_tensor, k_tensor = next(frame_source_resource) # image_tensor is [1,C,H,W], k_tensor is [1,3,3]

                    # ---- START DEBUG PRINTS ----
                    print(f"\nDEBUG in _get_video_frame:")
                    print(f"  image_tensor type: {type(image_tensor)}")
                    if hasattr(image_tensor, 'shape'):
                        print(f"  image_tensor shape: {image_tensor.shape}")
                    if hasattr(image_tensor, 'dim'):
                        print(f"  image_tensor dim: {image_tensor.dim()}")

                    arg_to_convert = image_tensor[0] # This is the tensor passed to convert_tensor_to_image
                    print(f"  arg_to_convert type: {type(arg_to_convert)}")
                    if hasattr(arg_to_convert, 'shape'):
                        print(f"  arg_to_convert shape: {arg_to_convert.shape}")
                    if hasattr(arg_to_convert, 'dim'):
                        print(f"  arg_to_convert dim: {arg_to_convert.dim()}\n")
                    # ---- END DEBUG PRINTS ----

                    # Potential Fix: If arg_to_convert is 2D [H,W], unsqueeze to [1,H,W] (CHW with C=1)
                    if arg_to_convert.dim() == 2:
                        print(f"DEBUG: arg_to_convert was 2D, unsqueezing to add channel dimension.")
                        arg_to_convert = arg_to_convert.unsqueeze(0) # Now [1,H,W]
                        print(f"DEBUG: arg_to_convert new shape: {arg_to_convert.shape}, new dim: {arg_to_convert.dim()}")


                    image_numpy_rgb = dataloading.convert_tensor_to_image(arg_to_convert) # Pass CHW part
                    processed_frames_count += 1
                    return image_numpy_rgb, k_tensor[0].cpu() # Return HWC RGB numpy array and 3x3 K tensor on CPU
                except StopIteration:
                    return None, None

        # Initialize CoMotion model tracks with the actual image resolution
        interactive_model.init_tracks((actual_image_height, actual_image_width))

        # --- Start of the Vispy visualization and processing loop (adapted from original webcam block) ---
        # Initialize Vispy Canvas (e.g., 720p for recording quality)
        # Match canvas to actual input for better 1:1, or fix for consistent output.
        # Original webcam block used fixed 1280x720 for canvas.
        vis_canvas_width, vis_canvas_height = 1280, 720
        canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(vis_canvas_width, vis_canvas_height))
        canvas.native.activateWindow()
        canvas.native.raise_()
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 12
        view.camera.center = (0, 0, 0.5)
        view.camera.elevation = 20

        keypoints_3d_vis = scene.visuals.Markers(parent=view.scene)
        max_people_to_visualize = 10 # As in original webcam block
        smpl_mesh_visuals = []
        smpl_vertices_visuals = []
        for _ in range(max_people_to_visualize):
            mesh_vis = VispyMesh(parent=view.scene)
            mesh_vis.visible = False
            smpl_mesh_visuals.append(mesh_vis)
            vertices_vis = scene.visuals.Markers(parent=view.scene)
            vertices_vis.visible = False
            smpl_vertices_visuals.append(vertices_vis)

        user_height_m_current = user_height if user_height else 1.75
        height_correction_enabled_current = user_height is not None
        previous_scale_factor_current = None
        pelvis_centering_enabled_current = not height_correction_enabled_current
        
        grid_z_offset_current = (-user_height_m_current / 2) - 0.15
        
        # Grid and Axes (copied and adapted from webcam block, ensure variables like grid_z_offset_current are used)
        grid_size = 15.0; grid_divisions = 38; step = grid_size / grid_divisions
        grid_lines_np = []
        for i in range(grid_divisions + 1): # Horizontal
            y_coord = -grid_size/2 + i * step
            grid_lines_np.extend([[-grid_size/2, y_coord, grid_z_offset_current], [grid_size/2, y_coord, grid_z_offset_current]])
        for i in range(grid_divisions + 1): # Vertical
            x_coord = -grid_size/2 + i * step
            grid_lines_np.extend([[x_coord, -grid_size/2, grid_z_offset_current], [x_coord, grid_size/2, grid_z_offset_current]])
        grid_lines_np = np.array(grid_lines_np, dtype=np.float32)
        grid_visual_obj = scene.visuals.Line(pos=grid_lines_np, color=(0.2, 0.4, 0.6, 0.7), width=1.0, parent=view.scene, connect='segments')

        main_grid_lines_np = [] # Thicker main lines
        for i in range(0, grid_divisions + 1, 5):
            y_coord = -grid_size/2 + i * step; x_coord = -grid_size/2 + i * step
            main_grid_lines_np.extend([[-grid_size/2, y_coord, grid_z_offset_current], [grid_size/2, y_coord, grid_z_offset_current]])
            main_grid_lines_np.extend([[x_coord, -grid_size/2, grid_z_offset_current], [x_coord, grid_size/2, grid_z_offset_current]])
        main_grid_lines_np = np.array(main_grid_lines_np, dtype=np.float32)
        main_grid_visual_obj = scene.visuals.Line(pos=main_grid_lines_np, color=(0.3, 0.5, 0.7, 0.9), width=2.0, parent=view.scene, connect='segments')
        
        axes_lines_np = np.array([[0,0,grid_z_offset_current],[0.4,0,grid_z_offset_current], [0,0,grid_z_offset_current],[0,0.4,grid_z_offset_current], [0,0,grid_z_offset_current],[0,0,grid_z_offset_current+0.4]], dtype=np.float32)
        axes_colors_np = np.array([[1,0,0,0.8],[1,0,0,0.8], [0,0,1,0.8],[0,0,1,0.8], [0,1,0,0.8],[0,1,0,0.8]], dtype=np.float32) # X=R, Y=B(Vispy Y is up), Z=G
        axes_visual_obj = scene.visuals.Line(pos=axes_lines_np, color=axes_colors_np, width=8.0, parent=view.scene, connect='segments')

        current_view_mode_vis = 1  # 0=keypoints, 1=mesh, 2=vertices
        current_vertices_data_vis = None
        grid_visible_vis = True
        axes_visible_vis = True
        
        session_detections_data = []
        session_frames_2d_data = []
        session_frames_3d_data = []
        interactive_frame_count = 0
        
        import time # For FPS calculation
        session_start_time_interactive = time.time()
        last_fps_calc_time_interactive = session_start_time_interactive
        fps_calc_frame_count_interactive = 0
        current_processing_fps = output_target_fps # Initialize with target

        should_exit_interactive_loop = False

        text_vis_obj = scene.visuals.Text(text='', color='white', pos=(20,100), anchor_x='left', anchor_y='top', font_size=10, parent=canvas.scene)

        def update_text_display_interactive():
            nonlocal text_vis_obj, current_processing_fps
            hc_status = "ON" if height_correction_enabled_current else "OFF"
            pc_status = "ON" if pelvis_centering_enabled_current else "OFF"
            grid_stat = "ON" if grid_visible_vis else "OFF"
            axes_stat = "ON" if axes_visible_vis else "OFF"
            mode_str = "Height Correction" if height_correction_enabled_current else ("Pelvis Centered" if pelvis_centering_enabled_current else "Raw 3D")
            
            text_to_display = (f'1=KP 2=Mesh 3=Verts | G:{grid_stat} A:{axes_stat} | FPS: {current_processing_fps:.1f}\n'
                               f'Height:{hc_status} Pelvis:{pc_status} | Mode: {mode_str}\n'
                               f'H/P=toggle +/-=height({user_height_m_current:.2f}m) | Q/ESC=exit')
            text_vis_obj.text = text_to_display
            canvas.update()


        def update_grid_position_interactive():
            nonlocal grid_z_offset_current, grid_visual_obj, main_grid_visual_obj, axes_visual_obj, user_height_m_current
            grid_z_offset_current = (-user_height_m_current / 2) - 0.15 # Recalculate Z offset
            
            # Update grid lines (positions are N,3 arrays)
            new_grid_lines = []
            for i in range(grid_divisions + 1):
                y = -grid_size/2 + i * step
                new_grid_lines.extend([[-grid_size/2, y, grid_z_offset_current], [grid_size/2, y, grid_z_offset_current]])
            for i in range(grid_divisions + 1):
                x = -grid_size/2 + i * step
                new_grid_lines.extend([[x, -grid_size/2, grid_z_offset_current], [x, grid_size/2, grid_z_offset_current]])
            grid_visual_obj.set_data(pos=np.array(new_grid_lines, dtype=np.float32))

            new_main_grid_lines = []
            for i in range(0, grid_divisions + 1, 5):
                y = -grid_size/2 + i * step; x = -grid_size/2 + i * step
                new_main_grid_lines.extend([[-grid_size/2, y, grid_z_offset_current], [grid_size/2, y, grid_z_offset_current]])
                new_main_grid_lines.extend([[x, -grid_size/2, grid_z_offset_current], [x, grid_size/2, grid_z_offset_current]])
            main_grid_visual_obj.set_data(pos=np.array(new_main_grid_lines, dtype=np.float32))
            
            # Update axes lines
            new_axes_lines = np.array([
                [0,0,grid_z_offset_current],[0.4,0,grid_z_offset_current], 
                [0,0,grid_z_offset_current],[0,0.4,grid_z_offset_current], 
                [0,0,grid_z_offset_current],[0,0,grid_z_offset_current+0.4]], dtype=np.float32)
            axes_visual_obj.set_data(pos=new_axes_lines)
            canvas.update()

        update_text_display_interactive() # Initial text

        @canvas.events.key_press.connect
        def on_key_press_interactive(event):
            nonlocal current_view_mode_vis, height_correction_enabled_current, pelvis_centering_enabled_current, user_height_m_current
            nonlocal grid_visible_vis, axes_visible_vis, should_exit_interactive_loop
            if event.key == '1' or event.key == 'Space': current_view_mode_vis = 0
            elif event.key == '2': current_view_mode_vis = 1
            elif event.key == '3': current_view_mode_vis = 2
            elif event.key == 'h':
                height_correction_enabled_current = not height_correction_enabled_current
                if height_correction_enabled_current: pelvis_centering_enabled_current = False
            elif event.key == 'p':
                pelvis_centering_enabled_current = not pelvis_centering_enabled_current
            elif event.key == '=' or event.key == '+':
                user_height_m_current += 0.05
                update_grid_position_interactive()
            elif event.key == '-':
                user_height_m_current = max(1.0, user_height_m_current - 0.05)
                update_grid_position_interactive()
            elif event.key == 'g':
                grid_visible_vis = not grid_visible_vis
                grid_visual_obj.visible = grid_visible_vis
                main_grid_visual_obj.visible = grid_visible_vis
            elif event.key == 'a':
                axes_visible_vis = not axes_visible_vis
                axes_visual_obj.visible = axes_visible_vis
            elif event.key == 'q' or event.key == 'Escape':
                should_exit_interactive_loop = True
            update_text_display_interactive()


        def update_view_mode_interactive():
            nonlocal keypoints_3d_vis, smpl_mesh_visuals, smpl_vertices_visuals, current_view_mode_vis, current_vertices_data_vis
            num_people_data = len(current_vertices_data_vis) if current_vertices_data_vis else 0
            keypoints_3d_vis.visible = (current_view_mode_vis == 0 and num_people_data > 0)
            for i in range(max_people_to_visualize):
                has_data_for_person_i = current_vertices_data_vis and i < len(current_vertices_data_vis)
                smpl_mesh_visuals[i].visible = (current_view_mode_vis == 1 and has_data_for_person_i)
                smpl_vertices_visuals[i].visible = (current_view_mode_vis == 2 and has_data_for_person_i)


        # Main Loop for interactive (Vispy) processing
        logging.info("Starting Vispy visualization loop...")
        while not should_exit_interactive_loop:
            frame_np_rgb, k_for_frame_cpu = frame_source_fn() # k_for_frame_cpu is 3x3 tensor

            if frame_np_rgb is None: # End of frames or error from source
                logging.info("Frame source finished or encountered an error.")
                break
            
            # FPS calculation
            fps_calc_frame_count_interactive += 1
            current_time_interactive = time.time()
            if current_time_interactive - last_fps_calc_time_interactive >= 1.0:
                current_processing_fps = fps_calc_frame_count_interactive / (current_time_interactive - last_fps_calc_time_interactive)
                fps_calc_frame_count_interactive = 0
                last_fps_calc_time_interactive = current_time_interactive
                update_text_display_interactive() # Update FPS on screen

            # Prepare model inputs
            # convert_image_to_tensor expects HWC numpy array -> returns BCHW tensor
            image_tensor_bchw = dataloading.convert_image_to_tensor(frame_np_rgb).to(device)
            k_for_model_dev = k_for_frame_cpu.to(device).unsqueeze(0) # Add batch dim: [1, 3, 3]

            # Model inference
            with torch.no_grad():
                detection_results, track_results = interactive_model(image_tensor_bchw, k_for_model_dev, use_mps=use_mps)

            detection_cpu = {k: v.cpu() for k, v in detection_results.items()}
            track_cpu = track_results.cpu() # track_results is a Tracks object
            session_detections_data.append(detection_cpu) # Store raw detections for potential later use

            # Prepare SMPL parameters for visualization and correction
            # track_cpu attributes (like .pose, .trans, .betas) are [1, 1, num_people, dim]
            # We need [num_people, dim] for processing
            params_for_vis = {}
            num_detected_people_in_frame = 0
            if track_cpu.id.numel() > 0 and track_cpu.id.shape[2] > 0: # id is [1,1,num_people,1]
                num_detected_people_in_frame = track_cpu.id.shape[2]
                params_for_vis["betas"] = track_cpu.betas[0, 0]  # Shape: [num_people, 10]
                params_for_vis["pose"] = track_cpu.pose[0, 0]    # Shape: [num_people, 72]
                params_for_vis["trans"] = track_cpu.trans[0, 0]  # Shape: [num_people, 3]
            
            corrected_params_for_vis = params_for_vis.copy()

            if height_correction_enabled_current and num_detected_people_in_frame > 0:
                # apply_height_based_depth_correction expects a dict with 'betas', 'pose', 'trans'
                # It internally handles the first person [0] from the [num_people, dim] tensors.
                # To correct all, one might loop, but current func corrects based on first person and applies scale.
                # The function modifies 'trans' in the passed dict.
                temp_dict_for_correction = {"betas": params_for_vis["betas"], "pose": params_for_vis["pose"], "trans": params_for_vis["trans"].clone()} # Operate on a clone for trans
                
                corrected_dict_result, current_scale_factor_val = apply_height_based_depth_correction(
                    temp_dict_for_correction, # Pass the dict containing all people's params
                    user_height_m_current,
                    smpl_kinematics_instance, # Use the model created in this block
                    previous_scale_factor_current,
                    smooth_factor=0.8 
                )
                # Update trans in corrected_params_for_vis if correction was successful
                if "trans" in corrected_dict_result and corrected_dict_result["trans"].shape == params_for_vis["trans"].shape:
                     corrected_params_for_vis["trans"] = corrected_dict_result["trans"]
                previous_scale_factor_current = current_scale_factor_val
            
            # Pelvis centering (if enabled and height correction is not, applied after potential height correction)
            pelvis_offset_val = torch.zeros(3, device=corrected_params_for_vis.get("trans", torch.zeros(1,3)).device) # Match device
            if pelvis_centering_enabled_current and not height_correction_enabled_current and num_detected_people_in_frame > 0:
                pelvis_offset_val = corrected_params_for_vis["trans"][0].clone() # Center on the first detected person's pelvis

            # Generate 3D mesh/keypoints data for Vispy
            all_vertices_for_frame = []
            all_joints_for_frame = []
            if num_detected_people_in_frame > 0:
                for i in range(num_detected_people_in_frame):
                    beta_p = corrected_params_for_vis["betas"][i].unsqueeze(0).to(smpl_kinematics_instance.joint_template.device)
                    pose_p = corrected_params_for_vis["pose"][i].unsqueeze(0).to(smpl_kinematics_instance.joint_template.device)
                    # Apply pelvis centering to translation
                    trans_p = (corrected_params_for_vis["trans"][i] - pelvis_offset_val).unsqueeze(0).to(smpl_kinematics_instance.joint_template.device)
                    
                    with torch.no_grad():
                        smpl_out = smpl_kinematics_instance.forward(beta_p, pose_p, trans=trans_p, output_format="vertices_joints")
                        all_vertices_for_frame.append(smpl_out["vertices"].cpu().numpy().squeeze(0)) # Squeeze batch dim
                        all_joints_for_frame.append(smpl_out["joints"].cpu().numpy().squeeze(0))   # Squeeze batch dim
            
            current_vertices_data_vis = all_vertices_for_frame # Used by update_view_mode_interactive

            # Update Vispy 3D scene
            update_view_mode_interactive() # Sets visibility based on mode

            if current_view_mode_vis == 0 and all_joints_for_frame: # Keypoints
                combined_joints_np = np.concatenate(all_joints_for_frame, axis=0) if all_joints_for_frame else np.empty((0,3))
                keypoints_3d_vis.set_data(combined_joints_np, edge_color=None, face_color=(1,0,0,0.7), size=8)
            
            for i in range(max_people_to_visualize): # Update active visuals, hide unused
                if i < len(all_vertices_for_frame):
                    person_vertices = all_vertices_for_frame[i]
                    # smpl_faces is a global variable from the script's top
                    if current_view_mode_vis == 1: # Mesh
                        smpl_mesh_visuals[i].set_data(vertices=person_vertices, faces=smpl_faces, color=(0.7,0.7,0.9,0.8))
                    elif current_view_mode_vis == 2: # Vertices
                        smpl_vertices_visuals[i].set_data(person_vertices, face_color=(0.5,0.9,0.5,0.6), size=3)
                else: # No data for this visual index, ensure it's hidden
                    smpl_mesh_visuals[i].visible = False
                    smpl_vertices_visuals[i].visible = False
            
            # 2D Visualization (display input frame with simple overlay)
            frame_for_2d_display_rgb = frame_np_rgb.copy()
            # Example: Draw bounding boxes or keypoints if available from 'detection_cpu'
            # For simplicity, just text overlay for now.
            cv2.putText(frame_for_2d_display_rgb, f"Frame: {interactive_frame_count}", (15, int(actual_image_height - 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_bgr_for_saving = cv2.cvtColor(frame_for_2d_display_rgb, cv2.COLOR_RGB2BGR)
            session_frames_2d_data.append(frame_bgr_for_saving)
            
            # Show live 2D (optional, can be intensive)
            cv2.imshow("2D Visualization (Input)", frame_bgr_for_saving)

            # Capture 3D Vispy scene for saving
            rendered_3d_frame_rgba = canvas.render(alpha=True) # RGBA
            session_frames_3d_data.append(rendered_3d_frame_rgba)

            canvas.app.process_events() # Vital for Vispy window responsiveness and updates
            if cv2.waitKey(1) & 0xFF == ord('q'): # Allow exit via 'q' in OpenCV window
                should_exit_interactive_loop = True
            
            interactive_frame_count += 1
        
        # --- End of interactive processing loop ---
        logging.info(f"Vispy visualization loop finished. Processed {interactive_frame_count} frames.")

        # Cleanup resources
        if webcam and frame_source_resource: # frame_source_resource is 'cap' for webcam
            frame_source_resource.release()
        cv2.destroyAllWindows() # Close OpenCV windows
        canvas.close() # Close Vispy canvas

        # Save recorded 2D and 3D visualization videos
        if session_frames_2d_data:
            out_2d_video_path = output_dir / f"{input_name}_2D_vispy_style.mp4"
            logging.info(f"Saving 2D Vispy-style visualization to {out_2d_video_path}")
            # Get dimensions from first frame (they should all be same)
            h_2d, w_2d = session_frames_2d_data[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_2d = cv2.VideoWriter(str(out_2d_video_path), fourcc, output_target_fps, (w_2d, h_2d))
            for frame_2d_bgr in tqdm(session_frames_2d_data, desc="Saving 2D Video"):
                video_writer_2d.write(frame_2d_bgr)
            video_writer_2d.release()

        if session_frames_3d_data:
            out_3d_video_path = output_dir / f"{input_name}_3D_vispy_style.mp4"
            logging.info(f"Saving 3D Vispy-style visualization to {out_3d_video_path}")
            h_3d, w_3d = session_frames_3d_data[0].shape[:2] # Vispy render gives H,W,C (RGBA)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_3d = cv2.VideoWriter(str(out_3d_video_path), fourcc, output_target_fps, (w_3d, h_3d))
            for frame_3d_rgba in tqdm(session_frames_3d_data, desc="Saving 3D Video"):
                frame_3d_bgr = cv2.cvtColor(frame_3d_rgba, cv2.COLOR_RGBA2BGR) # Convert to BGR for VideoWriter
                video_writer_3d.write(frame_3d_bgr)
            video_writer_3d.release()
        
        # Optionally save detections (session_detections_data) if needed, similar to track_poses
        # For now, focusing on visualization saving as per request.

    else: # Not webcam and ( (not video) or (video and skip_visualization is True) )
          # This handles: single image, image directory, or video with visualization explicitly skipped
        
        # Model for non-interactive processing (can be distinct or shared if managed carefully)
        # For simplicity here, assume track_poses/run_detection will manage their model instances if not passed
        # or we can create one here.
        processing_model = comotion.CoMotion(use_coreml=use_mps)
        processing_model.to(device).eval()

        cache_path = output_dir / f"{input_name}.pt" # Used by track_poses and run_detection

        if input_path:
            if is_video_input: # Video input, but skip_visualization was True
                logging.info(f"Tracking poses for video {input_path.name} (visualization skipped).")
                track_poses(input_path, cache_path, start_frame, num_frames, frameskip, model=processing_model)
            
            elif input_path.is_dir() and any(f.is_file() for f in input_path.iterdir()): # Directory of images
                logging.info(f"Tracking poses for image directory {input_path.name}.")
                track_poses(input_path, cache_path, start_frame, num_frames, frameskip, model=processing_model)
                if not skip_visualization: # AITViewer for image directories if not skipped
                    if aitviewer_available:
                        logging.info(f"Visualizing poses for {input_path.name} with AITViewer.")
                        visualize_poses(
                            input_path, cache_path, output_dir / f"{input_name}_aitviewer.mp4",
                            start_frame, num_frames, frameskip, fps=30 # Default FPS for image sequence
                        )
                    else:
                        logging.warning("AITViewer not available. Skipping AITViewer visualization for directory.")
            
            elif input_path.is_file(): # Single image file (not video)
                logging.info(f"Running detection for single image {input_path.name}.")
                # run_detection handles its own simple .png visualization if not skip_visualization
                run_detection(input_path, cache_path, skip_visualization, model=processing_model, user_height_m=user_height)
            
            else:
                logging.error(f"Input path '{input_path}' is not a valid video, directory, or file for this mode.")
        else:
            # This case should ideally not be reached if not webcam and no input_path
            logging.error("No input path provided and not in webcam mode. Nothing to do.")

if __name__ == "__main__":
    # Ensure global device/use_mps are set if they are determined outside main in the original script
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # use_mps = torch.mps.is_available() and torch.backends.mps.is_built() # Example
    main()
