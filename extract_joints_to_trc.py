from pathlib import Path # Path는 계속 사용됩니다.
import torch
import numpy as np
import pandas as pd # Pandas 추가
# import os # 사용 안 함
# import pandas as pd # 사용 안 함

# PYTHONPATH=src 환경 변수 설정 또는 프로젝트 루트에 __init__.py 파일 추가 필요
from comotion_demo.utils.smpl_kinematics import SMPLKinematics

# ---------- 기본 설정값 (main 함수 파라미터의 기본값으로 사용) ----------
DEFAULT_PT_FILE   = "Test2.pt"
DEFAULT_TRC_FILE_TEMPLATE = "Test2_OpenSim_P{person_id}.trc" # ID별 파일명 템플릿
DEFAULT_FPS       = 30.0
DEFAULT_UNITS_MM  = False
# CoMotion/SMPL (X:오른쪽, Y:위, Z:앞) -> OpenSim (X:앞, Y:위, Z:왼쪽)으로 변환
# 즉, 입력 X -> OpenSim -Z, 입력 Y -> OpenSim +Y, 입력 Z -> OpenSim +X
# idx: (X,Y,Z)에서 어떤 순서로 가져올지 -> (Z,Y,X) 순서로 가져와서 OpenSim의 X,Y,Z에 각각 대응
# sign: 가져온 값에 어떤 부호를 곱할지 -> (Z값, Y값, -X값) (X축 값의 부호 반전)
# SMPL X, Y, Z 순서 유지, Y축 부호만 반전
# OpenSim X = Input X, OpenSim Y = -Input Y, OpenSim Z = Input Z
DEFAULT_AXES_SWAP = ((0, 1, 2), (1, -1, 1))
DEFAULT_ZERO_PELVIS = False # 골반 원점화 기본값 False
DEFAULT_FORCE_ZERO_PELVIS = False # 모든 프레임 골반 강제 원점화 (시각화용)
# --------------------------------------------------------------------

SMPL_STANDARD_JOINT_NAMES = [ # SMPL 표준 관절 이름 정의
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

def load_pt(pt_path: str | Path, weights_only: bool = False) -> dict:
    """
    PyTorch .pt 파일을 로드하고 필수 키 존재 여부를 확인합니다.

    Args:
        pt_path: .pt 파일 경로.
        weights_only: torch.load의 weights_only 파라미터. 보안을 위해 True 권장.

    Returns:
        로드된 데이터 (dict).

    Raises:
        KeyError: 필수 키('pose', 'betas', 'trans', 'id', 'frame_idx')가 누락된 경우.
    """
    # PyTorch 1.13+ 부터 weights_only=True가 권장됩니다.
    # 다만, CoMotion .pt 파일이 단순 텐서 외 다른 객체를 포함할 수 있으므로
    # 사용자가 파일 내용을 알고 있을 때만 False를 사용하도록 유의해야 합니다.
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
        print(f"Warning: Expected pose dimension to be 72, but got {data['pose'].shape[-1]}. "
              f"SMPLKinematics expects full 72-dim pose.")
    return data

def smpl_kinematics_to_joints(data: dict, smpl_model: SMPLKinematics) -> torch.Tensor:
    """
    SMPLKinematics 모델을 사용하여 SMPL 파라미터로부터 관절 좌표를 계산합니다.
    입력 데이터는 PyTorch 텐서로 변환됩니다.

    Args:
        data: 'betas', 'pose', 'trans' 키를 포함하는 dict.
              이때 betas는 (num_person_frames, 10) 또는 (10,) 형태 모두 가능.
              (10,)인 경우 내부에서 num_person_frames에 맞게 확장됨.
        smpl_model: 초기화된 SMPLKinematics 모델 인스턴스.

    Returns:
        계산된 관절 좌표 (N, 24, 3) PyTorch 텐서 (미터 단위).
    """
    with torch.no_grad():
        pose = torch.as_tensor(data["pose"], dtype=torch.float32, device="cpu")
        trans = torch.as_tensor(data["trans"], dtype=torch.float32, device="cpu")
        betas_input = torch.as_tensor(data["betas"], dtype=torch.float32, device="cpu")

        num_frames = pose.shape[0]
        if betas_input.ndim == 1 and betas_input.shape[0] == 10: # (10,) 형태인 경우
            betas = betas_input.unsqueeze(0).repeat(num_frames, 1) # (num_frames, 10)으로 확장
        elif betas_input.ndim == 2 and betas_input.shape[0] == num_frames and betas_input.shape[1] == 10:
            betas = betas_input # 이미 (num_frames, 10) 형태
        else:
            raise ValueError(f"Betas must be of shape (10,) or (num_frames, 10), got {betas_input.shape}")

        joints = smpl_model(
            betas=betas,
            pose=pose,
            trans=trans,
            output_format="joints"
        )
    return joints

def write_trc(
    joints: torch.Tensor,
    out_path: str | Path,
    fps: float,
    units_mm: bool = False,
    marker_names: list[str] | None = None,
    axes_swap: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    zero_pelvis: bool = True, # 골반 원점화 옵션 추가
    force_zero_pelvis: bool = False # 모든 프레임 골반 강제 원점화 옵션
) -> None:
    """
    관절 좌표를 .trc 파일로 저장합니다.
    Args:
        joints: (N, M, 3) 모양의 PyTorch 텐서 (관절 좌표).
        out_path: 출력 .trc 파일 경로.
        fps: 초당 프레임 수.
        units_mm: True이면 단위를 밀리미터로, False이면 미터로 설정.
        marker_names: 마커 이름 목록. None이면 J00, J01, ...로 자동 생성.
        axes_swap: 축 변환을 위한 튜플 ((idx_tuple), (sign_tuple)).
                   예: ((0,2,1),(1,1,-1)) -> (X, Z, -Y).
        zero_pelvis: True이면 첫 프레임의 골반 위치를 (0,0,0)으로 이동시키고,
                     모든 프레임의 모든 관절에 동일한 오프셋을 적용합니다.
        force_zero_pelvis: True이면 각 프레임마다 해당 프레임의 골반 위치를 (0,0,0)으로 강제 고정합니다.
                           이 작업은 `zero_pelvis` 적용 후에 수행됩니다.
                           주로 시각화 및 골반 움직임 제거 목적입니다.
    """
    processed_joints = joints.clone() # 원본 수정을 피하기 위해 복제

    # (0) Pelvis 원점화 (좌표축 변환 *전*에 수행)
    if zero_pelvis:
        if processed_joints.shape[0] > 0 and processed_joints.shape[1] > 0:
            pelvis_origin = processed_joints[0:1, 0:1, :].clone() # 첫 프레임, 첫 관절(골반) 위치
            processed_joints -= pelvis_origin # 모든 프레임, 모든 관절에서 해당 오프셋 빼기
            print(f"  Pelvis zeroed based on first frame. Original pelvis[0,0]: {pelvis_origin.numpy().squeeze()}")
        else:
            print("  Warning: Cannot apply zero_pelvis, joints tensor is empty or has no joints.")

    # (0.5) 모든 프레임 골반 강제 원점화 (zero_pelvis 이후, 축 변환 전)
    if force_zero_pelvis:
        if processed_joints.shape[0] > 0 and processed_joints.shape[1] > 0:
            num_frames_for_force_zero = processed_joints.shape[0]
            for i_frame in range(num_frames_for_force_zero):
                # 현재 프레임의 골반 위치 (0번 관절)를 가져옴
                current_pelvis_pos_for_frame = processed_joints[i_frame:i_frame+1, 0:1, :].clone()
                # 현재 프레임의 모든 관절에서 해당 프레임의 골반 위치를 뺌
                processed_joints[i_frame, :, :] = processed_joints[i_frame, :, :] - current_pelvis_pos_for_frame
            print(f"  Pelvis forced to origin for all {num_frames_for_force_zero} frames.")
        else:
            print("  Warning: Cannot apply force_zero_pelvis, joints tensor is empty or has no joints.")

    # (1) numpy 변환
    J_np = processed_joints.detach().cpu().numpy()

    # (2) 축 변환
    if axes_swap:
        idx, sign = axes_swap
        assert len(idx) == 3 and all(isinstance(i, int) for i in idx), "axes_swap idx_tuple must contain 3 integers."
        assert len(sign) == 3 and all(isinstance(s, (int, float)) for s in sign), "axes_swap sign_tuple must contain 3 numbers."
        
        # 임시 변수를 사용하여 축 스왑 (in-place-like modification 방지)
        J_swapped = np.empty_like(J_np)
        for i_axis in range(3):
            J_swapped[..., i_axis] = J_np[..., idx[i_axis]] * sign[i_axis]
        J_np = J_swapped

    # (3) 단위 변환
    scale = 1000.0 if units_mm else 1.0
    J_np *= scale
    
    J_np = J_np.astype(np.float64, copy=False)

    N, M, _ = J_np.shape
    actual_marker_names = marker_names if marker_names and len(marker_names) == M else [f"J{i:02d}" for i in range(M)]
    if marker_names and len(marker_names) != M:
        print(f"Warning: Provided marker_names count ({len(marker_names)}) does not match "
              f"number of joints ({M}). Using default names J00-J{M-1}.")

    output_units = "mm" if units_mm else "m"
    out_path_obj = Path(out_path)

    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{out_path_obj.name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{fps:.1f}\t{fps:.1f}\t{N}\t{M}\t{output_units}\t{fps:.1f}\t1\t{N}",
        "Frame#\tTime\t" + "\t\t\t".join(actual_marker_names),
        "\t\t" + "\t".join([ax + str(i+1) for i in range(M) for ax in ("X","Y","Z")])
    ]

    with open(out_path_obj, "w") as f:
        for h_line in header:
            f.write(h_line + "\n")
        for i_frame in range(N):
            time_stamp = i_frame / fps
            row_data = [str(i_frame + 1), f"{time_stamp:.6f}"]
            row_data += [f"{coord:.6f}" for coord in J_np[i_frame].reshape(-1)]
            f.write("\t".join(row_data) + "\n")
    print(f"✓ TRC file saved to {out_path_obj} ({N} frames, {M} markers, {output_units})")

def main(
    pt_file: str | Path = DEFAULT_PT_FILE,
    trc_file_template: str = DEFAULT_TRC_FILE_TEMPLATE,
    fps: float = DEFAULT_FPS,
    units_mm: bool = DEFAULT_UNITS_MM,
    axes_swap: tuple[tuple[int, int, int], tuple[int, int, int]] | None = DEFAULT_AXES_SWAP,
    zero_pelvis: bool = DEFAULT_ZERO_PELVIS, # main 함수에도 옵션 추가
    force_zero_pelvis: bool = DEFAULT_FORCE_ZERO_PELVIS, # 모든 프레임 골반 강제 원점화 옵션 추가
    custom_marker_names: list[str] | None = None,
    load_weights_only: bool = False
):
    """
    SMPL 파라미터 .pt 파일을 각 사람(ID)별 .trc 파일로 변환하는 메인 함수.
    Args:
        zero_pelvis: True이면 각 사람의 첫 프레임 골반을 원점으로 이동.
        force_zero_pelvis: True이면 각 사람의 모든 프레임에서 골반을 (0,0,0)으로 강제 고정.
                           (주로 시각화 목적, zero_pelvis 이후에 적용됨)
    (이하 Args 설명은 이전과 동일)
    """
    print(f"Starting multi-person conversion: {pt_file}")
    full_data = load_pt(pt_file, weights_only=load_weights_only)
    
    smpl_model = SMPLKinematics().eval()
    smpl_model.to("cpu")

    # Pandas DataFrame으로 데이터 래핑
    # SMPLKinematics에 전달하기 전에 numpy()로 변환 후 list()로 감싸는 것은
    # DataFrame이 각 행에 다차원 배열을 직접 저장하기 어렵기 때문입니다.
    # PyTorch 텐서를 직접 다루는 것이 더 효율적일 수 있으나, 정렬 및 그룹화에는 Pandas가 편리합니다.
    df_data_list = []
    for i in range(full_data['id'].shape[0]):
        df_data_list.append({
            "id": full_data["id"][i].item(),
            "frame_idx": full_data["frame_idx"][i].item(),
            "pose": full_data["pose"][i],
            "trans": full_data["trans"][i],
            "betas": full_data["betas"][i]
        })
    df = pd.DataFrame(df_data_list)

    unique_ids = sorted(df["id"].unique())
    print(f"Found {len(unique_ids)} unique person IDs: {unique_ids}")

    for person_id in unique_ids:
        print(f"\nProcessing data for person ID: {person_id}")
        person_df = df[df["id"] == person_id].sort_values(by="frame_idx")
        
        if person_df.empty:
            print(f"No data found for person ID: {person_id}. Skipping.")
            continue

        # 정렬된 데이터로부터 텐서 재구성
        person_pose = torch.stack(person_df["pose"].tolist())
        person_trans = torch.stack(person_df["trans"].tolist())
        # Betas는 해당 사람의 첫 번째 프레임 값 사용 (모든 프레임에서 동일하다고 가정)
        # SMPLKinematics가 (10,) 형태의 betas를 처리하도록 smpl_kinematics_to_joints 함수 수정
        person_betas_first_frame = person_df["betas"].iloc[0] 

        person_data_for_smpl = {
            "pose": person_pose,
            "trans": person_trans,
            "betas": person_betas_first_frame # (10,) 형태의 텐서
            # frame_idx는 SMPL 모델에 직접 필요하지 않음
        }

        print(f"  Person {person_id}: {person_pose.shape[0]} frames.")
        print(f"  Pose shape: {person_pose.shape}, Trans shape: {person_trans.shape}, Betas (first frame): {person_betas_first_frame.shape}")

        joints = smpl_kinematics_to_joints(person_data_for_smpl, smpl_model)
        
        # ID별로 출력 파일명 생성
        # Path 객체를 사용하여 안전하게 파일명 조립
        base_trc_path = Path(trc_file_template)
        # person_id를 문자열로 변환 (정수일 수 있으므로)
        actual_trc_file = base_trc_path.with_name(base_trc_path.name.format(person_id=str(person_id)))
        
        # custom_marker_names가 None이면 SMPL_STANDARD_JOINT_NAMES 사용
        marker_names_to_use = custom_marker_names if custom_marker_names is not None else SMPL_STANDARD_JOINT_NAMES

        write_trc(
            joints,
            actual_trc_file,
            fps,
            units_mm=units_mm,
            marker_names=marker_names_to_use,
            axes_swap=axes_swap,
            zero_pelvis=zero_pelvis, # 옵션 전달
            force_zero_pelvis=force_zero_pelvis # 새 옵션 전달
        )
    
    print("\nMulti-person conversion complete.")

if __name__ == "__main__":
    # --- 실행 예시 (argparse 등으로 대체 가능) ---
    # 기본값 사용
    # main()
    
    # 사용자 정의 값 사용 예시
    # main(pt_file="path/to/your/input.pt",
    #      trc_file="path/to/your/output.trc",
    #      fps=60.0,
    #      units_mm=True,
    #      axes_swap=((0, 2, 1), (1, 1, -1)), # 예: (X,Z,-Y)
    #      custom_marker_names=["Pelvis", "L_Hip", ...] # 24개 이름 제공
    # )

    # 스크립트 실행 시 기본값으로 동작하도록 main() 호출
    main()

    # --- PYTHONPATH 안내 ---
    # 이 스크립트를 프로젝트 루트 외부에서 실행하거나,
    # `src` 폴더가 파이썬의 기본 검색 경로에 없는 경우,
    # `from comotion_demo.utils.smpl_kinematics import SMPLKinematics` 라인에서
    # ImportError가 발생할 수 있습니다.
    # 해결 방법:
    # 1. 터미널에서 스크립트 실행 시 PYTHONPATH 환경 변수 설정:
    #    PYTHONPATH=/path/to/your/project/root/src python extract_joints_to_trc.py
    #    (또는 프로젝트 루트에서 PYTHONPATH=src python your_script_path/extract_joints_to_trc.py)
    # 2. 프로젝트 루트 디렉토리에 빈 __init__.py 파일을 생성하여 `src`를 패키지로 만듭니다.
    #    그러면 `src`가 포함된 상위 디렉토리에서 파이썬 인터프리터가 `comotion_demo`를 찾을 수 있게 됩니다.
    #    (예: /project_root/__init__.py, /project_root/src/comotion_demo/...)
    #    이 경우, IDE 설정에 따라 import 방식 조정이 필요할 수 있습니다. 