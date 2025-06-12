import numpy as np
from pathlib import Path
import re

# --- User Configuration ---
INPUT_DIR = "output/trc"  # Directory containing input TRC files
OUTPUT_DIR = "output/trc_processed" # Directory to save processed TRC files

# Processing options
APPLY_Y_FLIP = True  # True to flip the Y-axis (marker_coord_Y *= -1)

# Pelvis/Reference marker zeroing options
# Note: If FORCE_ZERO_PELVIS_ALL_FRAMES is True, ZERO_PELVIS_FIRST_FRAME is effectively ignored.
ZERO_PELVIS_FIRST_FRAME = False # True to zero pelvis based on the first frame
FORCE_ZERO_PELVIS_ALL_FRAMES = True # True to force zero pelvis for all frames (overrides ZERO_PELVIS_FIRST_FRAME)
REFERENCE_MARKER_NAME = "CLAV"  # Name of the marker to use as reference for zeroing operations
                                    # Common names might be 'Pelvis', 'Hip', 'L_ASIS', 'R_ASIS', 'C7' etc.
                                    # Check your TRC files for actual marker names.

# Left-right marker swapping options
SWAP_LEFT_RIGHT_MARKERS = True  # Set to True to enable left-right marker swapping
# SWAP_COORDINATE_AXIS 설정은 더 이상 사용되지 않음 - 항상 모든 축(X,Y,Z)을 교환합니다

# Known marker pairs from the marker dictionary
# This is automatically populated by extract_marker_pairs_from_dict() if provided
KNOWN_MARKER_PAIRS = {}

# Manual marker pairs - use this for specifying additional pairs
# that may not be detected automatically
MANUAL_MARKER_PAIRS = {
    # Example: 'LANK': 'RANK',  # Uncomment and add marker pairs if needed
}

OUTPUT_FILENAME_SUFFIX = "_processed" # Suffix to add to output filenames (e.g., input.trc -> input_processed.trc)
                                     # If empty, it will overwrite original files in OUTPUT_DIR (if INPUT_DIR and OUTPUT_DIR are different)
                                     # If INPUT_DIR and OUTPUT_DIR are the same and suffix is empty, it WILL overwrite originals.
# --------------------------

def extract_marker_pairs_from_dict(marker_dict):
    """
    Extracts left-right marker pairs from a dictionary of markers with descriptions.
    
    Args:
        marker_dict: Dictionary with format {marker_name: (vertex_id, description)}
        
    Returns:
        Dictionary mapping left marker names to right marker names
    """
    pairs = {}
    
    # Create dictionaries for left and right markers based on prefix
    l_markers = {name: desc for name, (_, desc) in marker_dict.items() if name.startswith('L')}
    r_markers = {name: desc for name, (_, desc) in marker_dict.items() if name.startswith('R')}
    
    # Match markers with the same suffix
    for l_name, l_desc in l_markers.items():
        if len(l_name) <= 1:
            continue
            
        suffix = l_name[1:]
        r_name = 'R' + suffix
        
        if r_name in r_markers:
            pairs[l_name] = r_name
            print(f"  Extracted pair from dictionary: {l_name} <-> {r_name}")
    
    # Look for special cases: markers that form pairs but don't follow the L/R convention
    special_pairs = [
        ('BLTI', 'BRTI'),  # Left/Right 대퇴 후방 상단
        ('FLTI', 'FRTI'),  # Left/Right 대퇴 후방 내측
        ('RIBL', 'RIBR'),  # Left/Right 늑골
        ('RITL', 'RITR')   # Left/Right 장골 능선
    ]
    
    for left, right in special_pairs:
        if left in marker_dict and right in marker_dict:
            pairs[left] = right
            print(f"  Special case pair: {left} <-> {right}")
    
    return pairs

# Initialize KNOWN_MARKER_PAIRS from the full marker dictionary if available
full_marker_dict = {
    "BLTI": (1131, "왼쪽 대퇴 후방 상단"),
    "BRTI": (4615, "오른쪽 대퇴 후방 상단"),
    "C7": (1306, "7번 경추, 등 위쪽 (spine)"),
    "CHIN": (3051, "턱"),
    "CLAV": (3169, "쇄골 (clavicle)"),
    "FBBT": (1121, "대퇴 기저부"),
    "FLTI": (1372, "왼쪽 대퇴 후방 내측"),
    "FRBT": (4605, "오른쪽 대퇴 후방 하단"),
    "FRTI": (4845, "오른쪽 대퇴 후방 내측"),
    "LAKI": (3199, "왼쪽 복사뼈 내측"),
    "LANK": (3328, "왼쪽 발목"),
    "LBAC": (1834, "왼쪽 팔 뒤 (삼두)"),
    "LBHD": (395, "머리 뒤 좌측"),
    "LBWT": (3099, "왼쪽 허리 뒤 상단"),
    "LELB": (1657, "왼쪽 팔꿈치"),
    "LELS": (3013, "왼쪽 견갑골 아래"),
    "LELSO": (1621, "왼쪽 견갑골 외측 아래"),
    "LFBB": (1168, "왼쪽 대퇴 후방 하단"),
    "LFBT": (962, "왼쪽 대퇴 후방 상단"),
    "LFFB": (1172, "왼쪽 대퇴 전방 하단"),
    "LFFT": (909, "왼쪽 대퇴 전방 상단"),
    "LFHD": (136, "머리 앞 좌측"),
    "LFIN": (2214, "왼손 손가락 끝"),
    "LFLB": (1021, "왼쪽 대퇴 하단 후면"),
    "LFLT": (848, "왼쪽 대퇴 하단 상단"),
    "LFMB": (1028, "왼쪽 대퇴 중간 하단"),
    "LFMT": (935, "왼쪽 대퇴 전방 중간 상단"),
    "LFPI": (3223, "왼쪽 무릎 전하방"),
    "LFRM": (1688, "왼쪽 아래팔"),
    "LFWT": (3156, "왼쪽 허리 앞"),
    "LHAP": (780, "왼쪽 고관절 앞 (ASIS 유사)"),
    "LHBA": (1389, "왼쪽 고관절 뒤"),
    "LHEB": (3466, "왼쪽 무릎 뒤관절"),
    "LHEE": (3387, "왼쪽 뒤꿈치"),
    "LHFR": (628, "왼쪽 고관절 앞"),
    "LHME": (2445, "왼쪽 고관절 내측"),
    "LHPI": (2082, "왼쪽 고관절 뒤 (PSIS 유사)"),
    "LHTH": (2251, "왼쪽 대전자"),
    "LHTO": (1505, "왼쪽 대전자 외측"),
    "LKNE": (1010, "왼쪽 무릎 앞"),
    "LKNI": (1148, "왼쪽 무릎 내측"),
    "LMT5": (3303, "왼쪽 새끼발가락 기저부"),
    "LSCA": (2886, "왼쪽 견갑골"),
    "LSHN": (1083, "왼쪽 정강이"),
    "LSHO": (1239, "왼쪽 어깨 관절"),
    "LTIA": (1178, "왼쪽 경골 상단"),
    "LTIB": (1076, "왼쪽 경골 하단"),
    "LTIC": (3321, "왼쪽 경골 중심"),
    "LTOE": (3216, "왼쪽 엄지발가락"),
    "LTOP": (3348, "왼쪽 발가락 위"),
    "LTOS": (3350, "왼쪽 발가락 옆"),
    "LUMB": (3023, "요추 중심"),
    "LUMC": (3502, "요추 중앙부"),
    "LWRA": (2112, "왼쪽 손목 A"),
    "LWRB": (2108, "왼쪽 손목 B"),
    "NOSE": (410, "코"),
    "RAKI": (6599, "오른쪽 복사뼈 내측"),
    "RANK": (6727, "오른쪽 발목"),
    "RBAC": (5295, "오른쪽 팔 뒤"),
    "RBHD": (3897, "머리 뒤 우측"),
    "RBWT": (6522, "오른쪽 허리 뒤 상단"),
    "RELB": (5112, "오른쪽 팔꿈치"),
    "RELS": (6471, "오른쪽 견갑골 아래"),
    "RELSO": (5090, "오른쪽 견갑골 외측 아래"),
    "RFBB": (4655, "오른쪽 대퇴 후방 하단"),
    "RFBT": (4386, "오른쪽 대퇴 후방 상단"),
    "RFFB": (4515, "오른쪽 대퇴 전방 하단"),
    "RFFT": (4396, "오른쪽 대퇴 전방 상단"),
    "RFHD": (3648, "머리 앞 우측"),
    "RFIN": (5675, "오른손 손가락 끝"),
    "RFLB": (4507, "오른쪽 대퇴 하단 후면"),
    "RFLT": (4336, "오른쪽 대퇴 하단 상단"),
    "RFMB": (4514, "오른쪽 대퇴 중간 하단"),
    "RFMT": (4926, "오른쪽 대퇴 전방 중간 상단"),
    "RFPI": (6623, "오른쪽 무릎 전하방"),
    "RFRM": (5155, "오른쪽 아래팔"),
    "RFWT": (6573, "오른쪽 허리 앞"),
    "RHAP": (4266, "오른쪽 고관절 앞 (ASIS 유사)"),
    "RHBA": (4862, "오른쪽 고관절 뒤"),
    "RHEB": (6866, "오른쪽 무릎 뒤관절"),
    "RHEE": (6786, "오른쪽 뒤꿈치"),
    "RHFR": (4115, "오른쪽 고관절 앞"),
    "RHME": (5905, "오른쪽 고관절 내측"),
    "RHPI": (5545, "오른쪽 고관절 뒤 (PSIS 유사)"),
    "RHTH": (5714, "오른쪽 대전자"),
    "RHTO": (6282, "오른쪽 대전자 외측"),
    "RIBL": (1485, "왼쪽 늑골"),
    "RIBR": (6289, "오른쪽 늑골"),
    "RITL": (589, "왼쪽 장골 능선"),
    "RITR": (4079, "오른쪽 장골 능선"),
    "RKNE": (4495, "오른쪽 무릎 앞"),
    "RKNI": (4634, "오른쪽 무릎 내측"),
    "RMT5": (6703, "오른쪽 새끼발가락 기저부"),
    "RSCA": (4219, "오른쪽 견갑골"),
    "RSHN": (4580, "오른쪽 정강이"),
    "RSHO": (4724, "오른쪽 어깨 관절"),
    "RTIA": (4664, "오른쪽 경골 상단"),
    "RTIB": (4560, "오른쪽 경골 하단"),
    "RTIC": (6721, "오른쪽 경골 중심"),
    "RTOE": (6618, "오른쪽 엄지발가락"),
    "RTOP": (6749, "오른쪽 발가락 위"),
    "RTOS": (6750, "오른쪽 발가락 옆"),
    "RWRA": (5573, "오른쪽 손목 A"),
    "RWRB": (5568, "오른쪽 손목 B"),
    "THD": (414, "머리 꼭대기 (정수리)")
}

# Extract marker pairs from the full dictionary and add to KNOWN_MARKER_PAIRS
KNOWN_MARKER_PAIRS = extract_marker_pairs_from_dict(full_marker_dict)

def detect_marker_pairs(marker_names: list[str], manual_pairs: dict[str, str] = None, 
                       known_pairs: dict[str, str] = None) -> dict[int, int]:
    """
    Detects pairs of markers that are left-right counterparts.
    
    Args:
        marker_names: List of marker names from the TRC file
        manual_pairs: Optional dictionary of {marker_name: paired_marker_name} for manual pair specification
        known_pairs: Dictionary of known marker pairs from marker dictionaries
        
    Returns:
        Dictionary mapping marker indices to their paired marker indices {index1: paired_index1, ...}
        Both directions are included, e.g., if 3 pairs with 5, then 5 also pairs with 3
    """
    # Initialize result dictionary
    pairs_dict = {}
    used_markers = set()  # To track markers already paired
    
    # Process known pairs from marker dictionary first if provided
    if known_pairs:
        for marker1, marker2 in known_pairs.items():
            if marker1 in marker_names and marker2 in marker_names:
                idx1 = marker_names.index(marker1)
                idx2 = marker_names.index(marker2)
                pairs_dict[idx1] = idx2
                pairs_dict[idx2] = idx1
                used_markers.add(marker1)
                used_markers.add(marker2)
                print(f"  Known pair: {marker1} <-> {marker2}")
    
    # Process manual pairs next if provided
    if manual_pairs:
        for marker1, marker2 in manual_pairs.items():
            if marker1 in marker_names and marker2 in marker_names:
                if marker1 in used_markers or marker2 in used_markers:
                    print(f"  Warning: Manual pair {marker1} <-> {marker2} overlaps with already paired markers. Overriding.")
                
                idx1 = marker_names.index(marker1)
                idx2 = marker_names.index(marker2)
                pairs_dict[idx1] = idx2
                pairs_dict[idx2] = idx1
                used_markers.add(marker1)
                used_markers.add(marker2)
                print(f"  Manual pair: {marker1} <-> {marker2}")
    
    # Common prefix patterns to check
    prefix_patterns = [
        # (left_prefix, right_prefix, description)
        ('L', 'R', 'Left/Right'),
        ('LT', 'RT', 'Left/Right'),
        ('LEFT', 'RIGHT', 'Left/Right'),
        ('B', 'F', 'Back/Front'),
        ('BACK', 'FRONT', 'Back/Front'),
    ]
    
    # Check for each prefix pattern
    for left_prefix, right_prefix, desc in prefix_patterns:
        # First pass: Find markers with the left prefix
        for i, name_i in enumerate(marker_names):
            if name_i in used_markers:
                continue
                
            # Check if this marker starts with the left prefix
            if name_i.startswith(left_prefix):
                # Get the part after the prefix
                suffix_i = name_i[len(left_prefix):]
                
                # Look for a matching right marker
                right_name = right_prefix + suffix_i
                if right_name in marker_names:
                    j = marker_names.index(right_name)
                    pairs_dict[i] = j
                    pairs_dict[j] = i
                    used_markers.add(name_i)
                    used_markers.add(right_name)
                    print(f"  {desc} pair: {name_i} <-> {right_name}")
    
    # String similarity-based detection for more complex cases
    # For each marker not already paired, look for another marker that is similar except for the first character(s)
    remaining_markers = [name for name in marker_names if name not in used_markers]
    for i, name_i in enumerate(marker_names):
        if name_i in used_markers:
            continue
            
        # Check each other unpaired marker
        for j, name_j in enumerate(marker_names):
            if i == j or name_j in used_markers:
                continue
                
            # Check if they differ only in the first character
            # More sophisticated string similarity could be used here
            if len(name_i) > 1 and len(name_j) > 1 and name_i[1:] == name_j[1:]:
                pairs_dict[i] = j
                pairs_dict[j] = i
                used_markers.add(name_i)
                used_markers.add(name_j)
                print(f"  Similar name pair: {name_i} <-> {name_j}")
                break
    
    # Report unpaired markers
    unpaired = [name for name in marker_names if name not in used_markers]
    if unpaired:
        print(f"  Unpaired markers: {', '.join(unpaired)}")
        
    return pairs_dict

def parse_trc_file(file_path: Path) -> tuple[dict, list[str], np.ndarray | None, np.ndarray | None]:
    """
    Parses a .trc file and returns header info, marker names, time data, and coordinate data.
    Returns: (header_info, marker_names, time_data, coordinates_data)
             Returns (None, None, None, None) if parsing fails significantly.
    """
    header_info = {}
    marker_names = []
    time_data_list = []
    coordinates_list_of_frames = []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Basic check for TRC structure
        if len(lines) < 5:
            print(f"Error: File {file_path.name} has too few lines to be a valid TRC file.")
            return {}, [], None, None

        # Line 1: PathFileType
        header_info['PathFileType'] = lines[0].strip()
        # Line 2: Headers for DataRate, CameraRate, NumFrames, NumMarkers, Units, etc.
        header_info['DataRateHeader'] = lines[1].strip()
        # Line 3: Values for DataRate, CameraRate, NumFrames, NumMarkers, Units, etc.
        data_values_line = lines[2].strip().split('\t')
        header_info['DataRate'] = float(data_values_line[0])
        header_info['CameraRate'] = float(data_values_line[1])
        header_info['NumFrames'] = int(data_values_line[2])
        header_info['NumMarkers'] = int(data_values_line[3])
        header_info['Units'] = data_values_line[4]
        # Optional original data rate info
        if len(data_values_line) > 5: header_info['OrigDataRate'] = float(data_values_line[5])
        if len(data_values_line) > 6: header_info['OrigDataStartFrame'] = int(data_values_line[6])
        if len(data_values_line) > 7: header_info['OrigNumFrames'] = int(data_values_line[7])

        # Line 4: Marker names header
        marker_names_header = lines[3].strip().split('\t')
        marker_names = [name for name in marker_names_header[2:] if name] # Skip Frame#, Time, and empty names
        
        # Check if parsed NumMarkers matches actual marker names found
        if header_info['NumMarkers'] != len(marker_names):
            print(f"Warning: NumMarkers in header ({header_info['NumMarkers']}) does not match parsed marker names count ({len(marker_names)}) in {file_path.name}. Using parsed count.")
            header_info['NumMarkers'] = len(marker_names)
            if header_info['NumMarkers'] == 0 and header_info['NumFrames'] > 0:
                 print(f"Error: No marker names found but NumFrames > 0 in {file_path.name}. Cannot process.")
                 return {}, [], None, None

        # Line 5: Coordinate axes (X1, Y1, Z1, X2, Y2, Z2 ...)
        header_info['CoordinateHeader'] = lines[4].strip()

        # Data lines (starting from line 6)
        if header_info['NumFrames'] == 0:
            print(f"Info: File {file_path.name} has 0 frames. No coordinate data to parse.")
            return header_info, marker_names, np.array([]), np.array([]).reshape(0, header_info.get('NumMarkers', 0), 3)

        for i in range(5, len(lines)):
            line_data = lines[i].strip().split('\t')
            if not line_data or not line_data[0]: # Skip empty lines
                continue
            
            try:
                # frame_num = int(line_data[0]) # Not strictly needed if we trust order
                time_val = float(line_data[1])
                time_data_list.append(time_val)

                coords_flat = [float(c) for c in line_data[2:]]
                if len(coords_flat) != header_info['NumMarkers'] * 3:
                    print(f"Error: Incorrect number of coordinates in frame {len(time_data_list)} of {file_path.name}. Expected {header_info['NumMarkers'] * 3}, got {len(coords_flat)}. Skipping frame.")
                    # Attempt to recover by padding or truncating, or skip frame
                    # For now, we'll be strict and assume this frame is corrupt / or end of useful data
                    # If this happens, NumFrames might need adjustment later or file is considered corrupt
                    time_data_list.pop() # Remove last added time
                    continue 
                
                frame_coords = np.array(coords_flat).reshape((header_info['NumMarkers'], 3))
                coordinates_list_of_frames.append(frame_coords)
            except ValueError as ve:
                print(f"Error parsing data line {i+1} in {file_path.name}: {ve}. Skipping line.")
                continue
        
        # If frames were skipped, NumFrames might be inconsistent. Update based on actual data read.
        actual_num_frames = len(coordinates_list_of_frames)
        if actual_num_frames != header_info['NumFrames']:
            print(f"Warning: NumFrames in header ({header_info['NumFrames']}) doesn't match actual frames read ({actual_num_frames}) for {file_path.name}. Using actual frames read.")
            header_info['NumFrames'] = actual_num_frames

        if not coordinates_list_of_frames: # If all frames had issues or no frames
            if header_info.get('NumFrames', 0) > 0: # Check original expectation
                 print(f"Error: No valid coordinate frames could be parsed from {file_path.name} despite header indicating frames.")
                 return {}, [], None, None
            # If 0 frames expected and 0 frames read, this is fine.
            return header_info, marker_names, np.array(time_data_list), np.array([]).reshape(0, header_info.get('NumMarkers', 0), 3)

        return header_info, marker_names, np.array(time_data_list), np.stack(coordinates_list_of_frames)

    except Exception as e:
        print(f"Error parsing TRC file {file_path.name}: {e}")
        return {}, [], None, None

def process_trc_data(coordinates: np.ndarray, header_info: dict, marker_names: list[str],
                       apply_y_flip: bool, zero_first: bool, force_zero_all: bool, 
                       ref_marker_name: str, swap_lr_markers: bool = False,
                       manual_marker_pairs: dict[str, str] = None,
                       known_pairs: dict[str, str] = None) -> np.ndarray | None:
    """
    Processes coordinate data based on the given options.
    Returns modified coordinates_data or None if critical error (e.g., ref marker not found).
    """
    if coordinates.size == 0: # No data to process
        return coordinates

    processed_coords = coordinates.copy()
    num_frames, num_markers, _ = processed_coords.shape

    # 1. Apply Y-flip
    if apply_y_flip:
        processed_coords[:, :, 1] *= -1
        print("  Applied Y-axis flip.")

    # Determine reference marker index for zeroing operations
    ref_marker_idx = -1
    if zero_first or force_zero_all:
        try:
            ref_marker_idx = marker_names.index(ref_marker_name)
        except ValueError:
            print(f"  Error: Reference marker '{ref_marker_name}' not found in marker list: {marker_names}. Cannot perform pelvis zeroing.")
            return None # Critical error, cannot proceed with zeroing

    # 2. Force Zero Pelvis (All Frames) - takes precedence
    if force_zero_all:
        if ref_marker_idx != -1:
            for i_frame in range(num_frames):
                offset = processed_coords[i_frame, ref_marker_idx, :].copy()
                processed_coords[i_frame, :, :] -= offset
            print(f"  Applied force zero pelvis (all frames) using marker '{ref_marker_name}'.")
        else:
            # Should have been caught above, but as a safeguard
            print("  Skipping force zero pelvis: reference marker index not found.")
    # 3. Zero Pelvis (First Frame) - if force_zero_all is false
    elif zero_first:
        if ref_marker_idx != -1 and num_frames > 0:
            offset = processed_coords[0, ref_marker_idx, :].copy()
            processed_coords -= offset # Apply to all frames, all markers
            print(f"  Applied zero pelvis (first frame) using marker '{ref_marker_name}'.")
        elif num_frames == 0:
             print("  Skipping zero pelvis (first frame): no frames to process.")
        else:
            # Should have been caught above, but as a safeguard
            print("  Skipping zero pelvis (first frame): reference marker index not found.")
    
    # 4. Apply left-right marker swapping if enabled
    if swap_lr_markers:
        # Get marker pairs
        marker_pairs = detect_marker_pairs(marker_names, manual_marker_pairs, known_pairs)
        
        # Apply swapping: for each pair, swap the specified coordinates
        if marker_pairs:
            # We'll only process each pair once by checking a condition
            processed_pairs = set()
            for idx1, idx2 in marker_pairs.items():
                # Skip if already processed or invalid index
                pair_key = tuple(sorted([idx1, idx2]))
                if pair_key in processed_pairs or idx1 >= num_markers or idx2 >= num_markers:
                    continue
                
                # Swap all coordinates (X, Y, Z)
                temp_coords_idx1 = processed_coords[:, idx1, :].copy()
                processed_coords[:, idx1, :] = processed_coords[:, idx2, :].copy()
                processed_coords[:, idx2, :] = temp_coords_idx1
                
                processed_pairs.add(pair_key)
            
            print(f"  Applied left-right marker swapping to {len(processed_pairs)} marker pairs (all X,Y,Z coordinates).")
        else:
            print("  No marker pairs found for left-right swapping.")
            
    return processed_coords

def write_trc_file(output_path: Path, header_info: dict, marker_names: list[str], 
                     time_data: np.ndarray, coordinates_data: np.ndarray):
    """
    Writes the processed data to a new .trc file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            # Line 1: PathFileType (use original or a generic one)
            f.write(header_info.get('PathFileType', f"PathFileType\t4\t(X/Y/Z)\t{output_path.name}") + "\n")
            
            # Line 2: DataRateHeader
            f.write(header_info.get('DataRateHeader', "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames") + "\n")
            
            # Line 3: Values - NumFrames might have changed if parsing skipped frames
            num_frames_to_write = coordinates_data.shape[0]
            num_markers_to_write = coordinates_data.shape[1] if num_frames_to_write > 0 else len(marker_names)

            line3_parts = [
                f"{header_info.get('DataRate', 0.0):.6f}",
                f"{header_info.get('CameraRate', 0.0):.6f}",
                str(num_frames_to_write),
                str(num_markers_to_write),
                header_info.get('Units', 'm'),
                f"{header_info.get('OrigDataRate', header_info.get('DataRate', 0.0)):.6f}",
                str(header_info.get('OrigDataStartFrame', 1)),
                str(header_info.get('OrigNumFrames', num_frames_to_write))
            ]
            f.write("\t".join(line3_parts) + "\n")

            # Line 4: Marker names
            f.write("Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\n")
            
            # Line 5: Coordinate axes
            coord_headers = []
            for i in range(num_markers_to_write):
                marker_label = re.sub(r'[^a-zA-Z0-9_\-\.:]', '', marker_names[i]) if i < len(marker_names) else f"M{i+1}"
                coord_headers.extend([f"X{i+1}", f"Y{i+1}", f"Z{i+1}"]) # Use simple X,Y,Z if marker names are problematic for header
            # Original version: coord_headers = [ax + str(i+1) for i in range(num_markers_to_write) for ax in ("X","Y","Z")] 
            f.write("\t\t" + "\t".join(coord_headers) + "\n") # Two tabs for Frame# and Time columns

            # Data lines
            for i_frame in range(num_frames_to_write):
                time_stamp = time_data[i_frame] if i_frame < len(time_data) else (i_frame / header_info.get('DataRate', 1.0) if header_info.get('DataRate', 0) > 0 else 0.0)
                row_data = [str(i_frame + 1), f"{time_stamp:.6f}"]
                row_data.extend([f"{coord:.6f}" for coord in coordinates_data[i_frame].reshape(-1)])
                f.write("\t".join(row_data) + "\n")
        
        print(f"  Successfully wrote processed TRC file to: {output_path}")
    except Exception as e:
        print(f"Error writing TRC file {output_path.name}: {e}")

def main_process():
    input_dir_path = Path(INPUT_DIR)
    output_dir_path = Path(OUTPUT_DIR)

    if not input_dir_path.is_dir():
        print(f"Error: Input directory '{input_dir_path}' does not exist or is not a directory.")
        return

    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting TRC File Post-Processing ---")
    print(f"Input Directory: {input_dir_path}")
    print(f"Output Directory: {output_dir_path}")
    print(f"Apply Y-Flip: {APPLY_Y_FLIP}")
    print(f"Zero Pelvis (First Frame): {ZERO_PELVIS_FIRST_FRAME}")
    print(f"Force Zero Pelvis (All Frames): {FORCE_ZERO_PELVIS_ALL_FRAMES}")
    if ZERO_PELVIS_FIRST_FRAME or FORCE_ZERO_PELVIS_ALL_FRAMES:
        print(f"Reference Marker for Zeroing: '{REFERENCE_MARKER_NAME}'")
    print(f"Swap Left-Right Markers: {SWAP_LEFT_RIGHT_MARKERS}")
    if SWAP_LEFT_RIGHT_MARKERS:
        print(f"Known Marker Pairs: {len(KNOWN_MARKER_PAIRS)} pairs from marker dictionary")
        if MANUAL_MARKER_PAIRS:
            print(f"Manual Marker Pairs: {MANUAL_MARKER_PAIRS}")
    print(f"Output Filename Suffix: '{OUTPUT_FILENAME_SUFFIX}'")
    print("-----------------------------------------")

    trc_files_found = list(input_dir_path.glob("*.trc"))
    if not trc_files_found:
        print(f"No .trc files found in '{input_dir_path}'.")
        return

    print(f"Found {len(trc_files_found)} .trc file(s) to process.")

    for trc_file_path in trc_files_found:
        print(f"\nProcessing file: {trc_file_path.name}...")
        
        header_info, marker_names, time_data, coordinates_data = parse_trc_file(trc_file_path)

        if coordinates_data is None or marker_names is None or not header_info: # Parsing failed critically
            print(f"  Skipping file {trc_file_path.name} due to parsing errors.")
            continue
        
        if coordinates_data.size == 0 and header_info.get('NumFrames', 0) > 0:
            print(f"  Skipping file {trc_file_path.name} as no valid coordinate data was loaded despite frames indicated in header.")
            continue
        if coordinates_data.size == 0 and header_info.get('NumFrames', 0) == 0:
            print(f"  File {trc_file_path.name} has 0 frames. Writing an empty processed file.")
            # Write empty or minimal TRC if desired, or just skip
        
        processed_coordinates = process_trc_data(coordinates_data, header_info, marker_names,
                                                 APPLY_Y_FLIP, 
                                                 ZERO_PELVIS_FIRST_FRAME, 
                                                 FORCE_ZERO_PELVIS_ALL_FRAMES, 
                                                 REFERENCE_MARKER_NAME,
                                                 SWAP_LEFT_RIGHT_MARKERS,
                                                 MANUAL_MARKER_PAIRS,
                                                 KNOWN_MARKER_PAIRS)
        
        if processed_coordinates is None: # Critical error during processing (e.g. ref marker not found)
            print(f"  Skipping file {trc_file_path.name} due to processing errors.")
            continue

        output_filename = f"{trc_file_path.stem}{OUTPUT_FILENAME_SUFFIX}{trc_file_path.suffix}"
        output_file_path = output_dir_path / output_filename
        
        write_trc_file(output_file_path, header_info, marker_names, time_data, processed_coordinates)

    print("\n--- TRC File Post-Processing Complete ---")

if __name__ == "__main__":
    # Create dummy input directory and files for testing if they don't exist
    # This is just for making the script runnable standalone for a quick test.
    # In a real scenario, these directories and files would be prepared by previous steps.
    _input_dir = Path(INPUT_DIR)
    _output_dir = Path(OUTPUT_DIR)
    _input_dir.mkdir(parents=True, exist_ok=True)
    _output_dir.mkdir(parents=True, exist_ok=True)

    if not list(_input_dir.glob("*.trc")):
        print(f"Creating a dummy TRC file in '{_input_dir}' for testing as none were found.")
        dummy_trc_content = (
            "PathFileType\t4\t(X/Y/Z)\tdummy_input.trc\n"
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            "100.0\t100.0\t2\t2\tmm\t100.0\t1\t2\n"
            "Frame#\tTime\tPelvis\t\t\tL_Ankle\t\t\t\n"
            "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\n"
            "1\t0.000000\t10.0\t20.0\t30.0\t100.0\t200.0\t300.0\n"
            "2\t0.010000\t11.0\t21.0\t31.0\t101.0\t201.0\t301.0\n"
        )
        with open(_input_dir / "dummy_input.trc", "w") as df:
            df.write(dummy_trc_content)
        print(f"Dummy file 'dummy_input.trc' created. Please run the script again or populate '{_input_dir}' with your TRC files.")
        # exit() # Optionally exit after creating dummy file for user to inspect/replace

    main_process() 