# PoseTrack21 MOT Evaluation

This is a slight adaptation of the [PoseTrack21 evaluation code](https://github.com/anDoer/PoseTrack21). We opt to use the bounding-box MOT evaluation here since our model does not output the appropriate format for the keypoint-based metrics. We make several adjustments to the MOT evaluation code.

Major changes that affect tracking evaluation:
- as discussed in the paper, we change the IOU calculation for ignore regions. This is a one line change:
```
# previous
region_ious[j, i] = poly_intersection / poly_union
# ours
region_ious[j, i] = poly_intersection / det_boxes[j].area
```
- some of the PoseTrack sequences are not annotated at every frame, but the code by default would penalize false positives on un-annotated frames. We have updated the code to ignore detections on frames with no ground-truth annotations.

Minor updates:
- we no longer load images in the `PTSequence` class
- we update the dtype behavior in `motmetrics` to support more recent versions of `numpy`
- turn on `use_ignore_regions` by default (no flag to set at runtime)

## Setup

A few extra packages are needed to run the evaluation code, these can be installed with:

```
conda install geos
pip install lap pandas shapely==1.7.1 xmltodict
```

Follow the instructions provided by the authors in the [original repository](https://github.com/anDoer/PoseTrack21) to obtain a copy of the dataset and annotations.

## Running the evaluation

To run:
```
python evaluate_mot --dataset_path $PATH_TO_DATASET_ROOT \
                    --mot_path $PATH_TO_RESPECTIVE_MOT_FOLDER \
                    --result_path $FOLDER_WITH_YOUR_RESULTS \
```

We also support evaluation on individual sequences with:
```
python evaluate_mot ... --sequence_choice [SEQUENCE_ID_0] [SEQUENCE_ID_1] ...
```
where `SEQUENCE_ID` is the integer sequence for a given PoseTrack example (e.g. `000342`) - no need to include `_mpii_test`.
