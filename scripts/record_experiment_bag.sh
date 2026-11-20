#!/usr/bin/env bash
# Records a rosbag2 with the inputs + outputs of dynamic_tracking (TIAGO / ROS 2).
#
# Interactive use (asks for experiment, run and an optional label):
#   ./scripts/record_experiment_bag.sh
#
# Use with arguments (no questions):
#   ./scripts/record_experiment_bag.sh T1 run01
#   ./scripts/record_experiment_bag.sh T2 run02 figure8
#
# Bag folder (default ~/rosbags/dtf_track). Override:
#   BAG_ROOT=/other/path ./scripts/record_experiment_bag.sh T1 run01
#
# Before recording: dynamic_tracking running, /scan and /detections/tracked publishing.

set -euo pipefail

ROBOT_ID="${ROBOT_ID:-tiago-pro-13c}"
BAG_ROOT="${BAG_ROOT:-$HOME/rosbags/dtf_track}"
QOS_OVERRIDES="${QOS_OVERRIDES:-$BAG_ROOT/qos_overrides.yaml}"

# ─── Experiment catalogue (suggested session order) ──────────────────────
# Copy SCENARIO + RUN from here when recording:
#
#   T1 run01  |  Robot: turns ~360° in place   |  Person: static
#   T1 run02  |  (same script)
#   T1 run03  |
#
#   T2 run01  |  Robot: static                 |  Person: figure 8
#   T2 run02  |
#   T2 run03  |
#
#   T3 run01  |  Robot: follows / patrols      |  Person: walks freely
#   T3 run02  |
#   T3 run03  |
#
#   A2 run01  |  Ablation: spawn max_score      |  T3-like scene
#   A4 run01  |  Ablation: TF without stamp     |  T1-like scene
#   A5 run01  |  Ablation: no YOLO              |  T2-like scene
#
# Suggested duration: T1 60–90 s | T2/T3 90–120 s

declare -A EXP_DESC=(
  [T1]="Robot turns ~360° in place | Person static (H1 drift)"
  [T2]="Robot static | Person figure 8"
  [T3]="Robot patrols/follows | Person walks freely"
  [A2]="Ablation: lidar_spawn_fusion_policy=max_score"
  [A4]="Ablation: TF without stamp (drift)"
  [A5]="Ablation: enable_yolo_detections=false"
)

sanitize_tag() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_' | sed 's/^_//;s/_$//')"
  echo "${raw:0:32}"
}

prompt_experiment() {
  echo ""
  echo "=== Experiments (suggested order) ==="
  for key in T1 T2 T3 A2 A4 A5; do
    printf "  %-4s  %s\n" "$key" "${EXP_DESC[$key]:-}"
  done
  echo ""
  while true; do
    read -r -p "Experiment [T1]: " ans
    ans="${ans:-T1}"
    ans="$(echo "$ans" | tr '[:lower:]' '[:upper:]')"
    if [[ -n "${EXP_DESC[$ans]+x}" ]]; then
      SCENARIO="$ans"
      break
    fi
    if [[ "$ans" =~ ^[A-Z0-9][A-Z0-9_-]*$ ]]; then
      echo "  (custom experiment: $ans)"
      SCENARIO="$ans"
      break
    fi
    echo "  Example: T1, T2, T3, A2, A4, A5"
  done
}

prompt_run() {
  while true; do
    read -r -p "Run [run01]: " ans
    ans="${ans:-run01}"
  if [[ "$ans" =~ ^run[0-9]{2}$ ]] || [[ "$ans" =~ ^[a-zA-Z0-9_-]+$ ]]; then
      RUN_ID="$ans"
      break
    fi
    echo "  Example: run01, run02, run03"
  done
}

prompt_tag() {
  read -r -p "Short label for the name (optional, Enter = skip): " ans
  TAG="$(sanitize_tag "$ans")"
}

# ─── Arguments or prompts ───────────────────────────────────────────────
ARG_COUNT=$#

if [[ $ARG_COUNT -ge 1 ]]; then
  SCENARIO="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
else
  prompt_experiment
fi

if [[ $ARG_COUNT -ge 2 ]]; then
  RUN_ID="$2"
else
  prompt_run
fi

if [[ $ARG_COUNT -ge 3 ]]; then
  TAG="$(sanitize_tag "$3")"
else
  prompt_tag
fi

DATE_TAG="$(date +%Y-%m-%d)"
if [[ -n "$TAG" ]]; then
  BAG_NAME="${ROBOT_ID}_${SCENARIO}_${RUN_ID}_${TAG}_${DATE_TAG}"
else
  BAG_NAME="${ROBOT_ID}_${SCENARIO}_${RUN_ID}_${DATE_TAG}"
fi
OUT_DIR="${BAG_ROOT}/${BAG_NAME}"

# ─── Topics: pipeline inputs + odom (context) + outputs ─────────────────
TOPICS=(
  /scan
  /tf
  /tf_static
  /map
  /head_front_camera/color/image_raw
  /head_front_camera/color/camera_info
  /mobile_base_controller/odom
  /detections/tracked
  /dynamic_tracks_markers
)

mkdir -p "$OUT_DIR" "$BAG_ROOT"

if [[ ! -f "$QOS_OVERRIDES" ]]; then
  cat > "$QOS_OVERRIDES" <<'EOF'
/head_front_camera/color/image_raw:
  reliability: best_effort
  history: keep_last
  depth: 10
/head_front_camera/color/camera_info:
  reliability: best_effort
  history: keep_last
  depth: 10
EOF
  echo "Created QoS overrides: $QOS_OVERRIDES"
fi

echo ""
echo "=== Summary ==="
echo "  Robot:      $ROBOT_ID"
echo "  Scenario:   $SCENARIO"
if [[ -n "${EXP_DESC[$SCENARIO]+x}" ]]; then
  echo "  Script:     ${EXP_DESC[$SCENARIO]}"
fi
echo "  Run:        $RUN_ID"
[[ -n "$TAG" ]] && echo "  Label:      $TAG"
echo "  Bag:        ${OUT_DIR}/${BAG_NAME}"
echo ""

echo "=== Pre-flight ==="
missing=0
for t in /scan /detections/tracked; do
  if ! ros2 topic list 2>/dev/null | grep -qx "$t"; then
    echo "  WARN: $t not visible"
    missing=1
  fi
done
if [[ "$missing" -eq 0 ]]; then
  echo "  Topics OK"
fi

if ros2 node list 2>/dev/null | grep -q dynamic_tracking_node; then
  ros2 param dump /dynamic_tracking_node > "${OUT_DIR}/params_${BAG_NAME}.yaml" 2>/dev/null || true
  echo "  Params saved to ${OUT_DIR}/params_${BAG_NAME}.yaml"
else
  echo "  WARN: dynamic_tracking_node not found — params not dumped"
fi

{
  echo "robot: ${ROBOT_ID}"
  echo "scenario: ${SCENARIO}"
  echo "run_id: ${RUN_ID}"
  [[ -n "$TAG" ]] && echo "tag: ${TAG}"
  echo "date: ${DATE_TAG}"
  echo "description: ${EXP_DESC[$SCENARIO]:-(custom)}"
  echo "topics: ${TOPICS[*]}"
} > "${OUT_DIR}/README_${BAG_NAME}.txt"

read -r -p "Press Enter to start recording (Ctrl+C to stop)..." _

echo ""
echo "=== Recording ==="
ros2 bag record \
  -o "${OUT_DIR}/${BAG_NAME}" \
  --qos-profile-overrides-path "$QOS_OVERRIDES" \
  "${TOPICS[@]}"

echo ""
echo "=== Bag saved ==="
ros2 bag info "${OUT_DIR}/${BAG_NAME}" || true
echo "Notes: ${OUT_DIR}/README_${BAG_NAME}.txt"
