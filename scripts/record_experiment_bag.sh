#!/usr/bin/env bash
# Graba rosbag2 con inputs + salidas de dynamic_tracking (TIAGO / ROS 2).
#
# Uso interactivo (pregunta experimento, run y etiqueta opcional):
#   ./scripts/record_experiment_bag.sh
#
# Uso con argumentos (sin preguntas):
#   ./scripts/record_experiment_bag.sh T1 run01
#   ./scripts/record_experiment_bag.sh T2 run02 figura8
#
# Carpeta de bags (por defecto ~/iri-lab/rosbag_edison). Override:
#   BAG_ROOT=/otra/ruta ./scripts/record_experiment_bag.sh T1 run01
#
# Antes de grabar: dynamic_tracking lanzado, /scan y /detections/tracked con Hz.

set -euo pipefail

ROBOT_ID="${ROBOT_ID:-tiago-pro-13c}"
BAG_ROOT="${BAG_ROOT:-$HOME/iri-lab/rosbag_edison}"
QOS_OVERRIDES="${QOS_OVERRIDES:-$BAG_ROOT/qos_overrides.yaml}"

# ─── Catálogo de experimentos (orden sugerido de sesión) ─────────────────
# Copia SCENARIO + RUN desde aquí al grabar:
#
#   T1 run01  |  Robot: gira ~360° in situ     |  Persona: estática
#   T1 run02  |  (mismo guion)
#   T1 run03  |
#
#   T2 run01  |  Robot: estático                |  Persona: figura 8
#   T2 run02  |
#   T2 run03  |
#
#   T3 run01  |  Robot: sigue / patrulla       |  Persona: camina libre
#   T3 run02  |
#   T3 run03  |
#
#   A2 run01  |  Ablation: spawn max_score      |  escena tipo T3
#   A4 run01  |  Ablation: TF sin stamp         |  escena tipo T1
#   A5 run01  |  Ablation: sin YOLO             |  escena tipo T2
#
# Duración orientativa: T1 60–90 s | T2/T3 90–120 s

declare -A EXP_DESC=(
  [T1]="Robot gira ~360° in situ | Persona estática (H1 drift)"
  [T2]="Robot estático | Persona figura 8"
  [T3]="Robot patrulla/sigue | Persona camina libre"
  [A2]="Ablación: lidar_spawn_fusion_policy=max_score"
  [A4]="Ablación: TF sin stamp (drift)"
  [A5]="Ablación: enable_yolo_detections=false"
)

sanitize_tag() {
  local raw="${1:-}"
  raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_' | sed 's/^_//;s/_$//')"
  echo "${raw:0:32}"
}

prompt_experiment() {
  echo ""
  echo "=== Experimentos (orden sugerido) ==="
  for key in T1 T2 T3 A2 A4 A5; do
    printf "  %-4s  %s\n" "$key" "${EXP_DESC[$key]:-}"
  done
  echo ""
  while true; do
    read -r -p "Experimento [T1]: " ans
    ans="${ans:-T1}"
    ans="$(echo "$ans" | tr '[:lower:]' '[:upper:]')"
    if [[ -n "${EXP_DESC[$ans]+x}" ]]; then
      SCENARIO="$ans"
      break
    fi
    if [[ "$ans" =~ ^[A-Z0-9][A-Z0-9_-]*$ ]]; then
      echo "  (experimento custom: $ans)"
      SCENARIO="$ans"
      break
    fi
    echo "  Ejemplo: T1, T2, T3, A2, A4, A5"
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
    echo "  Ejemplo: run01, run02, run03"
  done
}

prompt_tag() {
  read -r -p "Etiqueta corta para el nombre (opcional, Enter = omitir): " ans
  TAG="$(sanitize_tag "$ans")"
}

# ─── Argumentos o prompts ───────────────────────────────────────────────
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

# ─── Topics: inputs pipeline + odom (contexto) + salidas ───────────────
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
  echo "Creado QoS overrides: $QOS_OVERRIDES"
fi

echo ""
echo "=== Resumen ==="
echo "  Robot:      $ROBOT_ID"
echo "  Escenario:  $SCENARIO"
if [[ -n "${EXP_DESC[$SCENARIO]+x}" ]]; then
  echo "  Guion:      ${EXP_DESC[$SCENARIO]}"
fi
echo "  Run:        $RUN_ID"
[[ -n "$TAG" ]] && echo "  Etiqueta:   $TAG"
echo "  Bag:        ${OUT_DIR}/${BAG_NAME}"
echo ""

echo "=== Pre-flight ==="
missing=0
for t in /scan /detections/tracked; do
  if ! ros2 topic list 2>/dev/null | grep -qx "$t"; then
    echo "  WARN: $t no visible"
    missing=1
  fi
done
if [[ "$missing" -eq 0 ]]; then
  echo "  Topics OK"
fi

if ros2 node list 2>/dev/null | grep -q dynamic_tracking_node; then
  ros2 param dump /dynamic_tracking_node > "${OUT_DIR}/params_${BAG_NAME}.yaml" 2>/dev/null || true
  echo "  Params guardados en ${OUT_DIR}/params_${BAG_NAME}.yaml"
else
  echo "  WARN: dynamic_tracking_node no detectado — params no volcados"
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

read -r -p "Pulsa Enter para empezar a grabar (Ctrl+C para parar la grabación)..." _

echo ""
echo "=== Grabando ==="
ros2 bag record \
  -o "${OUT_DIR}/${BAG_NAME}" \
  --qos-profile-overrides-path "$QOS_OVERRIDES" \
  "${TOPICS[@]}"

echo ""
echo "=== Bag guardado ==="
ros2 bag info "${OUT_DIR}/${BAG_NAME}" || true
echo "Notas: ${OUT_DIR}/README_${BAG_NAME}.txt"
