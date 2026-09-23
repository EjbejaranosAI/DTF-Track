# Tracking (Kalman + asociación) — `dynamic_tracking`

Referencia: **Spencer** `srl_nearest_neighbor_tracker` en `old_spencer/`, launches en `example/iri_spencer_tracker_launch/`.

## Diseño 360° (láser primero)

1. **Detección** (`lidar_detection/`): segmentos → fusión (`require_two`, etc.) → clusters en `map`.
2. **Tracking** (`track_pipeline.update_tracks`): Kalman 4D + Hungarian + verificación.
3. **Confirmación** (`apply_yolo_auxiliary_person_confirmation`): YOLO solo metadatos (color, bbox), **sin** `correct()` en persona.

## Ciclo por frame (orden)

| Paso | Qué hace |
|------|----------|
| 1 | `predict(dt)` — velocidad constante; `dt` del stamp del `/scan` (cap `kalman_predict_dt_cap_sec`) |
| 2 | Asociación LiDAR (Hungarian): fase 1 high-score → 1b suelo laxo → fase 2 resto |
| 3 | `correct(x,y)` solo con cluster láser emparejado |
| 4 | Spawn + `lidar_spawn_confirm_frames` (Logic Initiator simplificado) |
| 5 | **Scan verify** (Spencer): sin cluster, ¿hay retorno en `/scan` en (r,θ) esperado? → mantiene track, **no** mueve Kalman |
| 6 | FOV: en FOV sin cluster ni verify → `missed_frames` (con debounce); fuera FOV → `consecutive_occlusions` |
| 7 | `occluded_update()` si no hubo medición |
| 8 | Poda por `missed` / oclusiones; dedup mapa; YOLO auxiliar |

## Verificación en el tracker (evitar FP y desapariciones)

| Mecanismo | Anti-FP | Anti-desaparición |
|-----------|---------|-------------------|
| **Fusión `require_two`** | Menos clusters espurios | Puede faltar 1 frame → debounce |
| **Filtros mapa/costmap** | Antes del tracker | — |
| **`lidar_spawn_confirm_frames`** | No abre track al primer blip | Retraso spawn ~2 frames |
| **`use_mahalanobis_lidar: true`** | Gate χ²; no empareja saltos raros | Re-asocia tras oclusión (cov crece) |
| **`scan_verify_*`** | Puede mantener FP en muro (rayo en pared) | Persona quieta sin cluster ese frame |
| **`person_missed_debounce_frames`** | — | 1–2 frames sin detección no borran |
| **`max_occlusions_*`** | — | Persona detrás del robot (360°) |
| **`person_lidar_only_max_missed_frames`** | — | Track sin YOLO aguanta ~88 ciclos |
| **YOLO auxiliar** | No crea posición | Confirma clase / verde |

## Spencer vs nosotros

| Spencer NNT | Nosotros |
|-------------|----------|
| Nodo separado, observaciones `DetectedPerson` | Mismo nodo: segmentación + track |
| `world_frame` típ. `odom` | `tracking_frame` típ. `map` |
| Occlusion manager **antes** del Hungarian | FOV binario + `consecutive_occlusions` |
| Logic Initiator (N scans, velocidad) | `PendingLidarSpawn` + `confirm_frames` |
| EKF + Mahalanobis en asociación | OpenCV Kalman + Mahalanobis opcional |
| `updateOccludedTrack` | `occluded_update()` |
| Low-confidence laser recovery | No (equivalente: fase 1b + verify) |
| `filter_tracks.launch` post-filtro | RViz / `require_visual_confirmation` |

## Parámetros clave (`dynamic_tracking_params.yaml`)

```yaml
use_mahalanobis_lidar: true
person_missed_debounce_frames: 2
scan_verify_range_tolerance_m: 0.65
scan_verify_window_deg: 22.0
tracking_person_lidar_assoc_cost_threshold: 1.0
tracking_person_max_missed_frames: 90
person_lidar_only_max_missed_frames: 88
max_occlusions_before_deletion: 50
lidar_spawn_confirm_frames: 2
tracking_min_confirmed_hits: 2
```

## Síntomas y ajuste

| Síntoma | Prueba |
|---------|--------|
| Desaparecen con persona visible | Subir `person_missed_debounce_frames` (3), `scan_verify_range_tolerance_m`, `tracking_person_lidar_assoc_cost_threshold` |
| Caja se mueve (robot quieto) | Jitter cluster + AMCL en `map`; probar `tracking_frame: odom` |
| FP en muro persisten | `require_two`, mapa, subir `lidar_spawn_person_min_leg_score`; bajar `scan_verify` tolerancia |
| Parpadeo IDs / nuevo ID al re-detectar | `person_missed_debounce_frames: 2`, `person_lidar_only_max_missed_frames: 22–28`, `scan_verify_*` un poco mas laxo, `tracking_person_lidar_assoc_cost_threshold` ~1.25–1.35 |
| Dos cajas misma persona (p. ej. ID 9+12) | `person_track_map_merge_lidar_only: false`, IoU ~0.18 |
| Caja salta al girar robot | `person_lidar_max_correction_jump_m: 0.55` |
| Caja deriva con robot quieto | `person_static_zero_velocity_speed_m: 0.08` |

## Referencias en `example/`

- **`iri_spencer_tracker_launch`**: relay sensores + `freiburg_people_tracking` (detección fusionada + NNT + filtros).
- **`smart_track`**: Kalman 3D con `N_meas` / covarianza para confirmar track (drones); idea similar a `tracking_min_confirmed_hits` + initiator.
