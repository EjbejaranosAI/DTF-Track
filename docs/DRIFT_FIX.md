# Drift de bounding boxes al girar el robot

Documento de referencia: síntoma, causa raíz, intentos fallidos y solución aplicada (2026).

---

## Síntoma

Con la persona **quieta en el mapa** y el robot **girando sobre sí mismo**, las cajas 3D y los clusters LiDAR proyectados en `map` **se desplazaban** varios decímetros o metros. En RViz la caja azul quedaba desalineada respecto a los puntos verdes del láser.

Los logs `[POSE]` mostraban:
- `robot=(x,y)` del track **cambiando** en frame robot aunque la persona no se movía
- velocidades imposibles (`v > 2 m/s`) en persona estática
- pérdida de track y respawns con IDs nuevos

**No era un fallo del Kalman** como causa primaria: el filtro seguía mediciones LiDAR ya mal proyectadas.

---

## Causa raíz

Proyección **LiDAR → `map`** con incoherencia temporal entre:

1. **`header.stamp` del `/scan`** (cuándo se tomó la medida)
2. **TF usada para transformar** puntos del láser a `map`

### Mecanismo

```
scan capturado en T  +  TF de T+Δt  →  punto en mapa incorrecto
```

Si el robot **gira** entre `T` y `T+Δt`, el error en mapa es proporcional a `Δt × velocidad angular × distancia al objeto`.

### Por qué ocurría en nuestro pipeline

| Factor | Efecto |
|--------|--------|
| `sync_image_with_lidar: true` | El láser solo se procesaba cuando llegaba imagen emparejada (cola del `ApproximateTimeSynchronizer` + YOLO ~100 ms) → **lag 200–400 ms** |
| `scan_tf=latest` (fallback) | TF del instante actual, no del stamp del scan |
| Executor monohilo | YOLO bloqueaba el callback del láser → más lag |
| Cola QoS `depth: 10` | Hasta ~1 s de scans viejos en cola (con `stamped` no rompe proyección, pero baja la tasa de actualización) |

Con **lag bajo** (~2 ms) y persona quieta, `robot=(x,y)` relativo se mantenía estable. Con **lag alto + giro**, clusters y tracks derivaban.

---

## Intentos que no solucionaron el problema

| Intento | Por qué no bastó |
|---------|------------------|
| Ajustar Kalman (gates, saltos máximos) en rotación | Parchea el síntoma; las mediciones siguen mal |
| Rechazar scans con `lag > 50 ms` (opción A) | Evita updates malos pero deja el track “ciego” al LiDAR durante lag alto |
| Forzar solo `TF@stamp` con timeouts largos | Pipeline lento (~0.75 Hz), parpadeo |
| Cache TF con `Time()` al llegar el scan | Ayuda si lag es bajo; insuficiente con lag crónico |

---

## Solución aplicada

### 1. LiDAR inmediato (desacoplado del sync imagen)

- Suscripción directa a `/scan` → `_scan_cb` → `_apply_scan` al llegar el mensaje.
- El sync imagen+láser **solo** empareja cámara para YOLO; ya no bloquea la extracción de clusters.

### 2. TF capturada en `_apply_scan`

- Intentar `TF@scan.header.stamp` primero (`lookup_planar_stamped_only`).
- Fallback a `latest` solo si el stamped no está en el buffer.
- `_transform_xy` usa la TF capturada en ese scan (no un lookup posterior).

### 3. `MultiThreadedExecutor` + callback group para `/scan`

- YOLO y procesamiento de imagen no bloquean el callback del láser.

### 4. `lidar_qos_depth: 1` (suscripción directa)

- Solo el scan más reciente; evita procesar colas de ~1 s.
- El sync de imagen mantiene cola propia (`max(lidar_qos_depth, sensor_sync_queue_size)`).

### 5. Buffer TF ampliado

- `Buffer(cache_time=60 s)` para permitir lookups en stamps pasados (simulación con `use_sim_time`).

---

## Verificación con logs `[POSE]`

Activar con `debug:=true` en el launch (ver README).

Campos útiles:

| Campo | Qué indica |
|-------|------------|
| `scan_tf=stamped` | Proyección coherente con el instante del scan |
| `scan_tf=latest` | Fallback; riesgo de drift si hay lag y rotación |
| `lag` | Retraso entre `now` y `scan.header.stamp` |
| `id=N map=(x,y) robot=(x,y)` | Track en mapa y en frame robot |
| `clusters LiDAR` | Posición de clusters tras proyección |

**Criterio de éxito** (persona quieta, robot gira): `map=(x,y)` del track y clusters **estables** (~15–25 cm de variación); `robot=` y `bear` cambian (normal al girar); `dist` ~constante.

Ejemplo estable:

```
id=2 map=(-1.94,+0.71) robot=(-2.44,+0.57) dist=2.51m  yaw robot -74°
id=2 map=(-1.94,+0.71) robot=(-1.94,+1.97) dist=2.61m  yaw robot -0°
```

La posición en **mapa** se mantiene; en **robot** rota con el vehículo.

---

## Parámetros relevantes (YAML)

| Parámetro | Rol |
|-----------|-----|
| `sync_image_with_lidar` | `false` en robot (YOLO async); `true` en sim si Gazebo empareja bien |
| `sensor_sync_queue_size` | Cola del sync imagen (4 en sim/robot alineado) |
| `lidar_qos_depth` | `1` en suscripción directa `/scan` |
| `lidar_fusion_use_scan_stamp_for_tf` | Kalman usa stamp del scan para `fusion_tf_stamp` |
| `debug_pose_log_enable` | Logs `[POSE]` (solo diagnóstico) |

Detalle del pipeline: [`PIPELINE.md`](PIPELINE.md). Tracking: [`TRACKING.md`](TRACKING.md).
