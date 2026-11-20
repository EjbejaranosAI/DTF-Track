import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_params_file(context):
    explicit = LaunchConfiguration("params_file").perform(context).strip()
    if explicit:
        return explicit
    profile = LaunchConfiguration("profile").perform(context).strip().lower()
    pkg = get_package_share_directory("dynamic_tracking")
    name = (
        "dynamic_tracking_params_sim.yaml"
        if profile == "sim"
        else "dynamic_tracking_params.yaml"
    )
    return os.path.join(pkg, "config", name)


def _maybe_float(context, key, node_params):
    val = LaunchConfiguration(key).perform(context).strip()
    if val:
        node_params[key] = float(val)


def _launch_setup(context, *args, **kwargs):
    params_path = _resolve_params_file(context)

    node_params = {
        "image_topic": LaunchConfiguration("image_topic"),
        "detections_topic": LaunchConfiguration("detections_topic"),
        "annotated_image_topic": LaunchConfiguration("annotated_image_topic"),
        "model_name": LaunchConfiguration("model_name"),
        "enable_lidar_fusion": LaunchConfiguration("enable_lidar_fusion"),
        "lidar_topic": LaunchConfiguration("lidar_topic"),
        "device": LaunchConfiguration("device"),
        "detection_source": LaunchConfiguration("detection_source"),
        "external_detections_topic": LaunchConfiguration(
            "external_detections_topic"
        ),
        "enable_yolo_detections": LaunchConfiguration("enable_yolo_detections"),
        "publish_dot_yolo_crops": LaunchConfiguration("publish_dot_yolo_crops"),
        "lidar_cluster_use_stamped_tf": LaunchConfiguration("lidar_cluster_use_stamped_tf"),
        "debug_tf_projection": LaunchConfiguration("debug_tf_projection"),
    }
    img_qos = LaunchConfiguration("image_qos_reliability").perform(context).strip()
    if img_qos:
        node_params["image_qos_reliability"] = img_qos
    area_enable = LaunchConfiguration("area_filter_enable").perform(context).strip()
    if area_enable.lower() not in ("", "auto"):
        node_params["area_filter_enable"] = area_enable.lower() in ("true", "1", "yes")
    spawn_all_det = LaunchConfiguration("lidar_spawn_require_all_detectors").perform(context).strip()
    if spawn_all_det.lower() not in ("", "auto"):
        node_params["lidar_spawn_require_all_detectors"] = spawn_all_det.lower() in (
            "true",
            "1",
            "yes",
        )
    reid_enable = LaunchConfiguration("enable_reid").perform(context).strip()
    if reid_enable.lower() not in ("", "auto"):
        node_params["enable_reid"] = reid_enable.lower() in ("true", "1", "yes")
    zones_file = LaunchConfiguration("area_filter_zones_file").perform(context).strip()
    if zones_file:
        node_params["area_filter_zones_file"] = "" if zones_file.lower() == "none" else zones_file
    area_zones = LaunchConfiguration("area_filter_zones_json").perform(context).strip()
    if area_zones:
        node_params["area_filter_zones_json"] = area_zones
    for key in (
        "area_filter_x_min",
        "area_filter_x_max",
        "area_filter_y_min",
        "area_filter_y_max",
    ):
        _maybe_float(context, key, node_params)

    debug_val = LaunchConfiguration("debug").perform(context).strip().lower()
    if debug_val in ("true", "1", "yes"):
        node_params["debug_pose_log_enable"] = True
    elif debug_val in ("false", "0", "no"):
        node_params["debug_pose_log_enable"] = False

    use_sim_time_val = LaunchConfiguration("use_sim_time").perform(context).strip().lower()
    use_sim_time = use_sim_time_val in ("true", "1", "yes")
    
    tracking_node = Node(
        package="dynamic_tracking",
        executable="dynamic_tracking_node",
        name="dynamic_tracking_node",
        parameters=[
            params_path,
            node_params,
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )

    rqt_image_view_node = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="rqt_image_view_detections",
        condition=IfCondition(LaunchConfiguration("show_image_view")),
        remappings=[("image", LaunchConfiguration("annotated_image_topic"))],
        output="screen",
    )

    pkg = get_package_share_directory("dynamic_tracking")
    rviz_config = os.path.join(pkg, "rviz", "dynamic_tracking.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_dynamic_tracking",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
        output="screen",
    )

    return [tracking_node, rqt_image_view_node, rviz_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "profile",
                default_value="robot",
                description="robot (default, real TIAGO) or sim (Gazebo)",
                choices=["robot", "sim"],
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="Override YAML; empty = chosen by profile",
            ),
            DeclareLaunchArgument(
                "model_name", default_value="yolo26n.pt", description="YOLO model filename"
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/head_front_camera/color/image_raw",
                description="Camera Image topic",
            ),
            DeclareLaunchArgument(
                "detections_topic",
                default_value="/detections",
                description="Detections topic name",
            ),
            DeclareLaunchArgument(
                "annotated_image_topic",
                default_value="/image_annotated",
                description="Annotated image topic name",
            ),
            DeclareLaunchArgument(
                "enable_lidar_fusion",
                default_value="true",
                description="Enable YOLO + LiDAR fusion (LaserScan)",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/scan",
                description="LiDAR topic (LaserScan)",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="auto",
                description="YOLO device: auto, cpu, cuda:0, mps",
            ),
            DeclareLaunchArgument(
                "detection_source",
                default_value="internal",
                description="internal or external",
            ),
            DeclareLaunchArgument(
                "external_detections_topic",
                default_value="/external/detections",
                description="vision_msgs/Detection2DArray if detection_source:=external",
            ),
            DeclareLaunchArgument(
                "show_image_view",
                default_value="false",
                description="Open rqt_image_view on the annotated image",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Open RViz2 with rviz/dynamic_tracking.rviz",
            ),
            DeclareLaunchArgument(
                "enable_yolo_detections",
                default_value="true",
                description="YOLO + auxiliary confirmation",
            ),
            DeclareLaunchArgument(
                "publish_dot_yolo_crops",
                default_value="false",
                description="Publish /dot_yolo_crops",
            ),
            DeclareLaunchArgument(
                "image_qos_reliability",
                default_value="",
                description="best_effort or reliable; empty = YAML. On a QoS RELIABILITY WARN, try the other value.",
            ),
            DeclareLaunchArgument(
                "enable_reid",
                default_value="auto",
                description="OSNet Re-ID (recovers IDs by appearance): true|false|auto (auto = YAML).",
            ),
            DeclareLaunchArgument(
                "area_filter_enable",
                default_value="auto",
                description="true/false or auto (uses the YAML of the profile)",
            ),
            DeclareLaunchArgument(
                "lidar_spawn_require_all_detectors",
                default_value="auto",
                description="true = RF+AdaBoost+leg needed to open an ID; false/auto = YAML (require_two)",
            ),
            DeclareLaunchArgument(
                "area_filter_zones_file",
                default_value="",
                description="Zones from a file: name in config/zones/ (e.g. lab_perimeter, "
                "lab_rectangle) or path to a .yaml; empty = YAML; none = no zones file",
            ),
            DeclareLaunchArgument(
                "area_filter_zones_json",
                default_value="",
                description='Override JSON ROI; empty = YAML or x_min/x_max/y_min/y_max',
            ),
            DeclareLaunchArgument(
                "area_filter_x_min",
                default_value="",
                description="ROI rectangle x_min in map (m); empty = YAML",
            ),
            DeclareLaunchArgument(
                "area_filter_x_max",
                default_value="",
                description="ROI rectangle x_max in map (m); empty = YAML",
            ),
            DeclareLaunchArgument(
                "area_filter_y_min",
                default_value="",
                description="ROI rectangle y_min in map (m); empty = YAML",
            ),
            DeclareLaunchArgument(
                "area_filter_y_max",
                default_value="",
                description="ROI rectangle y_max in map (m); empty = YAML",
            ),
            DeclareLaunchArgument(
                "lidar_cluster_use_stamped_tf",
                default_value="false",
                description="Use TF captured at scan arrival (fixes drift during rotation)",
            ),
            DeclareLaunchArgument(
                "debug_tf_projection",
                default_value="false",
                description="Log TF projection debug info",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation time (required for Gazebo/simulation)",
            ),
            DeclareLaunchArgument(
                "debug",
                default_value="false",
                description="true: [POSE] logs every ~100 ms (see docs/DRIFT_FIX.md)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
