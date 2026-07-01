# CARLA Testing Guide

This guide outlines the procedures for building, running, and verifying the 2D/3D perception nodes in CARLA on a native Linux PC.

---

## 1. Prerequisites & Environment Setup

### X11 Forwarding & Docker Compose
Start the CARLA simulator, ROS bridge, and custom nodes with an object definition file mounted at `/config/`.

> [!WARNING] TBD — confirm which file is current
> Prior notes reference three different filenames (`obstacles.json`, `objects.json`, `objects_record.json`) without saying which is authoritative. `obstacles.json` is used below as a placeholder — swap in the correct one before relying on this guide.

> [!IMPORTANT]
> If you are testing over SSH or Wireguard, run the setup script **without** `--up` first to configure host-side X11 access controls before running any container compose commands.

```bash
# Enable GUI applications from local containers
xhost +local:docker

# Prepare host-side X11 environment (fixes Wireguard & SSH forwarding issues)
./scripts/setup_x11_forwarding.sh

# Option A: Run setup script with X11 forwarding enabled
OBJECTS_DEFINITION_FILE=/config/obstacles.json \
LAUNCH_BUILTIN_AGENT=True \
START_GLOBAL_PLANNER_CARLA=True \
VIEW=True \
./scripts/setup_x11_forwarding.sh --up

# Option B: Spin up services directly with docker compose
OBJECTS_DEFINITION_FILE=/config/obstacles.json \
LAUNCH_BUILTIN_AGENT=True \
VIEW=True \
docker compose \
  -f docker-compose.yml -f docker-compose-native.override.yml \
  up carla-server carla-ros-bridge custom-nodes -d --force-recreate
```

### Driving Mode Env Vars

These control how the ego vehicle moves in the sim; set them alongside `OBJECTS_DEFINITION_FILE` in the commands above.

| Var | Default used above | Meaning |
| :--- | :--- | :--- |
| `LAUNCH_BUILTIN_AGENT` | `True` | Use CARLA's built-in autopilot agent to drive the ego vehicle. |
| `START_GLOBAL_PLANNER_CARLA` | `True` (Option A) | Run the global route planner, which generates waypoints from a goal pose. |
| `LAUNCH_ACTUATION` | not set (`True` in prior notes) | Enable the actuation/control stack that follows waypoints. |
| `PUBLISH_TWIST` | not set (`False` in prior notes) | Publish `Twist` commands instead of / in addition to actuation commands. |

> [!WARNING] TBD — confirm intended combination
> `LAUNCH_BUILTIN_AGENT=True` (autopilot drives) and `START_GLOBAL_PLANNER_CARLA=True` (planner drives from a goal) are two different driving sources — not clear from prior notes whether both are meant to run together or these are alternatives. `LAUNCH_ACTUATION`/`PUBLISH_TWIST` weren't included in the working commands at all; add them only if you're testing the waypoint-follower path instead of the built-in agent.

### Useful Debugging Commands
```bash
# Watch the ROS bridge come up
docker compose logs -f carla-ros-bridge

# Open RViz2 from custom-nodes container
docker compose exec carla-ros-bridge bash -c \
  "source /opt/ros/humble/setup.bash && source /home/carla/carla_ros_ws/install/setup.bash && rviz2"

# Enter the custom-nodes container shell
docker compose exec custom-nodes bash
```

---

## 2. ROS2 Discovery Verification

Before starting any synchronizer node, verify that discovery is working and topics are active. If these checks fail, the nodes will sit idle waiting for topic synchronizers.

```bash
# Check if CARLA topics are visible
ros2 topic list | grep carla

# Verify that the front camera is publishing frames
ros2 topic hz /carla/ego_vehicle/rgb_front/image
```

---

## 3. Step 1: Build & Source

From your ROS2 workspace root (parent of `src/`) on the native Linux host:

```bash
# Compile the package for release
colcon build --packages-select autodriver_image_object_detection --symlink-install \
  --cmake-args ' -DCMAKE_BUILD_TYPE=Release'

# Source the workspace
source install/setup.bash
```

> [!NOTE]
> If a virtual environment is used on other targets, add `-DPython3_FIND_VIRTUALENV="ONLY"` to the CMake arguments.

---

## 4. Step 2: Confirm CARLA Topic Defaults

The default parameters in `depth_fusion_node` and `yolo_detector` are tuned to CARLA. Verify they exist:

| Sensor / Topic Role | Node Parameter Default | Expected Format / Type | Verification Command |
| :--- | :--- | :--- | :--- |
| **RGB Image** | `/carla/ego_vehicle/rgb_front/image` | `sensor_msgs/msg/Image` | `ros2 topic info /carla/ego_vehicle/rgb_front/image` |
| **RGB Camera Info** | `/carla/ego_vehicle/rgb_front/camera_info` | `sensor_msgs/msg/CameraInfo` | `ros2 topic info /carla/ego_vehicle/rgb_front/camera_info` |
| **Depth Image** | `/carla/ego_vehicle/depth_front/image` | `sensor_msgs/msg/Image` (32FC1) | `ros2 topic info /carla/ego_vehicle/depth_front/image` |
| **Depth Camera Info**| `/carla/ego_vehicle/depth_front/camera_info`| `sensor_msgs/msg/CameraInfo` | `ros2 topic info /carla/ego_vehicle/depth_front/camera_info` |
| **LiDAR (Phase 2)** | `/carla/ego_vehicle/lidar` | `sensor_msgs/msg/PointCloud2` | `ros2 topic info /carla/ego_vehicle/lidar` |

> [!TIP]
> If topic names differ in your setup, override them at runtime using parameter arguments:
> `-p input_image_topic:=/new_topic` or `-p depth_image_topic:=/new_topic`.

---

## 5. Step 3: Phase 1 (RGB + Depth → 3D)

Run the detector node with pointclouds disabled. The time synchronizer will only align the RGB image, Camera Info, and Depth image/info.

```bash
ros2 run autodriver_image_object_detection yolo_detector --ros-args \
  -p model_path:=/home/digitalstorm/bolus_ws/Repositories/autonomous_driving_simulators/data/models/yolo11x-seg.engine \
  -p use_depth:=true \
  -p use_pointcloud:=false \
  -p depth_scale:=1.0 \
  -p depth_max:=50.0 \
  -p output_frame:=ego_vehicle \
  -p show_image:=true \
  -p classes:="['person','car','truck','bicycle','motorcycle']"
```
*(If the `.engine` fails to load, swap model_path to a `.pt` model, e.g., `/abs/path/to/yolo11n-seg.pt`)*

### Verification (Phase 1)
```bash
# Verify 2D detections are publishing
ros2 topic hz /yolo/detection_results

# Verify 3D detections lifted via depth are publishing
ros2 topic hz /yolo/detection3d_depth_results
ros2 topic echo /yolo/detection3d_depth_results --once
```
* **RViz2 Visualization:** Set Fixed Frame to `ego_vehicle`. Add `/yolo/detection_image` (Annotated Image) and `/yolo/markers_depth` (Depth Markers).

---

## 6. Step 4: Phase 2 (RGB + Depth + LiDAR → 3D)

To enable the LiDAR pointcloud branch:

```bash
# Preload Open3D to prevent python import errors on some systems
export LD_PRELOAD=${HOME}/sdks/open3d_install/lib/libOpen3D.so

# Run node with pointclouds enabled
ros2 run autodriver_image_object_detection yolo_detector --ros-args \
  -p model_path:=/home/digitalstorm/bolus_ws/Repositories/autonomous_driving_simulators/data/models/yolo11x-seg.engine \
  -p use_depth:=true \
  -p use_pointcloud:=true \
  -p pointcloud_topic:=carla/ego_vehicle/lidar \
  -p output_frame:=ego_vehicle \
  -p cluster_tolerance:=1.0 \
  -p min_cluster_size:=5 \
  -p bounding_box_type:=AABB \
  -p show_image:=true
```

### Verification (Phase 2)
```bash
# Verify 3D detections from pointclouds are publishing
ros2 topic hz /yolo/detection3d_pointcloud_results
ros2 topic echo /yolo/detection3d_pointcloud_results --once
```
* **RViz2 Visualization:** Add markers from `/yolo/markers_pointcloud`.
* *Note:* If results are empty, check `ros2 topic hz /carla/ego_vehicle/lidar` and consider loosening `synchronization_interval` if messages are out-of-sync.

---

## 7. Regression Checklist (Post-Fix Re-Verification)

The following behaviors changed in recent commits and should be specifically re-checked, not just assumed working:

- [ ] **Lidar-mount centroid offset fixed** — with RGB+pointcloud running, cluster centroids on `/depth_fusion/detection3d_pointcloud` should land on the object in `ego_vehicle`/`map`, not offset by the lidar mount position.
- [ ] **`yolo_detector` pointcloud path no longer crashes** — running with `-p use_pointcloud:=true` should start and run normally, not crash at (the previously reported) line 536.
- [ ] **`.pt` model at half precision no longer errors** — `single_stream_detector -p model_path:=.../yolo11x-seg.pt` (no `export_and_exit`) should run without a `Half != float` dtype error.

```bash
ros2 topic echo /depth_fusion/detection3d_depth --once
```

---

## 8. Engine Pre-Compilation Pass (No CARLA)

To compile the model and exit before running simulated tests:

```bash
ros2 run autodriver_image_object_detection single_stream_detector --ros-args \
  -p model_path:=/home/carla/data/models/yolo11x-seg.engine \
  -p export_and_exit:=true
```

---

## 9. Parameter Notes & Recent Fixes

### Depth Fusion Convention
* **`optical_frame_id` (String, default `""`):** Supersedes `apply_optical_to_body`. When empty, the node stamps points with the camera frame from `camera_info` and lets TF handle coordinate rotations. Since CARLA camera frames are already optical-compliant (RPY `[-90,0,-90]`), keeping it empty (`""`) is correct.
* **Fallback if 3D positions look wrong:** if detections still land in the wrong place with `optical_frame_id` empty, try the older convention as a debugging step:
  ```bash
  ros2 param set /depth_fusion_node apply_optical_to_body false
  ```

### Bounding Box Extent
* **`depth_box_thickness` (Double, default `4.0`):** Upper limit on the 3D bounding box thickness along the view-axis. Used to prevent background noise from expanding the box dimension. Run the following command at runtime to tweak:
  ```bash
  ros2 param set /depth_fusion_node depth_box_thickness 4.0
  ```

### Heartbeat / Empty Costmap Clearing
* **`publish_empty_detections` (Bool, default `true`):** When no objects are detected, the node publishes an empty `Detection3DArray` and a `DELETEALL` marker to clear downstream costmaps and remove old RViz markers.
