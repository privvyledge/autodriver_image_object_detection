"""
Usage:
    * python roi_tracker_counter.py <video_source> --weights yolo11n.pt --tracker bytetrack.yaml --show-help --show-img --save-video --video-save-path output.mp4

    Valid video sources: video file, rtsp stream, camera int, etc.

Step:
    1. Load the model
    2. Load the image
    3. Initialize regions of interest and their counters
    4. Run the model on the image and detect objects
    5. Find if any of the objects are in the regions of interest
    6. Track the objects
    7. Count the number of objects in the regions of interest

Todo:
    * run the tracker over the entire image [done]
    * add counter per class [done]
    * Implement detection in ROIs [done]
    * Implement counting in ROIs [done]
    * Implement tracking in ROIs [done]
    * fix video saving [done]
    * fix plotting all detections, i.e annotate the object myself since this will always plot all detections regardless of ROI and set flag [done]
    * Implement detection in ROIs using masking, i.e bitwise_and
    * run the tracker over specific objects in ROI
    * Implement separate tracking for objects in ROIs instead of tracking across the entire image
    * Create a ROS2 node
    * Switch to joblib for multithreading/multiprocessing (See: https://docs.pytorch.org/torchcodec/stable/generated_examples/decoding/parallel_decoding.html#method-3-multiprocessing)
    * Intersection tests (point in polygon, IoU, convex hull intersection)
    * Implement detection class (e.g similar to asOneDetector for future modularity)
    * Setup Torch interface
    * Setup batch processing
    * Switch CV2 to Transparent API
"""
import os
import pathlib
import argparse
import time
import json
from collections import defaultdict
from typing import Union, Any

import numpy as np
# import sklearn
import cv2
from ultralytics import YOLO
from interactive_cv_gui import CVGUI

try:
    import torch, torch.utils.dlpack, torchvision
    import torchvision.transforms as transforms
    from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from autodriver_image_object_detection.utils.geometry_utils import ensure_ccw_polygon
except ImportError:
    def ensure_ccw_polygon(pts):
        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
        signed_area = 0.5 * float(
            np.sum(pts_arr[:, 0] * np.roll(pts_arr[:, 1], -1) -
                   np.roll(pts_arr[:, 0], -1) * pts_arr[:, 1])
        )
        if signed_area < 0:
            pts_arr = pts_arr[::-1]
        return pts_arr.reshape(-1, 1, 2).astype(np.int32)


class ROITracker:
    def __init__(self, source=None, device='cpu', model_weight='yolo11n.pt', tracker='bytetrack.yaml', classes=None,
                 track_by_masking=False, plot_all_detections=False,
                 objects_path='objects.json', save_counts_path='counts.json',
                 enable_redetection_in_other_regions=True,
                 cv_gui=None, show_help=True, window_name='ROI Counter', object_image_path='object_counter_snapshot.png'):
        self.cap = None
        self.source = source
        if source is not None:
            try:
                # attempt opening a live camera using the V4L2 interface
                self.cap = cv2.VideoCapture(int(source))
            except ValueError:
                self.cap = cv2.VideoCapture(source)
        self.device = device.lower()
        self.model_weight = model_weight
        self.tracker = tracker
        self.plot_all_detections = plot_all_detections
        self.objects_path = objects_path
        self.save_counts_path = save_counts_path
        self.enable_redetection_in_other_regions = enable_redetection_in_other_regions

        # Setup Model
        use_gpu = 'cuda' in self.device and TORCH_AVAILABLE and torch.cuda.is_available()
        self.torch_device = torch.device('cuda:0' if use_gpu else 'cpu')
        # automatically get the best accelerator (CUDA, MTIA, XPU, MPS, HPU)
        torch_version = list(map(int,torch.__version__.split('.')[:2]))
        if torch_version[0] >= 2 and torch_version[1] >= 5:
            self.torch_device = torch.accelerator.current_accelerator() if use_gpu else torch.device('cpu')
        self.imgsz = None
        self.model = YOLO(f"{self.model_weight}")
        try:
            self.model.to("cuda") if use_gpu else self.model.to("cpu")
        except TypeError as e:
            # print(e)
            pass
        self.track_by_masking = track_by_masking  # note: does not update with change in regions
        self.track_history = defaultdict(list)
        self.track_history_length = 500
        self.counted_ids = dict()
        self.results = None
        self.total_counts = defaultdict(int)
        self.mask = None

        # Filter classes
        self.class_names: dict[int, str] = self.model.names
        num_model_classes = len(self.class_names)
        self.class_names_inv = {v: k for k, v in self.class_names.items()}
        self.supported_class_names = set(self.class_names_inv.keys())
        self.supported_class_keys = set(self.class_names.keys())

        if isinstance(classes[0], str):
            assert all(x in self.supported_class_names for x in classes)
            self.classes = [self.class_names_inv[x.strip()] for x in classes]

        # if classes is a list of ints
        elif isinstance(classes[0], int):
            assert all(x in self.supported_class_keys for x in classes)
            self.classes = classes

        # Setup Inference Parameters
        self.inference_dict = {
            'conf': 0.5,
            'iou': 0.5,
            'device': self.torch_device,
            'agnostic_nms': False,
            'max_det': 300,
            'half': True,
            'retina_masks': True,
            'show': False,
            'augment': False,
            'verbose': True,  # todo: set to False
            'classes': self.classes
        }

        self.show_help = show_help
        self.window_name = window_name
        self.object_image_path = object_image_path
        if cv_gui is None:
            cv_gui = CVGUI(source=None, in_place=False, save_file=self.objects_path,
                           classes=[self.class_names[x] for x in self.classes],
                           default_op_mode='add', default_add_type='polygon',
                           window_name=self.window_name, show_help=self.show_help,
                           save_object_image=self.object_image_path, horizontal_padding=0,
                           debug=False, gpu=False)
        self.cv_gui = cv_gui

    def save_counts(self):
        if not self.save_counts_path:
            return
        region_dict = {}
        for obj_idx, region in enumerate(self.cv_gui.objects[::-1]):
            if region.type not in ['rect', 'polygon']:
                continue
            region_idx = len(self.cv_gui.objects) - 1 - obj_idx  # region.id
            id = f"{region.label}"  # region_idx, f"region_id_region.id"
            region_dict[id] = region.count  # self.cv_gui.objects[region_idx].count

        save_dict = {
            'total_count': self.total_counts,  # global counts
            'regional_counts': region_dict,  # region counts
            'tracked_object_ids': {f"object_{k}": v for k, v in  self.counted_ids.items()}, # tracked ids. Useful for debugging
        }

        # save to json
        with open(self.save_counts_path, 'w') as f:
            json.dump(save_dict, f, indent=2)

    def plot_tracks(self, x, y, image, track_id):
        track = self.track_history[track_id]  # Tracking Lines plot
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > self.track_history_length:
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False,
                      color=(230, 230, 230), thickness=5)
        return track, points

    def run(self, frame, show_image=True, draw_label=True, draw_centroid=True, centroid_text='count', text_color=None):
        if self.imgsz is None:

            self.imgsz = frame.shape[:2]

        # run_start_time = time.time()
        detection_image = frame
        # run the model
        if not self.track_by_masking:
            # start_time = time.perf_counter()
            results = self.model.track(frame, persist=True, tracker=self.tracker, **self.inference_dict)
            # print(f"Detection rate: {1 / (time.perf_counter() - start_time)}")
        else:
            # todo: test and finish up
            if self.mask is None:  # Create a mask for the region
                self.mask = np.zeros_like(frame[:, :, 0])
                region_masks = [
                    np.array(region.pts, np.int32).reshape((-1, 1, 2)) for region in self.cv_gui.objects[::-1] if region.type in ['polygon', 'rect']]
                cv2.fillPoly(self.mask, pts=region_masks, color=(255, 255, 255))
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)
            results = self.model.track(masked_frame, persist=True, tracker=self.tracker, **self.inference_dict)

        # parse the results
        for result in results:
            if self.plot_all_detections:
                # update the image with the results
                detection_image = result.plot(
                        conf=True,
                        labels=True,
                        boxes=True,
                        masks=True,
                        probs=True,
                )

            bounding_box = result.boxes.cpu()  # Boxes object for bounding box outputs. n x 4
            classes = result.boxes.cls.cpu()  # n,
            confidence_score = result.boxes.conf.cpu()  # n,
            masks = result.masks
            keypoints = result.keypoints
            obb = result.obb
            probs = result.probs
            num_detections = bounding_box.xyxy.shape[0]

            if bounding_box.shape[0] < 1:
                continue

            track_ids = None
            if bounding_box.is_track:
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()

            else:
                continue

            indices_in_rois = []
            for detection_idx, (box, cls) in enumerate(zip(bounding_box, classes)):
                # preprocess
                bbox = box.xywh.cpu().numpy().flatten()
                x, y, w, h = bbox

                # track
                if box.id is not None:
                    track_id = box.id.int().cpu().item()
                else:
                    continue

                # class-aware centroid: bottom-center for person, center otherwise
                plot_cy = y + h if self.class_names[cls.item()] == 'person' else y

                # plot tracks
                if self.plot_all_detections:
                    track, points = self.plot_tracks(x, plot_cy, detection_image, track_id)

                if track_id in self.counted_ids.keys():
                    indices_in_rois.append(detection_idx)
                    # plot tracks
                    if not self.plot_all_detections:
                        track, points = self.plot_tracks(x, plot_cy, detection_image, track_id)

                # iterate through the regions and check if the box is in any of them
                for obj_idx, region in enumerate(self.cv_gui.objects[::-1]):
                    if region.type in ['polygon', 'rect']:
                        pts = ensure_ccw_polygon(region.pts)
                        x_, y_ = x, plot_cy

                        if cv2.pointPolygonTest(pts, (x_, y_), False) >= 0:
                            region_idx = len(self.cv_gui.objects) - 1 - obj_idx  # region.id

                            # global counts
                            if track_id not in self.counted_ids.keys():
                                self.counted_ids[track_id] = [region_idx]
                                cls_name = self.class_names[cls.item()]
                                self.total_counts[cls_name] += 1
                                self.cv_gui.objects[region_idx].count += 1
                                indices_in_rois.append(detection_idx)
                                # plot tracks
                                if not self.plot_all_detections:
                                    track, points = self.plot_tracks(x, plot_cy, detection_image, track_id)

                            # polygon/local/region counts
                            if self.enable_redetection_in_other_regions and (region_idx not in self.counted_ids[track_id]):
                                self.counted_ids[track_id].append(region_idx)
                                self.cv_gui.objects[region_idx].count += 1

            if not self.plot_all_detections:
                # update the image with detections within the ROIs
                image_tensor = torch.tensor(frame)
                image_tensor = image_tensor.permute(2, 0, 1)  # from (H, W, C) -> (C, H, W)
                plot_label = [
                    f"id: {track_ids[det_]}    {self.class_names[int(classes[det_].item())]}   {confidence_score[det_]:.2f}" for det_ in indices_in_rois
                ]   # id: id    cls    score
                detection_image_tensor = draw_bounding_boxes(image_tensor,
                                                             bounding_box.xyxy[indices_in_rois, :].cpu(),
                                                             labels=plot_label)
                if hasattr(result, "masks") and masks is not None:
                    detection_image_tensor = draw_segmentation_masks(detection_image_tensor, masks.data[indices_in_rois, :, :].bool().cpu())
                detection_image = detection_image_tensor.permute(1, 2, 0).numpy()

                # # plot tracks
                # xs, ys = bounding_box.xywh[indices_in_rois, :].flatten().tolist()[:2]
                # track, points = self.plot_tracks(xs, yy, detection_image, result.boxes.id[indices_in_rois].tolist())

        # modify the results in place
        self.results = results

        # update the GUI
        count_per_class = [f"{k}: {v}" for k, v in self.total_counts.items()]
        key, continue_stream = self.cv_gui.run(detection_image,
                                        show_image=show_image,
                                        draw_label=draw_label, draw_centroid=draw_centroid,
                                        centroid_text=centroid_text, text_color=text_color,
                                        help_extra_lines=count_per_class)

        out_image = self.cv_gui.last_canvas  # detection_image
        # print(f"Run rate: {1 / (time.time() - run_start_time)}")
        return self.results, key, continue_stream, out_image

    def stream_and_track(self, source=None, show_image=True, save_video='',
                         save_video_in_orig_res=False, loop=False, loop_delay=0, num_loops=-1):
        # stream_start_time = time.time()
        if source is not None:
            try:
                # attempt opening a live camera using the V4L2 interface
                self.cap = cv2.VideoCapture(int(source))
            except ValueError:
                self.cap = cv2.VideoCapture(source)

        video_writer = None
        if save_video:
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID, MJPG, mp4v, X264, mpeg

        frame_number = 0
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if loop:
            loop_number = 0

        # iterate over frames
        while self.cap.isOpened():
            # cap_start_time = time.time()
            ret, frame = self.cap.read()
            frame_number += 1
            # print(f"Frame_number: {frame_number}")
            if not ret:  # or frame_number == num_frames
                print(f"End of stream reached.")
                if loop:
                    # ret should return False at the end of the video
                    print(f"Looping after {loop_delay} seconds...")
                    if loop_delay > 0:
                        time.sleep(loop_delay)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    if 0 < num_loops <= loop_number:
                        print(f"Looping {num_loops} times complete. Terminating...")
                        break

                    loop_number += 1
                    continue
                else:
                    break

            # run the model and extract the results
            # run_start_time = time.time()
            results, key, continue_stream, out_image = self.run(frame, show_image=show_image, draw_label=True)
            # end_time = time.time()
            # print(f"Stream run rate: {1 / (end_time - run_start_time)},  {1 / (end_time - cap_start_time)}....")

            if not continue_stream:
                break

            # if show_image:
            #     cv2.imshow("Object Tracker", out_image)
            #     cv2.waitKey(1)

            if save_video:
                if video_writer is None:
                    if save_video_in_orig_res:
                        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (frame_width, frame_height))
                    else:
                        # new_width, new_height = self.cv_gui.window_width, self.cv_gui.window_height
                        new_height, new_width = out_image.shape[:2]
                        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (new_width, new_height))

                # resize out_image to frame_width and frame_height since the cv_gui may have resized it
                if save_video_in_orig_res:
                    out_image = cv2.resize(out_image, (frame_width, frame_height))

                video_writer.write(out_image)

            # print(f"Frame_number: {frame_number}")
            # print(f"Stream rate: {1 / (time.time() - cap_start_time)}.... \n")

        if save_video:
            video_writer.release()

        self.cv_gui.save_objects()
        self.save_counts()
        self.cap.release()
        cv2.destroyAllWindows()


def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for the region counting application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Path to a video source, e.g video file, rtsp stream, etc.")
    parser.add_argument("--device", default="cuda:0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--weights", type=str, default="yolo11n-seg.pt", help="initial weights path")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="path to tracker config file")
    parser.add_argument("--classes", nargs="+", default=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'],
                        type=str, help="filter by class: --classes person, or --classes person car bicycle")
    parser.add_argument("--track-by-masking", action="store_true", help="track by mask the image first with the region mask. Note: does not update with change in regions")
    parser.add_argument("--plot-all-detections", action="store_true",
                        help="Whether to plot all detections (True) or only detections within ROIs.")
    parser.add_argument("--regions-path", type=str, default="objects.json", help="The path to load the objects/regions.")
    parser.add_argument("--save-counts-path", type=str, default="counts.json", help="The path to save the counts.")
    parser.add_argument("--enable-redetection-in-other-regions", action="store_true", help="Whether to redectect a tracked object in other regions.")
    parser.add_argument("--show-help", action="store_true",
                        help="Whether to show help message for the region counting application.")
    parser.add_argument("--regions-img-path", type=str, default="object_counter_snapshot.png", help="The path to save a picture of the regions.")
    parser.add_argument("--window-name", type=str, default="ROI Counter", help="The name of the window.")

    parser.add_argument("--show-img", action="store_true", help="show results")
    parser.add_argument("--save-video", action="store_true", help="save results")
    parser.add_argument("--video-save-path", type=str, default="roi_counter.mp4", help="The path to save the video.")
    parser.add_argument("--save-video-in-orig-res", action="store_true", help="Save the video in the original resolution or allow resizing.")
    parser.add_argument("--loop-video", action="store_true", help="Loop the video.")
    parser.add_argument("--loop-video-delay", type=int, default=0, help="Delay between loops in seconds.")
    parser.add_argument("--loop-video-num-loops", type=int, default=-1, help="Number of times to loop. -1 for infinite.")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """Execute the main region counting functionality with the provided options."""
    options_dict = vars(options)
    counter = ROITracker(source=options.source, device=options.device, model_weight=options.weights,
                         tracker=options.tracker, classes=options.classes, track_by_masking=options.track_by_masking,
                         plot_all_detections=options.plot_all_detections,
                         objects_path=options.regions_path, save_counts_path=options.save_counts_path,
                         enable_redetection_in_other_regions=options.enable_redetection_in_other_regions,
                         cv_gui=None, show_help=options.show_help,
                         window_name=options.window_name, object_image_path=options.regions_img_path)
    counter.stream_and_track(
            source=None, show_image=options.show_img,
            save_video=options.video_save_path if options.save_video else None,
            save_video_in_orig_res=options.save_video_in_orig_res,
            loop=options.loop_video, loop_delay=options.loop_video_delay, num_loops=options.loop_video_num_loops
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)