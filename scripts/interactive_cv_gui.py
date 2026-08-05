"""
Usage:
    *

Todo:
    * switch Opencv to transparent API (not urgent)
"""
import os
import time
import json
import argparse
from typing import Union, Any
import numpy as np
import cv2
# from cv2 import gapi

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


def draw_help(img):
    lines = [
        "h: toggle help",
        "+: add mode | -: delete mode | e: edit mode | o: Add/delete classes",
        "In add mode: p=point, l=line, g=polygon(SPACE BAR or double click to close), r=rect, a=arrow",
        "Delete: click to remove object",
        "Edit: click & drag (1) within the shape to move or (2) near a vertex to edit",
        "c: clear all | Esc: exit & save"
    ]
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (400, 160), (0, 0, 0), -1)
    alpha = 0.5
    beta = (1.0 - alpha)
    cv2.addWeighted(overlay, alpha, img, beta,0.0, img)
    y = 25
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20

    return y


class DrawableObject:
    def __init__(self, obj_id, obj_type, pts, region_color=(0, 255, 0), thickness=2, label=None, text_color=None,
                 count=0):
        self.id = obj_id
        self.type = obj_type  # 'point','line','polygon','rect','arrow'
        self.pts = pts  # list of points [(x,y),...]
        self.region_color = region_color
        self.text_color = text_color
        if self.text_color is None:
            self.text_color = (0, 0, 0)
        self.thickness = thickness
        self.label = label or f"{obj_type}_{obj_id}"
        self.count = count

    def centroid(self):
        # xs = [p[0] for p in self.pts]
        # ys = [p[1] for p in self.pts]
        # centroid = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
        return np.mean(self.pts, axis=0, dtype=int).tolist()

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'pts': self.pts,
            'region_color': self.region_color,
            'text_color': self.text_color,
            'thickness': self.thickness,
            'label': self.label,
            'count': self.count
        }

    @staticmethod
    def from_dict(d):
        color = tuple(int(c * 255) if c <= 1.0 else c for c in d.get('region_color', (0, 1.0, 0)))
        return DrawableObject(
                d['id'],
                d['type'],
                d['pts'],
                color,
                d.get('thickness', 2),
                d.get('label'),
                tuple(d.get('text_color', (0, 255, 0))),
                0,  # d.get('count', 0)
        )

    def __str__(self):
        return (f"DrawableObject(type={self.type}, pts={self.pts}, region_color={self.region_color}, "
                f"thickness={self.thickness}, label={self.label}), text_color={self.text_color}), count={self.count}")


class CVGUI:
    def __init__(self, source: Union[int, str] = None, in_place: bool = False, save_file: str = 'objects.json',
                 classes=None, default_op_mode: str = 'add', default_add_type: str = 'polygon',
                 window_name: str = 'CV GUI', show_help: bool = True, save_object_image: str = 'cv_gui_snapshot.png',
                 horizontal_padding: int = 200,
                 debug: bool = False, gpu: bool = False):
        """
        Instructions:
            1. Start this module and specify the starting mode. Default: Add -> Polygon
            2. Press + to switch to "add" mode.
            3. Press - to switch to "delete" mode.
            4. Press e to switch to "move"/"edit" mode.
            5. Press h to toggle the help message
            6. Press o to add/delete classes
                i. Add the class name then press ENTER when done to add.
                ii. Type -class to delete a class from the list
            7. In "add" mode:
                i. Press "p" to add points
                ii. Press "l" to add line segments
                iii. Press "g" to add polygons:
                    a. Left click to add points
                    b. Press SPACE_BAR or double click to close the polygon
                iv. Press "r" to add axis-aligned rectangles
                    a. Hold down the left button to start.
                    b. Release the left button to stop.
                v. Press "a" to add an arrow
                    a. Hold down the left button to start.
                    b. Release the left button to stop.
            7. In "delete" mode:
                i. Select anywhere near the object to delete.
            8. In "move"/"edit" mode:
                i. Click anywhere inside the object to move.
                ii. Click anywhere near a vertex to adjust.

        :param source:
        :param in_place:
        :param save_file:
        :param window_name:
        :param gpu:
        :param debug:
        """
        self.cap = None
        if source is not None:
            self.cap = cv2.VideoCapture(source)

        self.in_place = in_place
        self.window = window_name

        self.objects = []  # list of DrawableObject
        self.classes = classes if classes else []  # list of object class strings
        self.op_mode = default_op_mode  # 'add','delete','edit'
        self.add_type = default_add_type
        self.letter_to_mode_mapping = {'p': 'point', 'l': 'line', 'g': 'polygon', 'r': 'rect', 'a': 'arrow'}
        self.type_to_color_mapping = {'point': (255, 0, 0), 'line': (0, 255, 0), 'polygon': (0, 0, 255),
                                      'rect': (0, 255, 255), 'arrow': (255, 255, 0)}
        self.selected_idx = None
        self.dragging = False
        self.edit_vertex = False
        self.vertex_idx = None
        self.drag_offset = (0,0)
        self.current_pts = []  # For polygon and arrow
        self.show_help = show_help
        self.save_file = save_file
        self.save_object_image = save_object_image
        self.horizontal_padding = horizontal_padding
        self.next_id = 0
        self.last_canvas = None

        self.gpu = gpu and TORCH_INSTALLED and torch.cuda.is_available()
        if self.save_file:
            self.load_objects()
        self.debug = debug

        self.space_bar_pressed = False

        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.on_mouse)

    def load_objects(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                data = json.load(f)
            for d in data.get('objects', []):
                obj = DrawableObject.from_dict(d)
                self.objects.append(obj)
                self.next_id = max(self.next_id, obj.id + 1)
            self.classes = data.get('classes', [])

    def save_objects(self):
        data = {
            'objects': [obj.to_dict() for obj in self.objects],
            'classes': self.classes
        }
        # save to json
        if self.save_file:
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)

        # save last image
        if self.last_canvas is not None and self.save_object_image:
            fname = self.save_object_image
            cv2.imwrite(fname, self.last_canvas)
            print(f"Saved snapshot to {fname}.")

    def edit_classes(self):
        print("Current classes:", self.classes)
        inp = input("Enter class to add or '-class' to remove (empty to finish): ")
        while inp:
            if inp.startswith('-'):
                cls = inp[1:]
                if cls in self.classes:
                    self.classes.remove(cls)
                    print(f"Removed '{cls}'")
            else:
                if inp not in self.classes:
                    self.classes.append(inp)
                    print(f"Added '{inp}'")
            inp = input("Enter class to add or '-class' to remove (empty to finish): ")

    def handle_key(self, key):
        c = chr(key) if key < 256 else ''
        if c == 'h':
            self.show_help = not self.show_help
        elif c == '+':
            self.op_mode = 'add'
            self.current_pts.clear()
        elif c == '-':
            self.op_mode = 'delete'
            self.current_pts.clear()
        elif c == 'e':
            self.op_mode = 'edit'
            self.current_pts.clear()
        elif c == 'o':  # edit class list
            self.edit_classes()
        elif self.op_mode == 'add' and c in ['p', 'l', 'g', 'r', 'a']:
            self.add_type = self.letter_to_mode_mapping[c]
            self.current_pts.clear()
        elif key == ord('c'):
            # delete/clear all objects
            self.objects.clear()
            self.selected_idx = None
        elif key == ord(' '):  # SPACE_BAR = 32
            self.space_bar_pressed = True

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if self.op_mode == 'add':
            self.handle_add(event, x, y)
        elif self.op_mode == 'delete':
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self.find_object(x, y)
                if idx is not None:
                    del self.objects[idx]
        elif self.op_mode == 'edit':
            self.handle_edit(event, x, y)

    def handle_add(self, event, x, y, text_color=(0, 0, 0)):
        t = self.add_type
        if t == 'point' and event == cv2.EVENT_LBUTTONDOWN:
            self.objects.append(
                    DrawableObject(self.next_id,'point', [(x, y)],
                                   region_color=self.type_to_color_mapping['point'], text_color=text_color))
            self.next_id += 1
        elif t == 'line':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_pts = [(x,y)]
            elif event == cv2.EVENT_LBUTTONUP and len(self.current_pts)==1:
                self.current_pts.append((x,y))
                self.objects.append(
                        DrawableObject(self.next_id,'line', self.current_pts.copy(),
                                       region_color=self.type_to_color_mapping['line'], text_color=text_color))
                self.current_pts.clear()
                self.next_id += 1
        elif t == 'polygon':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_pts.append((x,y))
            elif (event == cv2.EVENT_LBUTTONDBLCLK) or self.space_bar_pressed and len(self.current_pts)>=3:
                self.objects.append(
                        DrawableObject(self.next_id,'polygon', self.current_pts.copy(),
                                       region_color=self.type_to_color_mapping['polygon'], text_color=text_color))
                self.current_pts.clear()
                self.next_id += 1
                self.space_bar_pressed = False
        elif t == 'rect':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_pts = [(x,y)]
            elif event == cv2.EVENT_LBUTTONUP and len(self.current_pts) == 1:
                self.current_pts.append((x,y))
                x0,y0 = self.current_pts[0]
                x1,y1 = self.current_pts[1]
                rect = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
                self.objects.append(
                        DrawableObject(self.next_id,'polygon', rect,
                                       region_color=self.type_to_color_mapping['rect'], text_color=text_color))
                self.next_id += 1
                self.current_pts.clear()
        elif t == 'arrow':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_pts = [(x,y)]
            elif event == cv2.EVENT_LBUTTONUP and len(self.current_pts)==1:
                self.current_pts.append((x,y))
                self.objects.append(
                        DrawableObject(self.next_id,'arrow', self.current_pts.copy(),
                                       region_color=self.type_to_color_mapping['arrow'], text_color=text_color))
                self.next_id += 1
                self.current_pts.clear()

    def handle_edit(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            # check vertex first
            res = self.find_vertex(x,y)
            if res:
                idx, vid = res
                self.selected_idx = idx
                self.vertex_idx = vid
                self.edit_vertex = True
                self.dragging = True
            else:
                idx = self.find_object(x,y)
                if idx is not None:
                    self.selected_idx=idx
                    self.dragging=True
                    self.edit_vertex=False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.selected_idx is not None:
            dx = x - self.drag_offset[0] if hasattr(self,'drag_offset') else 0
            dy = y - self.drag_offset[1] if hasattr(self,'drag_offset') else 0
            obj = self.objects[self.selected_idx]
            if self.edit_vertex and self.vertex_idx is not None:
                obj.pts[self.vertex_idx] = (x,y)
            else:
                # move whole
                obj.pts = [(px+dx, py+dy) for px,py in obj.pts]
            self.drag_offset = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging=False
            self.edit_vertex=False
            self.vertex_idx=None
        # initialize drag offset
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_offset = (x, y)

    def find_object(self, x, y, tol=5):
        # todo: switch to numpy to speed up
        for i, obj in enumerate(self.objects[::-1]):
            # reverse for topmost first
            if obj.type in ['polygon','rect']:
                pts = np.array(obj.pts, np.int32).reshape((-1, 1, 2))
                if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
                    return len(self.objects) - 1 - i  # obj.id
            else:
                for px, py in obj.pts:
                    if (x - px) ** 2 + (y - py) ** 2 < tol ** 2:
                        return len(self.objects) - 1 - i
        return None

    def find_vertex(self, x, y, tol=8):
        # todo: switch to numpy to speed up
        for i, obj in enumerate(self.objects[::-1]):
            for j, (px, py) in enumerate(obj.pts):
                if abs(px - x) < tol and abs(py - y) < tol:
                    return len(self.objects) - 1 - i,j
        return None

    def draw_objects(self, img, draw_label=True, draw_centroid=True, centroid_text='coords', text_color=(255, 255, 255)):
        for obj in self.objects:
            text_color_ = text_color
            if text_color is None:
                text_color_ = obj.region_color

            if obj.type == 'point':
                cv2.circle(img, obj.pts[0], 5, obj.region_color, -1)
            elif obj.type == 'line':
                cv2.line(img, obj.pts[0], obj.pts[1], obj.region_color, obj.thickness)
            elif obj.type == 'arrow':
                cv2.arrowedLine(img, obj.pts[0], obj.pts[1], obj.region_color, obj.thickness)
            else:  # obj.type == 'polygon' | 'rect'
                pts = np.array(obj.pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, obj.region_color, obj.thickness)

            # draw label
            x0, y0 = min(p[0] for p in obj.pts), min(p[1] for p in obj.pts)  # np.array(obj.pts).min(0)
            if draw_label:
                cv2.putText(img, obj.label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_, 2)

            # centroid
            cx, cy = obj.centroid()
            if draw_centroid:
                cv2.circle(img, (cx, cy), 3, obj.region_color, -1)

            if centroid_text:
                centroid_text_ = centroid_text
                if centroid_text == 'coords':
                    centroid_text_ = f"({cx}, {cy})"
                elif centroid_text == 'label':
                    centroid_text_ = obj.label
                elif centroid_text == 'id':
                    centroid_text_ = str(obj.id)
                elif centroid_text == 'count':
                    centroid_text_ = str(obj.count)

                # automatically determine the bounding box needed to fit the text
                text_size, _ = cv2.getTextSize(
                        centroid_text_, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
                )
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                offset = 5 if draw_centroid else 0

                # add a rectangle around the text
                cv2.rectangle(
                        img,
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),  # + offset
                        obj.region_color,
                        2,  # -1
                )
                # draw the text on the image
                cv2.putText(
                        img, centroid_text_, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color_,  # + offset
                        2
                )

        # preview current shape
        if self.current_pts and self.op_mode=='add':
            for i in range(len(self.current_pts) - 1):
                cv2.line(img, self.current_pts[i], self.current_pts[i + 1], (0, 0, 255), 1)

    def draw_extras(self, canvas, y=0, classes=None, extra_lines=None):
        # display classes
        if classes:
            cls_txt = "Classes: " + ", ".join(classes)
            y += 10
            cv2.putText(canvas, cls_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 20

        # add extra text
        if extra_lines:
            for line in extra_lines:
                cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 20

        h, w = canvas.shape[:2]
        mode_position = int(0.85 * w)
        if self.horizontal_padding > 0:
            mode_position = int(0.85 * w)
        cv2.putText(canvas, f"Mode: {self.op_mode}", (mode_position, 20 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if self.op_mode == "add":
            y_offset = 40 + 100  # int(20 + 0.05 * h)
            cv2.putText(canvas, f"Add type: {self.add_type}", (mode_position, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

    def run(self, frame, show_image=True, draw_label=True, draw_centroid=True, centroid_text='coords', text_color=None, help_extra_lines=None):
        start_time = time.perf_counter()

        if not self.in_place and self.horizontal_padding == 0:
            canvas = frame.copy()
        else:
            canvas = frame

        # add padding
        h, w = canvas.shape[:2]
        pad = self.horizontal_padding
        if abs(pad) > 0:
            left, right = 0, 0
            if pad < 0:
                left = -pad
            elif pad > 0:
                right = pad
            canvas = cv2.copyMakeBorder(canvas, top=0, bottom=0, left=left, right=right, borderType=cv2.BORDER_CONSTANT,
                                        value=(50, 50, 50))  # copies the original image

        # if self.gpu:
        #     # example GPU conversion
        #     tensor = torch.from_numpy(canvas).to(device='cuda', dtype=torch.float32)
        #     # tensor = tensor / 255.0
        #     # tensor = tensor.permute(2, 0, 1)
        #     canvas = (tensor.cpu().numpy() * 255).astype(np.uint8)

        self.draw_objects(canvas,
                          draw_label=draw_label, draw_centroid=draw_centroid,
                          centroid_text=centroid_text, text_color=text_color)

        left_height_start = 0
        if self.show_help:
            left_height_start = draw_help(canvas)

        self.draw_extras(canvas, y=left_height_start, classes=self.classes, extra_lines=help_extra_lines)

        if show_image:
            cv2.imshow(self.window, canvas)
        self.last_canvas = canvas

        # check what key is pressed if any
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return key, False

        self.handle_key(key)

        if self.debug:
            run_duration = time.perf_counter() - start_time
            all_objects = [str(obj) for obj in self.objects]
            try:
                run_rate = 1 / run_duration
            except ZeroDivisionError:
                run_rate = 0
            print(f"Run rate: {run_rate}. All objects: {all_objects} \n current points: {self.current_pts} \n\n")
        return key, True

    def stream_and_run(self, source=None, show_image=True, draw_label=True, draw_centroid=True, centroid_text='coords', text_color=None,
                       loop=False, loop_delay=0, num_loops=-1):
        if source is not None:
            self.cap = cv2.VideoCapture(source)

        frame_number = 0  # could use "cap.get(cv2.CAP_PROP_POS_FRAMES)" instead to get the current frame number
        num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if loop:
            loop_number = 0

        start_time = time.time()
        # iterate over frames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            frame_number += 1
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
            key, continue_stream = self.run(frame,
                                            show_image=show_image,
                                            draw_label=draw_label, draw_centroid=draw_centroid,
                                            centroid_text=centroid_text, text_color=text_color)
            if not continue_stream:
                break

            if self.debug:
                run_duration = time.time() - start_time
                try:
                    run_rate = 1 / run_duration
                except ZeroDivisionError:
                    run_rate = 0
                print(f"Total run rate: {run_rate} \n \n")

        self.save_objects()
        self.cap.release()
        cv2.destroyAllWindows()

def parse_opt() -> argparse.Namespace:
    """Parse command line arguments for the region counting application.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Path to a video source, e.g video file, rtsp stream, etc.")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--classes", nargs="+", default=['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'],
                        type=str, help="filter by class: --classes person, or --classes person car bicycle")
    parser.add_argument("--regions-path", type=str, default="objects.json", help="The path to load the objects/regions.")
    parser.add_argument("--show-help", action="store_true",
                        help="Whether to show help message for the region counting application.")
    parser.add_argument("--default-op-mode", type=str, default="add", help="The default operation mode.",
                        choices=["add", "edit", "delete"])
    parser.add_argument("--default-add-type", type=str, default="polygon", help="The default add type.",
                        choices=["polygon", "point", "rect", "arrow", "line"])
    parser.add_argument("--regions-img-path", type=str, default="cv_gui_snapshot.png", help="The path to save a picture of the regions.")
    parser.add_argument("--window-name", type=str, default="ROI Counter", help="The name of the window.")
    parser.add_argument("--horizontal-padding", type=int, default=0, help="The width of the window.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    parser.add_argument("--show-img", action="store_true", help="show results")
    parser.add_argument("--draw-label", action="store_true", help="Whether to plot the object label.")
    parser.add_argument("--draw-centroid", action="store_true", help="Whether to plot the object centroid.")
    parser.add_argument("--centroid-text", type=str, default="coords", help="The text to display for the centroid.")
    parser.add_argument("--text-color", nargs="+", type=int, default=[0, 0, 255], help="The text color.")
    parser.add_argument("--loop-video", action="store_true", help="Loop the video.")
    parser.add_argument("--loop-video-delay", type=int, default=0, help="Delay between loops in seconds.")
    parser.add_argument("--loop-video-num-loops", type=int, default=-1, help="Number of times to loop. -1 for infinite.")

    return parser.parse_args()


def main(options: argparse.Namespace) -> None:
    """Execute the CV GUI with the provided options."""
    options_dict = vars(options)
    gui = CVGUI(source=options.source, in_place=False, save_file=options.regions_path, classes=options.classes,
                default_op_mode=options.default_op_mode, default_add_type=options.default_add_type,
                window_name=options.window_name, show_help=options.show_help,
                save_object_image=options.regions_img_path, horizontal_padding=options.horizontal_padding,
                debug=options.debug, gpu=True if 'cuda' in options.device else False)
    # gui.run()
    gui.stream_and_run(source=None, show_image=options.show_img,
                       draw_label=options.draw_label, draw_centroid=options.draw_centroid, centroid_text=options.centroid_text,
                       text_color=options.text_color,
                       loop=options.loop_video, loop_delay=options.loop_video_delay, num_loops=options.loop_video_num_loops)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
