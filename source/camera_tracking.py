import argparse
import gstreamer
import os
import time
import mecademicpy.robot as mdr
from multiprocessing import Process, Queue

from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

MAX_QUEUE_SIZE = 2  # Set the maximum size of the queue

# Initialize robot connection
def control_robot(coord_queue):
    robot = mdr.Robot()
    robot.Connect(address='192.168.0.100', enable_synchronous_mode=True, disconnect_on_exception=False)
    robot.ActivateAndHome()
    robot.SetJointVel(100)

    while True:
        if not coord_queue.empty():
            x, y = coord_queue.get()
            # Apply interpolation equations
            joint1_angle = -0.0797 * x + 31
            joint3_angle = 0.06604 * (480 - y) - 19.5
            robot.MoveJoints(joint1_angle, 0, joint3_angle, 0, 0, 0)
            print(f"Moving robot to Joint1: {joint1_angle}°, Joint3: {joint3_angle}°")

def generate_svg(src_size, inference_box, objs, labels, text_lines, coord_queue, inference_counter):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        if labels.get(obj.id) == 'cell phone':  # Only show cell phone
            bbox = obj.bbox
            if not bbox.valid:
                continue
            # Absolute coordinates, input tensor space.
            x, y = bbox.xmin, bbox.ymin
            w, h = bbox.width, bbox.height
            # Subtract boxing offset.
            x, y = x - box_x, y - box_y
            # Scale to source coordinate space.
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            percent = int(100 * obj.score)
            label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
            svg.add_text(x, y - 5, label, 20)
            svg.add_rect(x, y, w, h, 'red', 2)
            # Calculate center coordinates
            center_x = x + w / 2
            center_y = y + h / 2
            center_label = '({:.1f}, {:.1f})'.format(center_x, center_y)
            svg.add_text(center_x, center_y, center_label, 20)
            print(f'Cell phone detected at ({center_x}, {center_y})')
            # Only add to queue after every 5 inferences
            if inference_counter % 1 == 0:
                if coord_queue.qsize() >= MAX_QUEUE_SIZE:
                    coord_queue.get()  # Remove the oldest entry if the queue is full
                coord_queue.put((center_x, center_y))
    
    return svg.finish()

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video3')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    coord_queue = Queue()
    robot_process = Process(target=control_robot, args=(coord_queue,))
    robot_process.start()

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    inference_counter = 0

    def user_callback(input_tensor, src_size, inference_box):
        nonlocal fps_counter, inference_counter
        start_time = time.monotonic()
        run_inference(interpreter, input_tensor)
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        objs = [obj for obj in objs if labels.get(obj.id) == 'cell phone']  # Filter for cell phones
        end_time = time.monotonic()
        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))),
        ]
        print(' '.join(text_lines))
        inference_counter += 1
        svg = generate_svg(src_size, inference_box, objs, labels, text_lines, coord_queue, inference_counter)
        return svg

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)

    robot_process.terminate()

if _name_ == '_main_':
    main()