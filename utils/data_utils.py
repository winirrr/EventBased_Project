import os
import numpy as np
import dv_processing as dv
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from datetime import timedelta


# Convert to frame using dv library and slicing with certain interval
def dv_data_frame_tSlice(file_path, duration):
    capture = dv.io.MonoCameraRecording(file_path)
    frames = []
    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input camera does not provide an event stream.")

    # Initialize an accumulator with some resolution
    accumulator = dv.Accumulator(capture.getEventResolution())

    # Apply configuration, these values can be modified to taste
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(1.0)
    accumulator.setNeutralPotential(0.5)
    accumulator.setEventContribution(0.15)
    accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
    accumulator.setDecayParam(1e+6)
    accumulator.setIgnorePolarity(False)
    accumulator.setSynchronousDecay(False)

    slicer = dv.EventStreamSlicer()

    def slicing_callback(events: dv.EventStore):
        accumulator.accept(events)
        frame = accumulator.generateFrame()
        # frame = np.array(frame)
        # frame = frame.astype(np.uint8)
        frames.append(frame.image)

    slicer.doEveryTimeInterval(timedelta(milliseconds=duration), slicing_callback)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    return frames



# Convert to time surface using dv library and slicing with certain interval
def dv_time_surface_tSlice(file_path, duration):
    capture = dv.io.MonoCameraRecording(file_path)
    frames = []
    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input camera does not provide an event stream.")

    surface = dv.TimeSurface(capture.getEventResolution())

    slicer = dv.EventStreamSlicer()

    def slicing_callback(events: dv.EventStore):
        surface.accept(events)
        frame = surface.generateFrame()
        # frame = np.array(frame)
        # frame = frame.astype(np.uint8)
        frames.append(frame.image)

    slicer.doEveryTimeInterval(timedelta(milliseconds=duration), slicing_callback)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    return frames

# Extract method from scratch
def extract_data(events):
    ts, x, y, p = [], [], [], []
    for event in events:
        ts.append(event.timestamp())
        x.append(event.x())
        y.append(event.y())
        p.append(event.polarity)
    return ts, x, y, p



# Generate frame (not optimize)
def data_frame(video_file):
    capture = dv.io.MonoCameraRecording(video_file)
    slicer = dv.EventStreamSlicer()
    sliced_events = []

    def collect_events(events: dv.EventStore):
        sliced_events.append(events)


    slicer.doEveryNumberOfEvents(10000, collect_events)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    frames = []
    for events_batch in sliced_events:
        frame = np.zeros((640, 480), dtype=int)
        # event_data = extract_data(events_batch)
        for event in events_batch:
            ts, x, y, p = [event.timestamp(),
                           event.x(),
                           event.y(),
                           event.polarity()]
            if p:
                frame[x, y] = +1
            else:
                frame[x, y] = -1
        frames.append(frame)

    return frames


# Convert to frame using our method and slicing within certain duration.
def data_frame_tSlice(video_file, duration=33):
    capture = dv.io.MonoCameraRecording(video_file)
    slicer = dv.EventStreamSlicer()
    sliced_events = []

    def collect_events(events: dv.EventStore):
        sliced_events.append(events)


    slicer.doEveryTimeInterval(timedelta(milliseconds=duration), collect_events)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    frames = []
    for events_batch in sliced_events:
        frame = np.zeros((480, 640), dtype=int)
        # event_data = extract_data(events_batch)
        for event in events_batch:
            ts, x, y, p = [event.timestamp(),
                           event.x(),
                           event.y(),
                           event.polarity()]
            if p:
                frame[y, x] = +1
            else:
                frame[y, x] = -1
        frames.append(frame)

    return frames


# Generate time surface (not optimized)
def time_surface(file_path, num_events=10000, tau=50e3):
    capture = dv.io.MonoCameraRecording(file_path)
    slicer = dv.EventStreamSlicer()
    sliced_events = []

    def print_event_number(events: dv.EventStore):
        sliced_events.append(events)

    slicer.doEveryNumberOfEvents(num_events, print_event_number)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

    np_sliced_events = np.array(sliced_events)
    ts_frames= []
    for events in np_sliced_events:
        ts, x, y, p = extract_data(events)
        t_ref = ts[-1]  # 'current' time
        sae = np.zeros((640, 480), np.float32)

        for i in range(len(ts)):
            if p[i]:
                sae[x[i], y[i]] = np.exp(-(t_ref - ts[i]) / tau)
            else:
                sae[x[i], y[i]] = -np.exp(-(t_ref - ts[i]) / tau)

        ts_frames.append(sae)

    return ts_frames


# Convert to time surface and slicing within curtain time duration
def time_surface_tSlice(file_path, tau=50e3):
    capture = dv.io.MonoCameraRecording(file_path)
    slicer = dv.EventStreamSlicer()
    sliced_events = []

    def collect_events(events: dv.EventStore):
        sliced_events.append(events)

    slicer.doEveryTimeInterval(timedelta(milliseconds=33), collect_events)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)


    ts_frames= []
    for events in sliced_events:
        ts, x, y, p = extract_data(events)
        t_ref = ts[-1]  # 'current' time
        sae = np.zeros((480, 640), np.float32)

        for i in range(len(ts)):
            if p[i]:
                sae[y[i], x[i]] = np.exp(-(t_ref - ts[i]) / tau)
            else:
                sae[y[i], x[i]] = -np.exp(-(t_ref - ts[i]) / tau)

        ts_frames.append(sae)

    return ts_frames



# process raw event into representation and save 
def process_directory(input_dir, output_dir, representation, duration=None):
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith('.aedat4'):
                    file_path = os.path.join(label_dir, file)

                    frames = representation(file_path, duration)

                    label_output_dir = os.path.join(output_dir, label)
                    os.makedirs(label_output_dir, exist_ok=True)


                    for i, frame in enumerate(frames):
                        plt.imsave(os.path.join(label_output_dir, f"{os.path.splitext(file)[0]}_frame_{i}.png"), frame, cmap="gray")


# decode event
def extract_event_data(events):
    return np.array([[event.timestamp(),
                     event.x(),
                     event.y(),
                     event.polarity()] for event in events])