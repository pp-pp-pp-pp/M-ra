import sys
import sounddevice as sd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPalette
import threading

# Define the key colors in order
KEY_COLORS = [
    (0, 0, 0),       # Black
    (0, 0, 255),     # Blue
    (0, 255, 255),   # Cyan
    (0, 255, 0),     # Green
    (255, 255, 0),   # Yellow
    (255, 0, 0),     # Red
    (255, 0, 255),   # Magenta
    (255, 255, 255)  # White
]

def normalize_sample(sample_value):
    """
    Normalize a 24-bit PCM sample to the range [0, 1].
    """
    max_val = 8388607
    min_val = -8388608
    normalized = (sample_value - min_val) / (max_val - min_val)
    return normalized

def sample_to_hex(normalized_sample):
    """
    Maps a normalized sample value in [0, 1] to a hex color.
    The gradient transitions through the KEY_COLORS list.
    """
    num_segments = len(KEY_COLORS) - 1
    segment_length = 1 / num_segments

    # Calculate segment index
    segment = int(normalized_sample / segment_length)
    
    # Handle edge case where normalized_sample == 1.0
    if segment >= num_segments:
        segment = num_segments - 1
        local_pos = 1.0
    else:
        local_pos = (normalized_sample - (segment * segment_length)) / segment_length

    # Debugging: Print segment and local_pos
    # print(f"Normalized Sample: {normalized_sample}, Segment: {segment}, Local Pos: {local_pos}")

    # Get the start and end colors of the segment
    start_color = KEY_COLORS[segment]
    end_color = KEY_COLORS[segment + 1]

    # Interpolate between the start and end colors
    r = int(start_color[0] + (end_color[0] - start_color[0]) * local_pos)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * local_pos)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * local_pos)

    # Ensure RGB values are within [0, 255]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f'#{r:02X}{g:02X}{b:02X}'

class AudioVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.latest_left_sample = 0
        self.latest_right_sample = 0
        self.lock = threading.Lock()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer_interval = 1000 / 60  # 60 updates per second
        self.stream = None
        self.is_running = False

    def init_ui(self):
        self.layout = QVBoxLayout()

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Start button
        self.start_button = QPushButton("Start Visualization")
        self.start_button.clicked.connect(self.start_visualization)
        buttons_layout.addWidget(self.start_button)

        # Stop button
        self.stop_button = QPushButton("Stop Visualization")
        self.stop_button.clicked.connect(self.stop_visualization)
        self.stop_button.setEnabled(False)
        buttons_layout.addWidget(self.stop_button)

        self.layout.addLayout(buttons_layout)

        # Labels for left and right colors
        colors_layout = QHBoxLayout()

        self.color_display_left = QLabel("Left Channel")
        self.color_display_left.setAutoFillBackground(True)
        self.color_display_left.setFixedSize(200, 200)
        self.update_color(self.color_display_left, "#000000")
        colors_layout.addWidget(self.color_display_left)

        self.color_display_right = QLabel("Right Channel")
        self.color_display_right.setAutoFillBackground(True)
        self.color_display_right.setFixedSize(200, 200)
        self.update_color(self.color_display_right, "#000000")
        colors_layout.addWidget(self.color_display_right)

        self.layout.addLayout(colors_layout)

        # Sample info label
        self.sample_info = QLabel("Sample Info: ")
        self.layout.addWidget(self.sample_info)

        self.setLayout(self.layout)
        self.setWindowTitle("Live Audio Sample Visualizer")
        self.resize(450, 300)
        self.show()

    def start_visualization(self):
        try:
            # Open the audio input stream
            self.stream = sd.InputStream(
                channels=2,  # Stereo
                samplerate=44100,  # Standard sample rate
                callback=self.audio_callback,
                blocksize=1024,  # Number of frames per block
                dtype='int32'
            )
            self.stream.start()
            self.is_running = True

            # Start the timer
            self.timer.start(int(self.timer_interval))

            # Update button states
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.sample_info.setText("Visualization started.")

            print("Visualization started.")

        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.sample_info.setText(f"Error starting audio stream: {e}")

    def stop_visualization(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_running = False

            # Stop the timer
            self.timer.stop()

            # Update button states
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            # Reset colors to black
            self.update_color(self.color_display_left, "#000000")
            self.update_color(self.color_display_right, "#000000")
            self.sample_info.setText("Visualization stopped.")

            print("Visualization stopped.")

        except Exception as e:
            print(f"Error stopping audio stream: {e}")
            self.sample_info.setText(f"Error stopping audio stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio Callback Status: {status}")

        # Extract the last sample from the incoming data
        with self.lock:
            self.latest_left_sample = indata[-1, 0]
            self.latest_right_sample = indata[-1, 1]

    def update_visualization(self):
        with self.lock:
            left = self.latest_left_sample
            right = self.latest_right_sample

        # Normalize samples
        left_normalized = normalize_sample(left)
        right_normalized = normalize_sample(right)

        # Clamp normalized samples to [0.0, 1.0]
        left_normalized = np.clip(left_normalized, 0.0, 1.0)
        right_normalized = np.clip(right_normalized, 0.0, 1.0)

        # Map normalized samples to hex colors using the custom gradient
        hex_left = sample_to_hex(left_normalized)
        hex_right = sample_to_hex(right_normalized)

        # Update the color displays
        self.update_color(self.color_display_left, hex_left)
        self.update_color(self.color_display_right, hex_right)

        # Update the sample info label with normalized values
        self.sample_info.setText(
            f"Sample Info: Left={left_normalized:.6f}, Hex Left={hex_left} | "
            f"Right={right_normalized:.6f}, Hex Right={hex_right}"
        )

    def update_color(self, label, hex_color):
        # Convert hex to QColor
        color = QColor(hex_color)
        palette = label.palette()
        palette.setColor(QPalette.Window, color)
        label.setPalette(palette)

    def closeEvent(self, event):
        # Ensure that the audio stream is closed when the application exits
        self.stop_visualization()
        event.accept()

def main():
    app = QApplication(sys.argv)
    visualizer = AudioVisualizer()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
