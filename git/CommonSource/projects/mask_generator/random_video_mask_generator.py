"""
Ryan Burgert 2025
This module is meant to get random masks for video inpainting training.
"""

__all__ = ['get_random_video_mask', 'demo']

import numpy as np
import random
import cv2
import rp


class _ShapePlugin:
    """Base class for shape plugins."""

    def __init__(self, name):
        self.name = name
        self.hyperparameters = {}  # Initialize an empty dictionary for hyperparameters
        self.duration = 1  # Default duration is 1 frame
        self.remaining_duration = 0
        self.is_animated = False

    def draw(self, frame, frame_width, frame_height):
        """Draws the shape onto the frame. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the draw method.")

    def randomize_hyperparameters(self, frame_width, frame_height):
        """Randomizes the hyperparameters of the plugin."""
        pass  # Default implementation does nothing

    def update_hyperparameters(self, frame_width, frame_height):
        """Updates hyperparameters for animation.  Default does nothing."""
        if self.is_animated:
            self._animate_movement(frame_width, frame_height)

    def start_drawing(self, frame_width, frame_height):
        """Called when the plugin is selected to start drawing."""
        self.randomize_hyperparameters(frame_width, frame_height)
        # Determine duration:  Mostly 1 frame, but sometimes longer.
        if random.random() < 0.2:  # 20% chance of longer duration
            self.duration = random.randint(2, 5)  # Duration between 2 and 5 frames
        else:
            self.duration = 1
        self.remaining_duration = self.duration

        # Determine if animated:
        self.is_animated = random.random() < 1  # 30% chance of being animated

    def is_active(self):
        """Returns True if the plugin should still be drawn, False otherwise."""
        return self.remaining_duration > 0

    def decrement_duration(self):
        """Decrements the remaining duration."""
        self.remaining_duration -= 1

    def _animate_movement(self, frame_width, frame_height):
        """Handles the common movement logic for animated shapes."""
        for param, delta_param in self._get_animated_parameters():
            self.hyperparameters[param] += self.hyperparameters[delta_param]

            # Basic bouncing logic (can be overridden in subclasses if needed)
            lower_bound = 0
            upper_bound = frame_width if "x" in param else frame_height
            size_param = None
            if "width" in self.hyperparameters and param in ("center_x", "center_y"):
                size_param = "width" if "x" in param else "height"

            if size_param:
                lower_bound = int(self.hyperparameters[size_param] / 2)  # for center
                upper_bound -= int(self.hyperparameters[size_param] / 2)

            if (
                self.hyperparameters[param] < lower_bound
                or self.hyperparameters[param] > upper_bound
            ):
                self.hyperparameters[delta_param] *= -1
            if "width" in param or "height" in param:
                if (
                    self.hyperparameters[param] < 10
                    or self.hyperparameters[param] > frame_width
                ) and "width" in param:  # make the min width 10 and max width,  width
                    self.hyperparameters[delta_param] *= -1
                if (
                    self.hyperparameters[param] < 10
                    or self.hyperparameters[param] > frame_height
                ) and "height" in param:  # make the min height 10 and max height
                    self.hyperparameters[delta_param] *= -1

    def _get_animated_parameters(self):
        """
        Returns a list of (parameter, delta_parameter) tuples for animation.
        Must be implemented by subclasses that support animation.
        """
        return []  # Default: no animation parameters


class _RectanglePlugin(_ShapePlugin):
    """Plugin for drawing rectangles using OpenCV."""

    def __init__(self):
        super().__init__("rectangle")
        self.hyperparameters = {
            "center_x": 0,
            "center_y": 0,
            "width": 0,
            "height": 0,
            "delta_x": 0,
            "delta_y": 0,
            "delta_width": 0,
            "delta_height": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        x1 = max(
            0, int(self.hyperparameters["center_x"] - self.hyperparameters["width"] / 2)
        )
        y1 = max(
            0,
            int(self.hyperparameters["center_y"] - self.hyperparameters["height"] / 2),
        )
        x2 = min(
            frame_width,
            int(self.hyperparameters["center_x"] + self.hyperparameters["width"] / 2),
        )
        y2 = min(
            frame_height,
            int(self.hyperparameters["center_y"] + self.hyperparameters["height"] / 2),
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        self.hyperparameters["center_x"] = random.randint(0, frame_width - 1)
        self.hyperparameters["center_y"] = random.randint(0, frame_height - 1)
        self.hyperparameters["width"] = int(random.random() * frame_width)
        self.hyperparameters["height"] = int(random.random() * frame_height)

        if self.is_animated:
            # Random deltas for movement and size change (variable speed)
            self.hyperparameters["delta_x"] = random.randint(-8, 8)
            self.hyperparameters["delta_y"] = random.randint(-8, 8)
            self.hyperparameters["delta_width"] = random.randint(-8, 8)
            self.hyperparameters["delta_height"] = random.randint(-8, 8)
        else:
            for key in ["delta_x", "delta_y", "delta_width", "delta_height"]:
                self.hyperparameters[key] = 0

    def _get_animated_parameters(self):
        return [
            ("center_x", "delta_x"),
            ("center_y", "delta_y"),
            ("width", "delta_width"),
            ("height", "delta_height"),
        ]


class _EllipsePlugin(_ShapePlugin):
    """Plugin for drawing ellipses using OpenCV."""

    def __init__(self):
        super().__init__("ellipse")
        self.hyperparameters = {
            "center_x": 0,
            "center_y": 0,
            "width": 0,
            "height": 0,
            "angle": 0,
            "delta_x": 0,
            "delta_y": 0,
            "delta_width": 0,
            "delta_height": 0,
            "delta_angle": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        axes = (
            int(self.hyperparameters["width"] / 2),
            int(self.hyperparameters["height"] / 2),
        )
        cv2.ellipse(
            frame,
            (
                int(self.hyperparameters["center_x"]),
                int(self.hyperparameters["center_y"]),
            ),
            axes,
            int(self.hyperparameters["angle"]),
            0,
            360,
            (255, 255, 255),
            -1,
        )
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        self.hyperparameters["center_x"] = random.randint(0, frame_width - 1)
        self.hyperparameters["center_y"] = random.randint(0, frame_height - 1)
        self.hyperparameters["width"] = int(random.random() * frame_width)
        self.hyperparameters["height"] = int(random.random() * frame_height)
        self.hyperparameters["angle"] = random.randint(0, 359)

        if self.is_animated:
            # Random deltas
            self.hyperparameters["delta_x"] = random.randint(-8, 8)
            self.hyperparameters["delta_y"] = random.randint(-8, 8)
            self.hyperparameters["delta_width"] = random.randint(-8, 8)
            self.hyperparameters["delta_height"] = random.randint(-8, 8)
            self.hyperparameters["delta_angle"] = random.randint(-15, 15)
        else:
            for key in [
                "delta_x",
                "delta_y",
                "delta_width",
                "delta_height",
                "delta_angle",
            ]:
                self.hyperparameters[key] = 0

    def _get_animated_parameters(self):
        return [
            ("center_x", "delta_x"),
            ("center_y", "delta_y"),
            ("width", "delta_width"),
            ("height", "delta_height"),
            ("angle", "delta_angle"),
        ]

    def _animate_movement(self, frame_width, frame_height):
        super()._animate_movement(frame_width, frame_height)  # Handle basic movement
        self.hyperparameters["angle"] %= 360  # Keep angle within 0-359


class _ScribblePlugin(_ShapePlugin):
    """Plugin for drawing scribbles using OpenCV."""

    def __init__(
        self, num_points_range=(3, 13), thickness_range=(1, 10), max_points=50
    ):
        super().__init__("scribble")
        self.num_points_range = num_points_range
        self.thickness_range = thickness_range
        self.max_points = max_points
        self.is_animated = True  # Make _ScribblePlugin always animated by default
        self.hyperparameters = {
            "center_x": 0,
            "center_y": 0,
            "width": 0,
            "height": 0,
            "num_points": 0,
            "thickness": 0,
            "points": [],
            "delta_points": [],
        }

    def draw(self, frame, frame_width, frame_height):
        points = np.array(self.hyperparameters["points"], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [points],
            isClosed=False,
            color=(255, 255, 255),
            thickness=int(self.hyperparameters["thickness"]),
        )
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        self.hyperparameters["center_x"] = random.randint(0, frame_width - 1)
        self.hyperparameters["center_y"] = random.randint(0, frame_height - 1)
        self.hyperparameters["width"] = int(random.random() * frame_width)
        self.hyperparameters["height"] = int(random.random() * frame_height)
        self.hyperparameters["num_points"] = min(
            random.randint(self.num_points_range[0], self.num_points_range[1]),
            self.max_points,
        )
        self.hyperparameters["thickness"] = random.randint(
            self.thickness_range[0], self.thickness_range[1]
        )
        self.hyperparameters["points"] = []
        self.hyperparameters["delta_points"] = []
        for _ in range(self.hyperparameters["num_points"]):
            x = random.randint(
                max(
                    0, self.hyperparameters["center_x"] - self.hyperparameters["width"]
                ),
                min(
                    frame_width - 1,
                    self.hyperparameters["center_x"] + self.hyperparameters["width"],
                ),
            )
            y = random.randint(
                max(
                    0, self.hyperparameters["center_y"] - self.hyperparameters["height"]
                ),
                min(
                    frame_height - 1,
                    self.hyperparameters["center_y"] + self.hyperparameters["height"],
                ),
            )
            self.hyperparameters["points"].append((x, y))
            # Make deltas larger for more "dancing" movement
            self.hyperparameters["delta_points"].append(
                (
                    random.randint(-10, 10),
                    random.randint(-10, 10),
                )  # Increased delta range
            )

    def _get_animated_parameters(self):
        #  This is a bit different because we have a list of points.
        return []  # We'll handle it directly in _animate_movement

    def _animate_movement(self, frame_width, frame_height):
        if (
            not self.is_animated
        ):  # This check is technically redundant now as we set is_animated=True in init, but good to keep for clarity
            return
        updated_points = []
        for i, (x, y) in enumerate(self.hyperparameters["points"]):
            dx, dy = self.hyperparameters["delta_points"][i]
            new_x = x + dx
            new_y = y + dy

            if new_x < 0 or new_x >= frame_width:
                dx *= -1
            if new_y < 0 or new_y >= frame_height:
                dy *= -1

            self.hyperparameters["delta_points"][i] = (dx, dy)
            updated_points.append((new_x, new_y))
        self.hyperparameters["points"] = updated_points


class _FramePersistenceAction:
    def __init__(self, persistence_range=(1, 3)):
        self.persistence_range = persistence_range
        self.name = "FramePersistence"  # For plugin list

    def apply(self, video):
        num_frames = video.shape[0]
        persistence = random.randint(
            self.persistence_range[0], self.persistence_range[1]
        )
        frames_to_persist = sorted(
            random.sample(range(num_frames), random.randint(1, num_frames))
        )  # randomly select frames to persist
        for t in frames_to_persist:
            for i in range(1, persistence + 1):
                if t + i < num_frames:
                    video[t + i] = video[t]  # copy the frame
        return video

    def start_drawing(
        self, frame_width, frame_height
    ):  # Dummy methods for plugin list compatibility
        pass

    def is_active(self):
        return False

    def decrement_duration(self):
        pass

    def draw(self, frame, frame_width, frame_height):
        return frame

    def update_hyperparameters(self, frame_width, frame_height):
        pass


class _VideoGenerator:
    def __init__(
        self, T=13, H=60, W=90, N=2, plugins=None, actions=None
    ):  # Renamed filters to actions
        self.T = T
        self.H = H
        self.W = W
        self.N = N
        self.plugins = plugins or []
        self.actions = actions or []  # Use actions instead of filters
        self.active_plugins = []
        self.frame_fill_percentages = []

    def generate_video(self):
        video = np.zeros((self.T, self.H, self.W), dtype=np.uint8)
        overall_fill_percent = random.random() ** self.N
        self.frame_fill_percentages = [0.0] * self.T

        for t in range(self.T):
            if random.random() < 0.4:
                plugin = random.choice(self.plugins)
                plugin.start_drawing(self.W, self.H)
                self.active_plugins.append(plugin)

            for plugin in self.active_plugins:
                if plugin.is_active():
                    video[t] = plugin.draw(video[t], self.W, self.H)
                    plugin.update_hyperparameters(self.W, self.H)
                    plugin.decrement_duration()

            self.active_plugins = [p for p in self.active_plugins if p.is_active()]
            current_white_pixels = np.sum(video[t] == 255)
            self.frame_fill_percentages[t] = current_white_pixels / (self.H * self.W)

        # Apply actions after drawing shapes
        for action in self.actions:  # Apply actions
            video = action.apply(video)

        return video


class _SaltPlugin(_ShapePlugin):
    def __init__(self, amount=0.05):
        super().__init__("salt")
        self.amount = amount

    def draw(self, frame, frame_width, frame_height):
        num_pixels = frame_width * frame_height
        num_salt = int(num_pixels * self.amount)
        coords = [random.randint(0, num_pixels - 1) for _ in range(num_salt)]
        for coord in coords:
            y = coord // frame_width
            x = coord % frame_width
            frame[y, x] = 255  # White salt
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        # You could randomize the salt amount here if desired.
        pass

    def _get_animated_parameters(self):
        return []  # No animation parameters for Salt


class _TrianglePlugin(_ShapePlugin):
    """Plugin for drawing triangles using OpenCV."""

    def __init__(self):
        super().__init__("triangle")
        self.hyperparameters = {
            "center_x": 0,
            "center_y": 0,
            "size": 0,  # Size of the triangle (e.g., side length)
            "angle": 0,  # Rotation angle
            "delta_x": 0,
            "delta_y": 0,
            "delta_size": 0,
            "delta_angle": 0,
        }

    def draw(self, frame, frame_width, frame_height):
        size = self.hyperparameters["size"]
        center_x = self.hyperparameters["center_x"]
        center_y = self.hyperparameters["center_y"]
        angle_radians = np.radians(self.hyperparameters["angle"])

        # Calculate triangle vertices based on center, size, and rotation
        pt1 = [
            int(center_x + size * np.cos(angle_radians)),
            int(center_y + size * np.sin(angle_radians)),
        ]
        pt2 = [
            int(center_x + size * np.cos(angle_radians + 2 * np.pi / 3)),
            int(center_y + size * np.sin(angle_radians + 2 * np.pi / 3)),
        ]
        pt3 = [
            int(center_x + size * np.cos(angle_radians + 4 * np.pi / 3)),
            int(center_y + size * np.sin(angle_radians + 4 * np.pi / 3)),
        ]
        points = np.array([pt1, pt2, pt3], np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.fillPoly(frame, [points], (255, 255, 255))  # Fill the triangle
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        self.hyperparameters["center_x"] = random.randint(0, frame_width - 1)
        self.hyperparameters["center_y"] = random.randint(0, frame_height - 1)
        self.hyperparameters["size"] = int(
            random.random() * min(frame_width, frame_height) / 2
        )  # Size relative to frame
        self.hyperparameters["angle"] = random.randint(0, 359)

        if self.is_animated:
            self.hyperparameters["delta_x"] = random.randint(-8, 8)
            self.hyperparameters["delta_y"] = random.randint(-8, 8)
            self.hyperparameters["delta_size"] = random.randint(-5, 5)  # Size change
            self.hyperparameters["delta_angle"] = random.randint(-10, 10)  # Rotation
        else:
            for key in ["delta_x", "delta_y", "delta_size", "delta_angle"]:
                self.hyperparameters[key] = 0

    def _get_animated_parameters(self):
        return [
            ("center_x", "delta_x"),
            ("center_y", "delta_y"),
            ("size", "delta_size"),
            ("angle", "delta_angle"),
        ]

    def _animate_movement(self, frame_width, frame_height):
        super()._animate_movement(frame_width, frame_height)
        self.hyperparameters["angle"] %= 360  # Keep angle in 0-359 range
        self.hyperparameters["size"] = max(
            10, min(self.hyperparameters["size"], min(frame_width, frame_height) / 2)
        )  # Clamp size


class _BoopySaltPlugin(_ShapePlugin):
    def __init__(
        self, name="salt", num_dots_range=(0, 50), radius_range=(1, 5)
    ):  # Added ranges for dots
        super().__init__(name)
        self.num_dots_range = num_dots_range
        self.radius_range = radius_range
        self.hyperparameters = {
            "dots": [],  # List to store dot parameters
        }

    def draw(self, frame, frame_width, frame_height):
        for dot in self.hyperparameters["dots"]:
            center_x = int(dot["center_x"])
            center_y = int(dot["center_y"])
            radius = int(dot["radius"])
            if radius > 0:  # Ensure radius is valid
                cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        return frame

    def randomize_hyperparameters(self, frame_width, frame_height):
        num_dots = random.randint(self.num_dots_range[0], self.num_dots_range[1])
        self.hyperparameters["dots"] = []  # Initialize or clear dots list
        for _ in range(num_dots):
            dot_params = {
                "center_x": random.randint(0, frame_width - 1),
                "center_y": random.randint(0, frame_height - 1),
                "radius": random.randint(self.radius_range[0], self.radius_range[1]),
                "delta_x": 0,
                "delta_y": 0,
            }
            if self.is_animated:
                dot_params["delta_x"] = random.uniform(
                    -3, 3
                )  # Independent velocities, adjust range
                dot_params["delta_y"] = random.uniform(-3, 3)
            self.hyperparameters["dots"].append(dot_params)

    def _animate_movement(self, frame_width, frame_height):
        updated_dots = []
        for dot in self.hyperparameters["dots"]:
            dot["center_x"] += dot["delta_x"]
            dot["center_y"] += dot["delta_y"]

            # Bounce logic for each dot
            if (
                dot["center_x"] < dot["radius"]
                or dot["center_x"] > frame_width - 1 - dot["radius"]
            ):
                dot["delta_x"] *= -1
            if (
                dot["center_y"] < dot["radius"]
                or dot["center_y"] > frame_height - 1 - dot["radius"]
            ):
                dot["delta_y"] *= -1

            updated_dots.append(dot)  # Keep the updated dot
        self.hyperparameters["dots"] = updated_dots

    def _get_animated_parameters(self):
        return []  # Animation is handled directly for each dot


# Example of how to use it in `_random_shapes_video` and `get_random_video_mask`:


def _random_shapes_video(plugins, *, T, H, W, N):
    actions = [
        _FramePersistenceAction(),
        _FrameShiftAction(),
    ]

    # Randomly select a subset of plugins AND actions
    plugins = random.sample(plugins, random.randint(1, len(plugins)))
    actions = random.sample(actions, random.randint(0, len(actions)))
    video_generator = _VideoGenerator(T, H, W, N, plugins, actions)
    return video_generator.generate_video()


# --- Helper function for easing ---
def _ease_in_out_cubic(t):
    """Cubic ease-in-out function for smoother animation."""
    return (4 * t * t * t) if t < 0.5 else (1 - pow(-2 * t + 2, 3) / 2)


class _FrameShiftAction:
    def __init__(
        self, motion_proportion=0.2, max_rotation_angle=20, duration_range=(1, 5)
    ):
        """
        Args:
            motion_proportion (float): Proportion of frame dimensions for shift range.
            max_rotation_angle (int): Maximum rotation angle in degrees.
            duration_range (tuple): Range for duration of the shift effect in frames.
        """
        self.motion_proportion = motion_proportion
        self.max_rotation_angle = max_rotation_angle
        self.duration_range = duration_range
        self.name = "FrameShift"
        self.is_animated = True

    def apply(self, video):
        num_frames = video.shape[0]
        duration = random.randint(self.duration_range[0], self.duration_range[1])
        frames_to_shift = sorted(
            random.sample(range(num_frames), random.randint(0, num_frames // 2))
        )

        for t in frames_to_shift:
            # Generate start and end parameters
            rows, cols = video[0].shape
            shift_x_range = int(cols * self.motion_proportion)
            shift_y_range = int(rows * self.motion_proportion)

            start_shift_x = random.randint(-shift_x_range, shift_x_range)
            end_shift_x = random.randint(-shift_x_range, shift_x_range)
            start_shift_y = random.randint(-shift_y_range, shift_y_range)
            end_shift_y = random.randint(-shift_y_range, shift_y_range)
            start_angle = random.randint(
                -self.max_rotation_angle, self.max_rotation_angle
            )
            end_angle = random.randint(
                -self.max_rotation_angle, self.max_rotation_angle
            )

            for i in range(duration):
                frame_index = t + i
                if frame_index < num_frames:
                    frame = video[frame_index]
                    rows, cols = frame.shape

                    # Use ease-in-out cubic interpolation for smoother motion
                    interpolation_factor = i / (duration - 1) if duration > 1 else 0
                    ease_factor = _ease_in_out_cubic(
                        interpolation_factor
                    )  # Apply easing function
                    current_shift_x = start_shift_x + ease_factor * (
                        end_shift_x - start_shift_x
                    )
                    current_shift_y = start_shift_y + ease_factor * (
                        end_shift_y - start_shift_y
                    )
                    current_angle = start_angle + ease_factor * (
                        end_angle - start_angle
                    )

                    M_translation = np.float32(
                        [[1, 0, current_shift_x], [0, 1, current_shift_y]]
                    )
                    M_rotation = cv2.getRotationMatrix2D(
                        (self.get_frame_center(frame)), current_angle, 1
                    )

                    # Apply Transformation
                    shifted_frame = cv2.warpAffine(
                        frame,
                        M_translation,
                        (cols, rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    rotated_shifted_frame = cv2.warpAffine(
                        shifted_frame,
                        M_rotation,
                        (cols, rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    video[frame_index] = rotated_shifted_frame
        return video

    def get_frame_center(self, frame):
        rows, cols = frame.shape
        return (cols / 2, rows / 2)

    def start_drawing(self, frame_width, frame_height):
        pass

    def is_active(self):
        return False

    def decrement_duration(self):
        pass

    def draw(self, frame, frame_width, frame_height):
        return frame

    def update_hyperparameters(self, frame_width, frame_height):
        pass


def get_random_video_mask(
    T=25,
    H=60,
    W=90,
):
    """
    Generates random video mask with given num frames, height and width. 
    Returns a boolean numpy video in THW form.
    """
    attempts=0
    while True:
        attempts+=1
        assert attempts<100, 'get_random_video_mask reached 100 attempts. Is it broken?'
        #If we error...keep going. Try again.
        try:
            N = 4  # Higher value --> more sparse
            plugins = [
                _RectanglePlugin(),
                _EllipsePlugin(),
                _ScribblePlugin(),
                _ScribblePlugin(),
                _ScribblePlugin(),
                _BoopySaltPlugin(),  # Use the modified _SaltPlugin
                *[_SaltPlugin()]
                * rp.random_chance(1 / 3),  # You can have multiple SaltPlugins now
                _TrianglePlugin(),
            ]

            def randvid():
                return _random_shapes_video(plugins=plugins, T=T, H=H, W=W, N=N)

            video = randvid() // 255
            plugins = rp.random_batch(plugins, rp.random_int(1, len(plugins)))
            thresh = rp.random_float() ** 4
            for _ in range(rp.random_int(2, 10)):
                try:
                    if rp.random_chance(0.1):
                        video |= randvid()
                    newvideo = randvid() // 255
                    if rp.random_chance():
                        newvideo = 1 - (newvideo)
                    newvideo = randvid() * newvideo
                    video |= newvideo
                    # print(video.mean() / 255)
                    if video.mean() / 255 > thresh:
                        break
                except Exception as e:
                    pass

            full_frame_indices = (
                [0] * rp.random_chance(0.1)
                + [T - 1] * rp.random_chance(0.1)
                + rp.random_batch(range(T), rp.random_int(0, 5) * rp.random_chance(0.1))
            )

            if rp.random_chance(1 / 10):
                full_frame_indices += list(range(rp.random_int(T)))

            for x in full_frame_indices:
                video[x] = 255

            video = rp.as_grayscale_images(rp.as_binary_images(video))
            video = rp.as_numpy_array(video)

            if rp.random_chance(1 / 10):
                video = ~video

            return video

        except Exception:
            rp.print_stack_trace()


def demo(output_folder="random_video_masks_demo"):
    """Demonstrates the get_random_video_mask() function"""
    print("PWD: " + rp.fansi_highlight_path(rp.get_current_directory()))
    for i in range(15):
        video = get_random_video_mask()
        rp.display_video(video)
        image = rp.tiled_images(video, border_color="red")
        path = rp.save_image(
            image, rp.get_unique_copy_path(rp.path_join(output_folder, "random_masks.jpg"))
        )
        print("    " + rp.fansi_highlight_path(path))

if __name__ == '__main__':
    demo()