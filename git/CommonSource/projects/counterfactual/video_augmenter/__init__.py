from rp import mean
import math
import numpy as np


class Quadrilateral:
    def __init__(self, x0, x1, x2, x3, y0, y1, y2, y3):
        self.x0, self.x1, self.x2, self.x3 = x0, x1, x2, x3
        self.y0, self.y1, self.y2, self.y3 = y0, y1, y2, y3

        self.points = np.array(
            [
                [x0, y0],
                [x1, y1],
                [x2, y2],
                [x3, y3],
            ]
        )

        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        self.min_x, self.min_y = min_coords
        self.max_x, self.max_y = max_coords

        self.center = np.mean(self.points, axis=0)
        self.mean_x, self.mean_y = self.center

        self.bounds = np.array([[self.min_x, self.max_x], [self.min_y, self.max_y]])
        self.ranges = self.bounds[:, 1] - self.bounds[:, 0]

    @classmethod
    def from_bounds(cls, bounds) -> "Quadrilateral":
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
        return cls(min_x, max_x, max_x, min_x, min_y, min_y, max_y, max_y)

    @classmethod
    def from_dimensions(cls, *, width, height):
        return cls.from_bounds([[0, width], [0, height]])

    @classmethod
    def from_points(cls, points) -> "Quadrilateral":
        x0, y0 = points[0]
        x1, y1 = points[1]
        x2, y2 = points[2]
        x3, y3 = points[3]
        return cls(x0, x1, x2, x3, y0, y1, y2, y3)

    def rotate(self, degrees, origin=None) -> "Quadrilateral":
        if origin is None:
            origin = self.center

        radians = math.radians(degrees)

        rotation_matrix = np.array(
            [
                [math.cos(radians), -math.sin(radians)],
                [math.sin(radians), math.cos(radians)],
            ]
        )

        origin = np.array(origin)

        centered_points = self.points - origin
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        new_points = rotated_points + origin

        return Quadrilateral.from_points(new_points)

    def rescale(self, factor, origin=None) -> "Quadrilateral":
        if origin is None:
            origin = self.center

        origin = np.array(origin)

        # Translate points to origin, apply scaling, translate back
        centered_points = self.points - origin
        scaled_points = centered_points * factor
        new_points = scaled_points + origin

        return Quadrilateral.from_points(new_points)

    def with_bounds(self, bounds) -> "Quadrilateral":
        bounds = np.asarray(bounds)
        min_target, max_target = bounds[:, 0], bounds[:, 1]
        min_current, max_current = self.bounds[:, 0], self.bounds[:, 1]

        scale = (max_target - min_target) / (max_current - min_current)
        centered = self.points - min_current
        scaled = centered * scale
        new_points = scaled + min_target

        return Quadrilateral.from_points(new_points)

    def fits_in_bounds(self, bounds) -> bool:
        bounds = np.asarray(bounds)
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]

        return (
            self.min_x >= min_x
            and self.max_x <= max_x
            and self.min_y >= min_y
            and self.max_y <= max_y
        )

    def with_center(self, center) -> "Quadrilateral":
        translation = np.array(center) - self.center
        new_points = self.points + translation

        return Quadrilateral.from_points(new_points)

    def __repr__(self):
        return f'Quadrilateral({str(self.points).replace("\n","")})'


def get_random_bounds(big_shape, small_shape):
    bounds = []
    for img_dim, crop_dim in zip(big_shape, small_shape):
        start = random_float(0, img_dim - crop_dim)
        end = start + crop_dim
        bounds.append([start, end])
    return np.asarray(bounds)


def get_random_quads(T=49, VH=480, VW=720):

    big_quad = Quadrilateral.from_dimensions(width=VW, height=VH)

    min_rot = -45
    max_rot = 45

    min_scale = 0.25

    random_scale = lambda: 1 - (1 - min_scale) * random_float() ** 2

    random_angle = lambda: random_float(min_rot, max_rot) * rp.random_float() ** 2

    while True:
        angle_a = random_angle()
        angle_b = random_angle()

        scale_a = random_scale()
        scale_b = random_scale()

        log_scale_a = np.log(scale_a)
        log_scale_b = np.log(scale_b)

        quad_a = big_quad.rotate(angle_a).rescale(scale_a)
        quad_b = big_quad.rotate(angle_b).rescale(scale_b)

        quad_a_center = quad_a.with_bounds(
            get_random_bounds([VW, VH], quad_a.ranges)
        ).center
        quad_b_center = quad_b.with_bounds(
            get_random_bounds([VW, VH], quad_b.ranges)
        ).center

        scales = np.exp(np.linspace(log_scale_a, log_scale_b, num=T))
        angles = np.linspace(angle_a, angle_b, num=T)
        centers = np.linspace(quad_a_center, quad_b_center, num=T)

        quads = [
            big_quad.rotate(angle).rescale(scale).with_center(center)
            for scale, angle, center in zip(scales, angles, centers)
        ]

        if all(quad.fits_in_bounds(big_quad.bounds) for quad in quads):
            return quads


def quads_to_image(quads, VH, VW):
    preview_image = cv_draw_contours(
        as_byte_image(uniform_float_color_image(VH, VW)),
        [quad.points for quad in quads],
    )
    return preview_image


def demo_get_random_quads():
    quads = get_random_quads()
    preview_image = quads_to_image(quads, 480, 720)
    display_image(preview_image)
    return preview_image


def augment_video(video, quads=None):
    T, VH, VW, C = video.shape
    if quads is None:
        quads = get_random_quads(T, VH, VW)

    assert len(quads) == len(video), (len(quads), len(video))

    frames = []
    for frame, quad in zip(video, quads):
        frame = unwarped_perspective_image(frame, quad.points)
        frames.append(frame)
    video = as_numpy_array(frames)

    return gather_vars("video quads")


def augment_videos(videos, quads=None):
    videos = np.stack(videos)
    B, T, VH, VW, C = videos.shape

    if quads is None:
        quads = get_random_quads(T, VH, VW)

    return np.stack([augment_video(video, quads) for video in videos])


def demo_augment_video():
    """See https://www.youtube.com/watch?v=Yi0MagvU86w"""
    T, VH, VW = 49, 480, 720
    old_video = resize_list(
        resize_images(
            load_video(
                download_to_cache(
                    # "https://videos.pexels.com/video-files/29081059/12567546_1920_1080_50fps.mp4"
                    "https://video-previews.elements.envatousercontent.com/77b2d7c5-a902-4389-afdd-5b8c9b6285fb/watermarked_preview/watermarked_preview.mp4"
                ),
                use_cache=True,
            ),
            size=(VH, VW),
        ),
        T,
    )

    augmentation = augment_video(old_video)
    new_video = augmentation.video
    quads = augmentation.quads

    old_video = [
        cv_draw_contour(frame, quad.points, color="blue")
        for frame, quad in zip(old_video, quads)
    ]

    arrow = load_image(
        "https://cdn-icons-png.flaticon.com/512/3031/3031716.png", use_cache=True
    )
    arrow = get_alpha_channel(arrow)
    arrow = resize_image_to_fit(arrow, width=VH // 3)
    arrow = crop_image(arrow, height=VH, origin="center")

    preview_video = horizontally_concatenated_videos(
        old_video, [arrow], new_video, [quads_to_image(quads, VH, VW)]
    )

    preview_video = video_with_progress_bar(
        preview_video, bar_color="green", position="bottom"
    )
    preview_video = labeled_images(
        preview_video, "Augmentation Preview", font="Futura", size=45
    )

    save_path = "demo_augment_video.mp4"
    save_path = get_unique_copy_path(save_path)

    save_video_mp4(preview_video, save_path, show_progress=False)
    fansi_print(f"Saved {save_path}", "bold green green italic on black black")

    display_video(
        preview_video,
        loop=False,
    )
