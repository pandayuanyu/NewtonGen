import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

class SlidingBoxSimulator:
    def __init__(self,
                 box_size=(2.0, 1.0),
                 box_pos=(1, 8),
                 slope_start=(0, 6),
                 slope_end=(12, 0),
                 mu=0.1,
                 g=9.81,
                 v0=0.0,
                 x_lim=(0, 18),
                 y_lim=(0, 12),
                 fig_w=2880,
                 fig_h=1920):
        self.width, self.height = box_size
        self.pos = np.array([box_pos[0], box_pos[1]], dtype=float)
        self.slope_start = np.array(slope_start, dtype=float)
        self.slope_end = np.array(slope_end, dtype=float)
        self.mu = mu
        self.g = g
        self.v0 = v0
        self.fig_w = fig_w
        self.fig_h = fig_h
        self.x_lim = x_lim
        self.y_lim = y_lim

        dx = slope_end[0] - slope_start[0]
        dy = slope_end[1] - slope_start[1]
        self.slope_angle = math.atan2(dy, dx)
        self.slope_vector = np.array([dx, dy])
        self.slope_length = np.linalg.norm(self.slope_vector)
        self.slope_unit = self.slope_vector / self.slope_length

        self.a_s = self.g * (-dy / self.slope_length) - self.mu * self.g * math.cos(self.slope_angle)

        self.slope_perp_unit = np.array([-self.slope_unit[1], self.slope_unit[0]])

    def _compute_state(self, t):
        s = self.v0 * t + 0.5 * self.a_s * t**2
        pos = self.pos + s * self.slope_unit
        v_mag = self.v0 + self.a_s * t
        vx = v_mag * self.slope_unit[0]
        vy = v_mag * self.slope_unit[1]
        return pos[0], pos[1], vx, vy

    def render_frame(self, t):
        fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        ax.set_position([0,0,1,1])
        ax.axis('off')
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_aspect('equal')

        ax.plot([self.slope_start[0], self.slope_end[0]],
                [self.slope_start[1], self.slope_end[1]],
                color='lightgray', linewidth=10, solid_capstyle='round', zorder=0)

        x, y, vx, vy = self._compute_state(t)

        rect = Rectangle((-self.width/2, 0), self.width, self.height, color='blue')
        trans = Affine2D().rotate(self.slope_angle).translate(x, y) + ax.transData
        rect.set_transform(trans)
        ax.add_patch(rect)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, [1,2,3]]
        plt.close(fig)
        return frame

    def compute_shape_params(self):
        theta = self.slope_angle
        omega = 0.0
        s = self.height
        l = self.width
        a = s * l
        return theta, omega, s, l, a


if __name__ == "__main__":
    test_times = [i*0.01 for i in range(101)]
    base_output_dir = ""
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10
    fig_w, fig_h = 2880, 1920
    slope_start = (0, 6)
    slope_end = (12, 0)
    dx = slope_end[0] - slope_start[0]
    dy = slope_end[1] - slope_start[1]
    slope_k = dy / dx

    for i in range(n_samples):
        width, height = 0.8, 0.3

        x0 = np.random.uniform((slope_start[0] + width/2)+1, (slope_end[0] - width/2) -6)
        y0 = slope_start[1] + slope_k * (x0 - slope_start[0])

        mu = 0.1
        v0 = np.random.uniform(0.0, 4.0)

        sim = SlidingBoxSimulator(
            box_size=(width, height),
            box_pos=(x0, y0),
            slope_start=slope_start,
            slope_end=slope_end,
            mu=mu,
            v0=v0
        )

        x_center = x0 + (height/2) * sim.slope_perp_unit[0]
        y_center = y0 + (height/2) * sim.slope_perp_unit[1]

        px = int((x_center / sim.x_lim[1]) * fig_w)
        py = int(((sim.y_lim[1] - y_center) / sim.y_lim[1]) * fig_h)

        theta, omega, s_box, l_box, a_box = sim.compute_shape_params()

        vx0 = v0 * sim.slope_unit[0]
        vy0 = v0 * sim.slope_unit[1]

        base_name = (
            f"px_{px}_py_{py}_x_{x_center:.2f}_y_{y_center:.2f}_"
            f"vx_{vx0:.2f}_vy_{vy0:.2f}_"
            f"theta_{theta:.2f}_omega_{omega:.2f}_s_{s_box:.4f}_l_{l_box:.4f}_a_{a_box:.4f}"
        ).replace('.', '_')

        video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
        txt_path = os.path.join(base_output_dir, f"{base_name}.txt")

        print(f"Simulating {i+1}/{n_samples}: saved to {base_name}")

        writer = imageio.get_writer(video_path, fps=25)
        with open(txt_path,'w') as ftxt:
            ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")
            for t in test_times:
                frame = sim.render_frame(t)
                writer.append_data(frame)
                x, y, vx, vy = sim._compute_state(t)
                ftxt.write(
                    f"{t:.4f}\t{x:.4f}\t{y:.4f}\t"
                    f"{vx:.4f}\t{vy:.4f}\t{theta:.4f}\t{omega:.4f}\t"
                    f"{s_box:.4f}\t{l_box:.4f}\t{a_box:.4f}\n"
                )
        writer.close()
