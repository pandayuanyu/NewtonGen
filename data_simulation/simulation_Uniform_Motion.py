import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ParabolaSimulator:
    def __init__(self,
                 newton_pos=(0, 0),
                 fruit_pos=(0, 10),
                 vx=1.0,
                 ax=1.0,
                 newton_img_path='newton.png',
                 fruit_img_path='apple.png',
                 newton_zoom=0.25,
                 fruit_zoom=0.25,
                 fixed_max_x=18,
                 fixed_max_y=12,
                 min_y=0.2):
        self.newton_pos = newton_pos
        self.x0, self.y0 = fruit_pos
        self.vx = vx
        self.ax = ax
        self.min_y = min_y

        self.newton_img = plt.imread(newton_img_path)
        self.fruit_img = plt.imread(fruit_img_path)
        self.newton_zoom = newton_zoom
        self.fruit_zoom = fruit_zoom

        self._set_fixed_axis(fixed_max_x, fixed_max_y)

        self.fruit_half_height = (self.fruit_img.shape[0] * self.fruit_zoom) / 80.0

        self.t_stop = 10.0

    def _set_fixed_axis(self, max_x, max_y):
        self.x_lim = (0, max_x)
        self.y_lim = (0, max_y)

    def _compute_position(self, t):
        if t <= 0:
            return (self.x0, self.y0)
        if t >= self.t_stop:
            x = self.x0 + self.vx * self.t_stop + 0.5 * self.ax * self.t_stop**2
            y = self.y0
            return (x, y)
        x = self.x0 + self.vx * t + 0.5 * self.ax * t**2
        y = self.y0
        return (x, y)

    def _add_image(self, ax, img, pos, zoom):
        H, W = img.shape[0], img.shape[1]
        phys_w = (W / 100.0) * zoom
        phys_h = (H / 100.0) * zoom
        x0 = pos[0] - phys_w / 2
        x1 = pos[0] + phys_w / 2
        y0 = pos[1] - phys_h / 2
        y1 = pos[1] + phys_h / 2
        ax.imshow(img, extent=[x0, x1, y0, y1], aspect='auto',
                  zorder=20, interpolation='bilinear')

    def _compute_shape_params(self):
        H, W = self.fruit_img.shape[0], self.fruit_img.shape[1]
        phys_w = (W / 100.0) * self.fruit_zoom
        phys_h = (H / 100.0) * self.fruit_zoom
        s = min(phys_w, phys_h)
        l = max(phys_w, phys_h)
        a = math.pi * (s / 2.0) * (l / 2.0)
        theta = 0.0
        omega = 0.0
        return theta, omega, s, l, a


if __name__ == "__main__":
    test_times = [i * 0.01 for i in range(101)]
    base_output_dir = ""
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10

    fig_w, fig_h = 2880, 1920

    for i in range(n_samples):
        x0 = float(np.random.uniform(0.5, 8.0))
        y0 = float(np.random.uniform(2.0, 10.0))
        vx = float(np.random.uniform(0.0, 8.0))
        ax = 0.0

        print(f"Simulating {i+1}/{n_samples}: vx={vx:.3f}, ax={ax:.3f}, x0={x0:.3f}, y0={y0:.3f}")

        sim = ParabolaSimulator(
            newton_pos=(10, 1),
            fruit_pos=(x0, y0),
            vx=vx,
            ax=ax,
            newton_img_path='',
            fruit_img_path='',
            newton_zoom=0.1,
            fruit_zoom=0.15,
            fixed_max_x=18,
            fixed_max_y=12,
            min_y=0.2
        )

        first_pos = sim._compute_position(0)
        px = int(first_pos[0] / sim.x_lim[1] * fig_w)
        py = int((sim.y_lim[1] - first_pos[1]) / sim.y_lim[1] * fig_h)

        theta0, omega0, s0, l0, a0 = sim._compute_shape_params()

        base_name = (
            f"px_{px}_py_{py}_x_{x0:.2f}_y_{y0:.2f}_vx_{vx:.2f}_"
            f"theta_{theta0:.2f}_omega_{omega0:.2f}_s_{s0:.4f}_l_{l0:.4f}_a_{a0:.4f}"
        ).replace('.', '_')

        video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
        txt_path = os.path.join(base_output_dir, f"{base_name}.txt")

        writer = imageio.get_writer(video_path, fps=25)
        with open(txt_path, 'w') as ftxt:
            ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")

            for t in test_times:
                fig, axplt = plt.subplots(figsize=(28.8, 19.2), dpi=100)
                axplt.set_position([0, 0, 1, 1])
                axplt.axis('off')
                axplt.set_xlim(*sim.x_lim)
                axplt.set_ylim(*sim.y_lim)
                axplt.set_aspect('equal')

                ground_y = 0.18
                axplt.axhline(ground_y, color='#689919', linewidth=10, zorder=10)

                sim._add_image(axplt, sim.newton_img, sim.newton_pos, sim.newton_zoom)
                fruit_pos = sim._compute_position(t)
                sim._add_image(axplt, sim.fruit_img, fruit_pos, sim.fruit_zoom)

                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                frame = buf[:, :, [1, 2, 3]]
                writer.append_data(frame)
                plt.close(fig)

                vx_t = sim.vx + sim.ax * t if t < sim.t_stop else 0.0
                vy_t = 0.0
                theta, omega, s, l, a = sim._compute_shape_params()
                ftxt.write(
                    f"{t:.4f}\t{fruit_pos[0]:.4f}\t{fruit_pos[1]:.4f}\t"
                    f"{vx_t:.4f}\t{vy_t:.4f}\t{theta:.4f}\t{omega:.4f}\t{s:.4f}\t{l:.4f}\t{a:.4f}\n"
                )

        writer.close()
        print(f"Saved: {video_path}, {txt_path}")
