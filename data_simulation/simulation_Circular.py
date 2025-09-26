
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CircularMotionSimulator:
    def __init__(self,
                 L=8.0,
                 theta0=0.5,
                 omega0=0.0,
                 bob_img_path="apple.png",
                 bob_zoom=0.25,
                 fixed_max_x=18,
                 fixed_max_y=12):

        self.L = L
        self.theta = theta0
        self.omega = omega0

        self.bob_img = plt.imread(bob_img_path)
        self.bob_zoom = bob_zoom
        self._set_fixed_axis(fixed_max_x, fixed_max_y)


        self.radius = (self.bob_img.shape[0] / 100.0) * self.bob_zoom / 2

    def _set_fixed_axis(self, max_x, max_y):
        self.x_lim = (0, max_x)
        self.y_lim = (0, max_y)

    def step(self, dt):

        self.theta += self.omega * dt

    def get_position(self):

        x = 9 + self.L * math.sin(self.theta)
        y = 12 - self.L * math.cos(self.theta)
        return x, y

    def get_velocity(self):

        vx = self.L * self.omega * math.cos(self.theta)
        vy = self.L * self.omega * math.sin(self.theta)
        return vx, vy

    def get_shape_properties(self):

        s = 2 * self.radius
        l = 2 * self.radius
        a = math.pi * self.radius**2
        return s, l, a

    def _add_bob(self, ax, pos):
        H, W = self.bob_img.shape[0], self.bob_img.shape[1]
        phys_w = (W / 100.0) * self.bob_zoom
        phys_h = (H / 100.0) * self.bob_zoom
        x0 = pos[0] - phys_w / 2
        x1 = pos[0] + phys_w / 2
        y0 = pos[1] - phys_h / 2
        y1 = pos[1] + phys_h / 2
        ax.imshow(self.bob_img, extent=[x0, x1, y0, y1], aspect='auto', zorder=20, interpolation='bilinear')



if __name__ == "__main__":
    base_output_dir = ""
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10
    dt = 0.01
    test_times = [i * dt for i in range(101)]

    fig_w, fig_h = 2880, 1920

    for i in range(n_samples):
        theta0 = float(np.random.uniform(-0.6, -0.2))

        omega0 = float(np.random.uniform(0.0, 1.0))

        sim = CircularMotionSimulator(
            L=8.0,
            theta0=theta0,
            omega0=omega0,
            bob_img_path="",
            bob_zoom=0.2,
            fixed_max_x=18,
            fixed_max_y=12
        )


        first_pos = sim.get_position()
        vx0, vy0 = sim.get_velocity()
        s0, l0, a0 = sim.get_shape_properties()


        px = int(first_pos[0] / sim.x_lim[1] * fig_w)
        py = int((sim.y_lim[1] - first_pos[1]) / sim.y_lim[1] * fig_h)


        base_name = (
            f"px_{px}_py_{py}"
            f"_x_{first_pos[0]:.2f}_y_{first_pos[1]:.2f}"
            f"_vx_{vx0:.2f}_vy_{vy0:.2f}"
            f"_theta_{sim.theta:.2f}_omega_{sim.omega:.2f}"
            f"_s_{s0:.2f}_l_{l0:.2f}_a_{a0:.2f}"
        ).replace('.', '_')

        video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
        txt_path = os.path.join(base_output_dir, f"{base_name}.txt")

        writer = imageio.get_writer(video_path, fps=int(1/dt))

        with open(txt_path, 'w') as ftxt:
            ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")

            t = 0.0
            for step in range(int(test_times[-1]/dt)+1):
                x, y = sim.get_position()
                vx, vy = sim.get_velocity()
                s, l, a = sim.get_shape_properties()

                if abs(t - test_times[0]) < 1e-8 or any(abs(t - tt) < 1e-8 for tt in test_times):

                    fig, ax = plt.subplots(figsize=(28.8, 19.2), dpi=100)
                    ax.set_position([0, 0, 1, 1])
                    ax.axis('off')
                    ax.set_xlim(*sim.x_lim)
                    ax.set_ylim(*sim.y_lim)
                    ax.set_aspect('equal')

                    ax.plot([9, x], [12, y], color="lightgray", linewidth=1, zorder=5)
                    sim._add_bob(ax, (x, y))

                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
                    frame = buf[:, :, [1, 2, 3]]
                    writer.append_data(frame)
                    plt.close(fig)

                    ftxt.write(f"{t:.4f}\t"
                               f"{x:.4f}\t{y:.4f}\t{vx:.4f}\t{vy:.4f}\t{sim.theta:.4f}\t{sim.omega:.4f}\t{s:.4f}\t{l:.4f}\t{a:.4f}\n")

                sim.step(dt)
                t += dt

        writer.close()
        print(f"Saved: {video_path}, {txt_path}")
