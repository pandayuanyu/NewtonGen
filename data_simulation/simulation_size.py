# import os
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio
# from matplotlib.patches import Circle
#
# class BalloonSimulator:
#     def __init__(self,
#                  r0=0.5,          # 初始半径
#                  dV=1.0,          # 体积增长速率 (体积/秒)
#                  center=(9, 6),   # 气球中心位置
#                  x_lim=(0, 18),
#                  y_lim=(0, 12),
#                  fig_w=2880,
#                  fig_h=1920):
#         self.r0 = r0
#         self.center = np.array(center, dtype=float)
#         self.dV = dV
#         self.x_lim = x_lim
#         self.y_lim = y_lim
#         self.fig_w = fig_w
#         self.fig_h = fig_h
#         self.V0 = 4/3 * math.pi * (r0**3)
#
#     def _compute_radius(self, t):
#         Vt = self.V0 + self.dV * t
#         r = (3 * Vt / (4 * math.pi))**(1/3)
#         return r
#
#     def render_frame(self, t):
#         fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
#         ax.set_position([0, 0, 1, 1])
#         ax.axis('off')
#         ax.set_xlim(*self.x_lim)
#         ax.set_ylim(*self.y_lim)
#         ax.set_aspect('equal')
#
#
#         r = self._compute_radius(t)
#         circle = Circle(self.center, radius=r, color="red", alpha=0.6)
#         ax.add_patch(circle)
#
#         fig.canvas.draw()
#         w, h = fig.canvas.get_width_height()
#         buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
#         frame = buf[:, :, [1, 2, 3]]  # RGB
#         plt.close(fig)
#         return frame
#
#     def compute_shape_properties(self, t):
#         r = self._compute_radius(t)
#         V = 4/3 * math.pi * r**3
#         A = math.pi * r**2
#         return r, r, A  # s=l=r, a=面积
#
#
# if __name__ == "__main__":
#     test_times = [i * 0.01 for i in range(101)]
#     base_output_dir = "unified_simulation/size"
#     os.makedirs(base_output_dir, exist_ok=True)
#
#     n_samples = 100
#
#     for i in range(n_samples):
#         r0 = np.random.uniform(0.1, 0.6)
#         dV = 3.0
#         x0 = np.random.uniform(6, 12)
#         y0 = np.random.uniform(5, 7)
#
#         vx, vy = 0.0, 0.0
#         theta = 0.0
#         omega = 0.0
#
#         sim = BalloonSimulator(r0=r0, dV=dV, center=(x0, y0))
#         s0_init, l0_init, a0_init = sim.compute_shape_properties(0)
#
#         px = int(x0 / sim.x_lim[1] * sim.fig_w)
#         py = int((sim.y_lim[1] - y0) / sim.y_lim[1] * sim.fig_h)
#
#         base_name = (
#             f"px_{px}_py_{py}"
#             f"_x_{x0:.2f}_y_{y0:.2f}"
#             f"_vx_{vx:.2f}_vy_{vy:.2f}"
#             f"_theta_{theta:.2f}_omega_{omega:.2f}"
#             f"_s_{s0_init:.2f}_l_{l0_init:.2f}_a_{a0_init:.2f}"
#         ).replace('.', '_')
#
#         video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
#         txt_path = os.path.join(base_output_dir, f"{base_name}.txt")
#
#         print(f"Simulating {i+1}/{n_samples}: saved to {base_name}")
#
#         writer = imageio.get_writer(video_path, fps=25)
#         with open(txt_path, 'w') as ftxt:
#             ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")
#             for t in test_times:
#                 frame = sim.render_frame(t)
#                 writer.append_data(frame)
#
#                 s, l, a = sim.compute_shape_properties(t)
#                 ftxt.write(
#                     f"{t:.4f}\t{x0:.4f}\t{y0:.4f}\t"
#                     f"{vx:.4f}\t{vy:.4f}\t"
#                     f"{theta:.4f}\t{omega:.4f}\t"
#                     f"{s:.4f}\t{l:.4f}\t{a:.4f}\n"
#                 )
#
#         writer.close()

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle

class BalloonSimulator:
    def __init__(self,
                 r0=0.5,          # 初始半径
                 alpha=0.2,       # 半径线性衰减系数
                 beta=0.3,        # 半径线性增长偏置
                 center=(9, 6),
                 x_lim=(0, 18),
                 y_lim=(0, 12),
                 fig_w=2880,
                 fig_h=1920):
        self.r0 = r0
        self.alpha = alpha
        self.beta = beta
        self.center = np.array(center, dtype=float)
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.fig_w = fig_w
        self.fig_h = fig_h

    # 使用线性 ODE dr/dt = -alpha*r + beta，时间步积分
    def _compute_radius(self, t):
        # 解线性 ODE: r(t) = r0*exp(-alpha*t) + beta*(1 - exp(-alpha*t))/alpha
        if self.alpha == 0.0:
            r = self.r0 + self.beta * t
        else:
            r = self.r0 * np.exp(-self.alpha * t) + (self.beta / self.alpha) * (1 - np.exp(-self.alpha * t))
        return r

    def render_frame(self, t):
        fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_aspect('equal')

        r = self._compute_radius(t)
        circle = Circle(self.center, radius=r, color="red", alpha=0.6)
        ax.add_patch(circle)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, [1, 2, 3]]  # RGB
        plt.close(fig)
        return frame

    def compute_shape_properties(self, t):
        r = self._compute_radius(t)
        V = 4/3 * math.pi * r**3
        A = math.pi * r**2
        return r, r, A  # s=l=r, a=面积


if __name__ == "__main__":
    test_times = [i * 0.01 for i in range(101)]
    base_output_dir = "unified_simulation_val/size"
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10

    for i in range(n_samples):
        r0 = np.random.uniform(0.1, 0.6)
        alpha = 1.0  # dr/dt = -alpha*r + beta
        beta = 2.0
        x0 = np.random.uniform(6, 12)
        y0 = np.random.uniform(5, 7)

        vx, vy = 0.0, 0.0
        theta = 0.0
        omega = 0.0

        sim = BalloonSimulator(r0=r0, alpha=alpha, beta=beta, center=(x0, y0))
        s0_init, l0_init, a0_init = sim.compute_shape_properties(0)

        px = int(x0 / sim.x_lim[1] * sim.fig_w)
        py = int((sim.y_lim[1] - y0) / sim.y_lim[1] * sim.fig_h)

        base_name = (
            f"px_{px}_py_{py}"
            f"_x_{x0:.2f}_y_{y0:.2f}"
            f"_vx_{vx:.2f}_vy_{vy:.2f}"
            f"_theta_{theta:.2f}_omega_{omega:.2f}"
            f"_s_{s0_init:.2f}_l_{l0_init:.2f}_a_{a0_init:.2f}"
        ).replace('.', '_')

        video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
        txt_path = os.path.join(base_output_dir, f"{base_name}.txt")

        print(f"Simulating {i+1}/{n_samples}: saved to {base_name}")

        writer = imageio.get_writer(video_path, fps=25)
        with open(txt_path, 'w') as ftxt:
            ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")
            for t in test_times:
                frame = sim.render_frame(t)
                writer.append_data(frame)

                s, l, a = sim.compute_shape_properties(t)
                ftxt.write(
                    f"{t:.4f}\t{x0:.4f}\t{y0:.4f}\t"
                    f"{vx:.4f}\t{vy:.4f}\t"
                    f"{theta:.4f}\t{omega:.4f}\t"
                    f"{s:.4f}\t{l:.4f}\t{a:.4f}\n"
                )

        writer.close()

