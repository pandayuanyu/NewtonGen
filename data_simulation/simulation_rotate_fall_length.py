import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio

class RotatingRodSimulator:
    def __init__(self,
                 omega=1.0,      # 角速度 rad/s
                 l0=3.0,         # 初始杆长
                 dl=1.0,         # 杆长增长速度 (单位长度/秒)
                 theta0=0.0,     # 初始角度
                 center0=(9, 6), # 初始中心位置
                 vx=0.0, vy0=0.0,# 中心点初速度
                 x_lim=(0, 18),
                 y_lim=(0, 12),
                 g=9.8,
                 fig_w=2880, fig_h=1920,
                 linewidth=24):
        self.omega = omega
        self.l0 = l0
        self.dl = dl
        self.theta0 = theta0
        self.center0 = np.array(center0, dtype=float)
        self.vx = vx
        self.vy0 = vy0
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.g = g

        # 图像参数
        self.fig_w = fig_w
        self.fig_h = fig_h
        self.linewidth = linewidth

        # 每像素代表的物理长度
        self.pixel_size_x = (x_lim[1] - x_lim[0]) / fig_w
        self.pixel_size_y = (y_lim[1] - y_lim[0]) / fig_h
        self.pixel_size = (self.pixel_size_x + self.pixel_size_y) / 2.0

        # 物理宽度（短边物理长度）
        self.physical_width = self.linewidth * self.pixel_size

    def _compute_theta(self, t):
        return self.theta0 + self.omega * t

    def _compute_length(self, t):
        return self.l0 + self.dl * t

    def _compute_center(self, t):
        cx = self.center0[0] + self.vx * t
        cy = self.center0[1] + self.vy0 * t - 0.5 * self.g * t**2
        cx = np.clip(cx, self.x_lim[0], self.x_lim[1])
        cy = np.clip(cy, self.y_lim[0], self.y_lim[1])
        return np.array([cx, cy])

    def _compute_velocity(self, t):
        vx = self.vx  # x 方向匀速
        vy = self.vy0 - self.g * t  # y 方向受重力加速度影响
        return vx, vy

    def _compute_endpoints(self, t):
        theta = self._compute_theta(t)
        l = self._compute_length(t)
        center = self._compute_center(t)
        dx = (l / 2) * math.cos(theta)
        dy = (l / 2) * math.sin(theta)
        x0 = center[0] - dx
        y0 = center[1] - dy
        x1 = center[0] + dx
        y1 = center[1] + dy
        return (x0, y0), (x1, y1)

    def render_frame(self, t):
        fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_aspect('equal')

        p0, p1 = self._compute_endpoints(t)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="blue", linewidth=self.linewidth)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, [1, 2, 3]]  # RGB
        plt.close(fig)
        return frame

    def compute_shape_properties(self, t):
        """短边 s、长边 l、面积 a"""
        l = self._compute_length(t)
        s = self.physical_width
        a = l * s
        return s, l, a


if __name__ == "__main__":
    test_times = [i * 0.01 for i in range(101)]
    base_output_dir = "unified_simulation_val/rotate_fall_length"
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10

    for i in range(n_samples):
        omega = float(np.random.uniform(0.0, 2.0))
        l0 = float(np.random.uniform(2.0, 3.0))
        dl = 1.0
        theta0 = math.pi / 4
        x0 = np.random.uniform(8, 10)
        y0 = np.random.uniform(8, 9)
        vx = np.random.uniform(-4.0, 4.0)
        vy0 = np.random.uniform(0.0, 1.0)

        sim = RotatingRodSimulator(omega=omega, l0=l0, dl=dl, theta0=theta0,
                                   center0=(x0, y0), vx=vx, vy0=vy0)

        # 初始中心点
        cx0, cy0 = sim.center0
        vx0, vy0 = sim._compute_velocity(0)
        s0, l0_phys, a0 = sim.compute_shape_properties(0)

        # 文件名
        px = int(cx0 / sim.x_lim[1] * sim.fig_w)
        py = int((sim.y_lim[1] - cy0) / sim.y_lim[1] * sim.fig_h)

        base_name = (
            f"px_{px}_py_{py}"
            f"_x_{cx0:.2f}_y_{cy0:.2f}"
            f"_vx_{vx0:.2f}_vy_{vy0:.2f}"
            f"_theta_{theta0:.2f}_omega_{omega:.2f}"
            f"_s_{s0:.2f}_l_{l0_phys:.2f}_a_{a0:.2f}"
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

                cx, cy = sim._compute_center(t)
                vx_t, vy_t = sim._compute_velocity(t)
                theta = sim._compute_theta(t)
                omega = sim.omega
                s, l_phys, a_phys = sim.compute_shape_properties(t)

                ftxt.write(
                    f"{t:.4f}\t{cx:.4f}\t{cy:.4f}\t"
                    f"{vx_t:.4f}\t{vy_t:.4f}\t"
                    f"{theta:.4f}\t{omega:.4f}\t"
                    f"{s:.4f}\t{l_phys:.4f}\t{a_phys:.4f}\n"
                )

        writer.close()
