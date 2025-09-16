import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

class ShapeSimulator:
    def __init__(self,
                 l0=3.0,         # 初始长边
                 dl=0.0,         # 长边增长速度
                 s0=0.5,         # 初始短边
                 ds=0.0,         # 短边增长速度
                 theta0=0.0,     # 初始角度，固定
                 omega=0.0,      # 角速度，固定
                 center0=(9, 6), # 矩形中心固定位置
                 vx=0.0, vy=0.0, # 中心速度
                 x_lim=(0, 18),
                 y_lim=(0, 12),
                 fig_w=2880,
                 fig_h=1920):
        self.l0 = l0
        self.dl = dl
        self.s0 = s0
        self.ds = ds
        self.theta0 = theta0
        self.omega = omega
        self.center0 = np.array(center0, dtype=float)
        self.vx = vx
        self.vy = vy
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.fig_w = fig_w
        self.fig_h = fig_h

    # 固定角度
    def _compute_theta(self, t):
        return self.theta0

    # 长边长度
    def _compute_length(self, t):
        return self.l0 + self.dl * t

    # 短边长度
    def _compute_width(self, t):
        return self.s0 + self.ds * t

    # 中心固定
    def _compute_center(self, t):
        return self.center0.copy()

    # 渲染矩形
    def render_frame(self, t):
        fig, ax = plt.subplots(figsize=(self.fig_w/100, self.fig_h/100), dpi=100)
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_aspect('equal')

        center = self._compute_center(t)
        l = self._compute_length(t)
        s = self._compute_width(t)
        theta = self._compute_theta(t)

        rect = Rectangle((-l/2, -s/2), width=l, height=s, color='blue')
        trans = Affine2D().rotate_deg(np.degrees(theta)).translate(center[0], center[1]) + ax.transData
        rect.set_transform(trans)

        ax.add_patch(rect)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, [1, 2, 3]]  # RGB
        plt.close(fig)
        return frame

    # 物理属性
    def compute_shape_properties(self, t):
        l = self._compute_length(t)
        s = self._compute_width(t)
        a = l * s
        return s, l, a

if __name__ == "__main__":
    test_times = [i * 0.01 for i in range(101)]
    base_output_dir = "unified_simulation_val/shape"
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10

    for i in range(n_samples):
        l0 = np.random.uniform(5.0, 8.0)
        dl = -1.0
        s0 = np.random.uniform(0.5, 1.0)
        ds = 2.0
        theta0 = np.random.uniform(0, np.pi)
        omega = 0.0
        x0, y0 = 9.0, 6.0
        vx, vy = 0.0, 0.0

        sim = ShapeSimulator(
            l0=l0, dl=dl,
            s0=s0, ds=ds,
            theta0=theta0,
            omega=omega,
            center0=(x0, y0),
            vx=vx, vy=vy
        )

        cx0, cy0 = sim.center0
        s0_init, l0_init, a0_init = sim.compute_shape_properties(0)

        px = int(cx0 / sim.x_lim[1] * sim.fig_w)
        py = int((sim.y_lim[1] - cy0) / sim.y_lim[1] * sim.fig_h)


        base_name = (
            f"px_{px}_py_{py}"
            f"x_{cx0:.2f}_y_{cy0:.2f}"
            f"_vx_{vx:.2f}_vy_{vy:.2f}"
            f"_theta_{theta0:.2f}_omega_{omega:.2f}"
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

                cx, cy = sim._compute_center(t)
                vx_t, vy_t = sim.vx, sim.vy
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
