import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Circle

class SpringSimulator:
    def __init__(self,
                 init_pos=(2,3),
                 init_vel=(0,0),
                 dt=0.01,
                 x_lim=(0,18),
                 y_lim=(0,12),
                 fig_w=2880,
                 fig_h=1920,
                 radius_phys=0.2):  # 物理单位的球半径
        self.pos = np.array(init_pos, dtype=float)
        self.vel = np.array(init_vel, dtype=float)
        self.dt = dt
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.fig_w = fig_w
        self.fig_h = fig_h
        self.radius_phys = radius_phys  # 物理半径

    def _compute_acceleration(self):
        # Hooke 弹簧力指向中心
        target = np.array([self.x_lim[1], self.y_lim[0]])
        delta = target - self.pos
        k = 1.5
        return k * delta

    def step(self):
        acc = self._compute_acceleration()
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt

    def render_frame(self):
        fig, ax = plt.subplots(figsize=(self.fig_w / 100, self.fig_h / 100), dpi=100)
        ax.set_xlim(*self.x_lim)
        ax.set_ylim(*self.y_lim)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # <-- 关键

        # 使用 Circle 绘制小球，中心在 self.pos
        circle = Circle(self.pos, radius=self.radius_phys, color='blue', zorder=10)
        ax.add_patch(circle)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[:, :, [1,2,3]]  # RGB
        plt.close(fig)
        return frame

    def compute_shape_properties(self):
        s = 2 * self.radius_phys
        l = 2 * self.radius_phys
        a = np.pi * self.radius_phys**2
        return s, l, a

if __name__ == "__main__":
    test_times = [i*0.01 for i in range(101)]
    base_output_dir = "unified_simulation_val/spring"
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10
    fig_w, fig_h = 2880, 1920

    for i in range(n_samples):
        x0 = np.random.uniform(1,9)
        y0 = np.random.uniform(1,11)
        vx0 = 0.0
        vy0 = 0.0
        print("x0",x0)

        sim = SpringSimulator(init_pos=(x0,y0), init_vel=(vx0,vy0), radius_phys=0.2)

        print('sim.pos[0]',sim.pos[0])
        # 第一帧物理坐标直接作为 px, py
        px = int(sim.pos[0] / sim.x_lim[1] * fig_w)
        py = int((sim.y_lim[1] - sim.pos[1]) / sim.y_lim[1] * fig_h)

        print('px',px)
        s0, l0, a0 = sim.compute_shape_properties()

        if not hasattr(sim, 'theta'):
            sim.theta = 0.0
        if not hasattr(sim, 'omega'):
            sim.omega = 0.0

        base_name = (
            f"px_{px}_py_{py}_x_{x0:.2f}_y_{y0:.2f}_vx_{vx0:.2f}_vy_{vy0:.2f}_"
            f"theta_{sim.theta:.2f}_omega_{sim.omega:.2f}_"
            f"s_{s0:.4f}_l_{l0:.4f}_a_{a0:.4f}"
        ).replace('.', '_')

        video_path = os.path.join(base_output_dir, f"{base_name}.mp4")
        txt_path = os.path.join(base_output_dir, f"{base_name}.txt")
        writer = imageio.get_writer(video_path, fps=25)
        with open(txt_path, 'w') as ftxt:
            ftxt.write("t\tx\ty\tvx\tvy\ttheta\tomega\ts\tl\ta\n")
            for t in test_times:
                frame = sim.render_frame()
                writer.append_data(frame)

                s, l, a = sim.compute_shape_properties()
                theta = 0.0
                omega = 0.0
                ftxt.write(f"{t:.4f}\t{sim.pos[0]:.4f}\t{sim.pos[1]:.4f}\t"
                           f"{sim.vel[0]:.4f}\t{sim.vel[1]:.4f}\t"
                           f"{theta:.4f}\t{omega:.4f}\t"
                           f"{s:.4f}\t{l:.4f}\t{a:.4f}\n")
                sim.step()


        writer.close()
        print(f"Saved {video_path} and {txt_path}")

