import os
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DampedPendulumSimulator:
    def __init__(self,
                 L=8.0,                # 摆长
                 theta0=0.5,           # 初始角度 (弧度)
                 omega0=0.0,           # 初始角速度
                 g=9.81,               # 重力
                 damping=0.8,          # 阻尼
                 bob_img_path="apple.png",
                 bob_zoom=0.25,
                 fixed_max_x=18,
                 fixed_max_y=12):

        self.L = L
        self.theta = theta0
        self.omega = omega0
        self.g = g
        self.damping = damping

        self.bob_img = plt.imread(bob_img_path)
        self.bob_zoom = bob_zoom
        self._set_fixed_axis(fixed_max_x, fixed_max_y)

        # 半径（像素大小控制 -> 转换成物理尺度）
        # 简单估算：假设100像素 ≈ 1个物理长度单位
        self.radius = (self.bob_img.shape[0] / 100.0) * self.bob_zoom / 2

    def _set_fixed_axis(self, max_x, max_y):
        self.x_lim = (0, max_x)
        self.y_lim = (0, max_y)

    def step(self, dt):
        """四阶Runge-Kutta 积分"""
        def deriv(theta, omega):
            dtheta = omega
            domega = -(self.g / self.L) * math.sin(theta) - self.damping * omega
            return dtheta, domega

        k1_theta, k1_omega = deriv(self.theta, self.omega)
        k2_theta, k2_omega = deriv(self.theta + 0.5 * dt * k1_theta, self.omega + 0.5 * dt * k1_omega)
        k3_theta, k3_omega = deriv(self.theta + 0.5 * dt * k2_theta, self.omega + 0.5 * dt * k2_omega)
        k4_theta, k4_omega = deriv(self.theta + dt * k3_theta, self.omega + dt * k3_omega)

        self.theta += (dt / 6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        self.omega += (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

    def get_position(self):
        """返回小球位置 (世界坐标)"""
        x = 9 + self.L * math.sin(self.theta)   # 挂点在 (9, 12)
        y = 12 - self.L * math.cos(self.theta)
        return x, y

    def get_velocity(self):
        """根据角速度求线速度 (vx, vy)"""
        vx = self.L * self.omega * math.cos(self.theta)
        vy = self.L * self.omega * math.sin(self.theta)
        return vx, vy

    def get_shape_properties(self):
        """计算 s, l, a"""
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
    base_output_dir = "unified_simulation_val/pendulum"
    os.makedirs(base_output_dir, exist_ok=True)

    n_samples = 10
    dt = 0.01
    test_times = [i * dt for i in range(101)]  # 0 到 1 秒，共 101 帧

    fig_w, fig_h = 2880, 1920  # 用于 px, py 转换

    for i in range(n_samples):
        # 随机初始角度 [-0.6, -0.2] 弧度
        theta0 = float(np.random.uniform(-0.6, -0.2))
        # 随机初始角速度 [0.0, 1.0] rad/s
        omega0 = float(np.random.uniform(0.0, 1.0))

        sim = DampedPendulumSimulator(
            L=8.0,
            theta0=theta0,
            omega0=omega0,
            damping=0.5,
            bob_img_path="E:/dataset/newton/ball.png",
            bob_zoom=0.2,
            fixed_max_x=18,
            fixed_max_y=12
        )

        # 初始位置 & 速度
        first_pos = sim.get_position()
        vx0, vy0 = sim.get_velocity()
        s0, l0, a0 = sim.get_shape_properties()

        # 转换到像素坐标
        px = int(first_pos[0] / sim.x_lim[1] * fig_w)
        py = int((sim.y_lim[1] - first_pos[1]) / sim.y_lim[1] * fig_h)

        # 文件名里加上 theta、omega、s、l、a
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
                    # 渲染
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

                    # 保存 txt
                    ftxt.write(f"{t:.4f}\t"
                               f"{x:.4f}\t{y:.4f}\t{vx:.4f}\t{vy:.4f}\t{sim.theta:.4f}\t{sim.omega:.4f}\t{s:.4f}\t{l:.4f}\t{a:.4f}\n")

                sim.step(dt)
                t += dt

        writer.close()
        print(f"Saved: {video_path}, {txt_path}")
