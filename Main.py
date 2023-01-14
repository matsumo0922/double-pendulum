import dataclasses
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dataclasses import dataclass


@dataclass
class PendulumData:
    angle1: float
    angle2: float
    p1: float
    p2: float
    length: float
    weight: float


class Pendulum:
    def __init__(self, init_value: PendulumData, g, dt):
        """
        Pendulumを初期化する
        2つの振り子のパラメーターは共通となるので注意

        :param init_value: 2つの振り子のパラメーター
        :param g: 重力加速度
        :param dt: 1tickで進む時間
        """

        self.data = init_value
        self.g = g
        self.dt = dt
        self.locus = [self.get_descartes()]

    def get_descartes(self):
        """
        二つの振り子の現在の位置と角度をDescartes座標系に変換する

        :return: [始点, 振り子１の座標, 振り子2の座標]
        """

        x1 = self.data.length * np.sin(self.data.angle1)
        y1 = -self.data.length * np.cos(self.data.angle1)

        x2 = x1 + self.data.length * np.sin(self.data.angle2)
        y2 = y1 - self.data.length * np.cos(self.data.angle2)

        return np.array([[0.0, 0.0], [x1, y1], [x2, y2]])

    def get_next_position(self):
        """
        次の振り子の座標を計算する
        細かい計算はネットに乗っているものを参照した
        dataclasses.replaceを使いたかったが何故か動かなかったのでインスタンスを再代入する方式で妥協

        :return: 次の振り子の座標
        """

        tp1 = self.data.p1
        tp2 = self.data.p2

        expr1 = np.cos(self.data.angle1 - self.data.angle2)
        expr2 = np.sin(self.data.angle1 - self.data.angle2)
        expr3 = 1 + expr2 ** 2
        expr4 = tp1 * tp2 * expr2 / expr3
        expr5 = (tp1 ** 2 + 2 * tp2 ** 2 - tp1 * tp2 * expr1) * np.sin(2 * (self.data.angle1 - self.data.angle2)) / 2 / expr3 ** 2

        self.data = PendulumData(
            angle1=self.data.angle1 + self.dt * (tp1 - tp2 * expr1) / expr3,
            angle2=self.data.angle2 + self.dt * (2 * tp2 - tp1 * expr1) / expr3,
            p1=tp1 + self.dt * (-2 * self.g * self.data.length * np.sin(self.data.angle1) - (expr4 - expr5)),
            p2=tp2 + self.dt * (-self.g * self.data.length * np.sin(self.data.angle2) + (expr4 - expr5)),
            length=self.data.length,
            weight=self.data.weight
        )

        new_position = self.get_descartes()

        self.locus.append(new_position)
        return new_position


class Animator:
    def __init__(self, pendulum: Pendulum, is_draw_locus: bool):
        """
        Animatorを初期化する
        グラフのサイズは振り子の長さによって自動調節される
        zOrderを指定して、軌跡よりも振り子が最前に描画されるようになっている

        :param pendulum: 振り子のインスタンス
        :param is_draw_locus: 軌跡を描画するか
        """

        self.pendulum = pendulum
        self.is_draw_locus = is_draw_locus
        self.tick = 0.0

        self.figure, self.ax = plt.subplots()

        graph_size = (pendulum.data.length * 2) + 0.5
        self.ax.set_ylim(-graph_size, graph_size)
        self.ax.set_xlim(-graph_size, graph_size)

        self.time_str = self.ax.text(0.05, 0.9, "", transform=self.ax.transAxes)
        self.pendulums_plot, = self.ax.plot(self.pendulum.locus[-1][:, 0], self.pendulum.locus[-1][:, 1], zorder=1.0, marker='o')

        if self.is_draw_locus:
            self.locus_plot, = self.ax.plot(
                [a[2, 0] for a in self.pendulum.locus],
                [a[2, 1] for a in self.pendulum.locus],
                zorder=0.5,
            )

    def tick_loop(self):
        """
        1tick内で振り子の軌跡を計算する関数
        振り子の座標の計算自体はpendulum自身が行う
        """

        while True:
            self.tick += self.pendulum.dt
            yield self.pendulum.get_next_position()

    def update(self, data):
        """
        1tickごとに呼ばれ、グラフに再プロットする関数
        is_draw_locusがFalseの場合は軌跡が表示されない

        :param data: Matplotlibが管理するので必要なし
        :return: 振り子の座標のプロット
        """

        self.time_str.set_text("time: {:6.2f}".format(self.tick))

        if self.is_draw_locus:
            self.locus_plot.set_xdata([a[2, 0] for a in self.pendulum.locus])
            self.locus_plot.set_ydata([a[2, 1] for a in self.pendulum.locus])

        self.pendulums_plot.set_xdata(data[:, 0])
        self.pendulums_plot.set_ydata(data[:, 1])

        return self.pendulums_plot

    def animate(self, dt: float, save_sec: float = 0):
        """
        Matplotlib.Animationを利用して振り子をアニメーションする
        save_secで指定した時間が過ぎるまではアニメーションウィンドウが表示されないので注意

        :param dt: 1tickで進む時間
        :param save_sec: 動画内時間で何秒まで動画を保存するか
        """

        self.animation = animation.FuncAnimation(self.figure, self.update, self.tick_loop, interval=15, blit=False, save_count=int(save_sec / dt))
        self.animation.save("double-pendulum.mp4", writer="ffmpeg")


def main():
    """
    エントリーポイント
    各種データの初期値を設定し、Animatorに渡す
    """

    matplotlib.use('tkAgg')

    dt = 0.01
    data = PendulumData(
        angle1=np.pi / 2,
        angle2=np.pi / 2,
        p1=0.0,
        p2=0.0,
        length=1.0,
        weight=1.0,
    )

    pendulum = Pendulum(data, g=9.8, dt=dt)
    animator = Animator(pendulum=pendulum, is_draw_locus=True)

    animator.animate(dt=dt, save_sec=30)
    plt.show()


if __name__ == "__main__":
    main()
