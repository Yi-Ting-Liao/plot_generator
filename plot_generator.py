import matplotlib.pyplot as plt
import numpy as np
import loguru as logger
from typing import Callable

OUTPUT_FOLDER = "/Users/test/Desktop/plot/"

line_colors = ["#8896AB", "#FFD5C2", "#C9D7F8", "#5C946E", "#4E5283"]
marker_types = ["h", "o", "^", "d", "D"]
font_size = 32

plt_graph_settings = {
    "andy_theme": {
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        # "font.family": "Times New Roman",
        # "mathtext.it": "Times New Roman:italic",
        # "mathtext.fontset": "custom",
    },
    "ax1_settings": {
        "lw": 2.5,  # 線寬
        "linestyle": "-",  # 線型
        "markersize": 12,  # 標記大小
        "markeredgewidth": 2.5,  # 標記邊緣寬度
    },
    "ax1_tick_settings": {
        "direction": "in",  # 刻度方向
        "bottom": True,  # 顯示底部刻度
        "top": True,  # 顯示頂部刻度
        "left": True,  # 顯示左側刻度
        "right": True,  # 顯示右側刻度
        "pad": 20,  # 刻度與標籤之間的距離
    },
    "ax1_xaxis_label_coords": {
        "x": 0.45,  # x軸標籤位置
        "y": -0.27,  # y軸標籤位置
    },
    "ax1_yaxis_label_coords": {
        "x": -0.3,  # x軸標籤位置
        "y": 0.5,  # y軸標籤位置
    },
    "plt_xlabel_settings": {
        "fontsize": 32,  # 字體大小
        "labelpad": 35,  # 標籤與軸之間的距離
    },
    "plt_ylabel_settings": {
        "fontsize": 32,
        "labelpad": 10,
    },
    "plt_xticks_settings": {
        "fontsize": 32,
    },
    "plt_yticks_settings": {
        "fontsize": 32,
    },
    "plt_leg_settings": {
        "loc": 10,  # 位置，10表示center
        "bbox_to_anchor": (
            0.4,
            1.25,
        ),  # bbox的位置， (0.4, 1.25)表示bbox的左下角在(0.4, 1.25)的位置
        "prop": {"size": 32},
        "frameon": "False",  # 是否顯示邊框
        "labelspacing": 0.2,  # 標籤間距
        "handletextpad": 0.2,  # 標籤與標記之間的距離
        "handlelength": 1,  # 標記長度
        "columnspacing": 0.2,  # 欄間距
        "ncol": 2,  # 欄數
        "facecolor": "None",  # 背景色
    },
    "plt_leg_frame_settings": {
        "linewidth": 0.0,  # legends 的邊框寬度
    },
    "subplots_setting": {
        "figsize": (7, 6),  # 圖表大小
        "dpi": 600,  # 圖表解析度
    },
    "subplots_adjust_settings": {
        "top": 0.75,  # 調整圖表上邊距，單位為圖表高度的比例
        "left": 0.3,  # 調整圖表左邊距，單位為圖表寬度的比例
        "right": 0.95,  # 調整圖表右邊距，單位為圖表寬度的比例
        "bottom": 0.25,  # 調整圖表下邊距，單位為圖表高度的比例
    },
}


# 生成 y 軸刻度
def y_ticks_generator(
    y_data: list[list[float]],
) -> list[float]:
    y_max = max([max(y) for y in y_data])
    y_min = min([min(y) for y in y_data])

    y_ticks_interval = max((y_max - y_min) / 4, y_min / 2)

    return np.arange(y_min, y_max + y_ticks_interval, step=y_ticks_interval)


# 改變 x, y 的資料
def data_preprocessing(data_x: list[float], data_y: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    # set new_data[i] t data[i] plus one
    new_data_y: list[list[float]] = []
    new_data_x: list[float] = []

    for i in range(len(data_y)):
        new_data_y.append([])

    unique, counts = np.unique(data_y[0], return_counts=True)

    for i in range(len(data_y)):
        for j in range(len(unique)):
            new_data_y[i].append(float(counts[j]) / len(data_y[i]))

    new_data_x = unique

    return new_data_x, new_data_y


class Data:
    def __init__(self, filename: str, x_label: str, y_label: str, y_legends: list[str]):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
            logger.logger.info(f"File {filename} is loaded, start initializing data")
        except FileNotFoundError:
            logger.logger.error(f"File {filename} cannot be found")
            return
        except PermissionError:
            logger.logger.error(f"File {filename} cannot be accessed")
            return

        self.x_label = x_label
        self.y_label = y_label
        self.y_legends = y_legends
        self.filename = filename
        self.x_count = 0
        self.y_count = 0
        self.x: list[float] = []
        self.y: list[list[float]] = []

        # set correct length for y
        for _ in range(len(lines[0].split()) - 1):
            self.y.append([])
            self.y_count += 1

        # read data from file
        for line in lines[0:]:
            data = line.split()  # spill data by space
            self.x.append(float(data[0]))  # the first data is x
            for i in range(1, len(data)):  # the rest of data is y
                self.y[i - 1].append(float(data[i]))
            self.x_count += 1

        if self.y_count != len(self.y_legends):
            logger.logger.error(f"In file {filename}, the number of y_labels is not equal to the number of y values")
            return

    def gen_graph(self, data_preprocessing: Callable[[list[float]], list[list[float]]]):
        if self.y_count > len(line_colors):
            logger.logger.warning("The number of y values is more than the number of colors")

        # ================================================================ plot the graph

        x_new, y_new = data_preprocessing(self.x, self.y)

        _, ax1 = plt.subplots(**plt_graph_settings["subplots_setting"])

        for i in range(len(y_new)):
            ax1.plot(
                x_new,
                y_new[i],
                color=line_colors[i],
                marker=marker_types[i],
                **plt_graph_settings["ax1_settings"],
            )  # 繪製折線圖 (x軸資料, y軸資料, 參數)

        # ================================================================

        y_ticks = y_ticks_generator(y_new)

        # ================================================================ set the graph appearance

        ax1.tick_params(**plt_graph_settings["ax1_tick_settings"])
        ax1.xaxis.set_label_coords(**plt_graph_settings["ax1_xaxis_label_coords"])
        ax1.yaxis.set_label_coords(**plt_graph_settings["ax1_yaxis_label_coords"])

        plt.rcParams.update(plt_graph_settings["andy_theme"])
        plt.xlabel(self.x_label, **plt_graph_settings["plt_xlabel_settings"])
        plt.ylabel(self.y_label, **plt_graph_settings["plt_ylabel_settings"])
        plt.xticks(**plt_graph_settings["plt_xticks_settings"])
        plt.yticks(y_ticks, **plt_graph_settings["plt_yticks_settings"])
        plt.subplots_adjust(**plt_graph_settings["subplots_adjust_settings"])

        leg = plt.legend(self.y_legends, **plt_graph_settings["plt_leg_settings"])
        leg.get_frame().set_linewidth(plt_graph_settings["plt_leg_frame_settings"]["linewidth"])

        # ================================================================ save the graph
        plt.savefig(f"{OUTPUT_FOLDER}test.png")
        plt.close()
        logger.logger.info(f"{self.filename} is saved as test.png")


def check_setting() -> bool:
    return_value = True
    if len(line_colors) != len(marker_types):
        logger.logger.warning("The number of colors is not equal to the number of markers")
        return_value = False

    return return_value


if __name__ == "__main__":
    if not check_setting():
        logger.logger.error("Setting check failed")
        exit(1)
    else:
        logger.logger.info("Setting check passed")

    data = Data(
        "/Users/test/Desktop/plot/input.txt",
        "# Node",
        "PSnode. num",
        ["PS"],
    )
    data.gen_graph(data_preprocessing)
