import os
import matplotlib.pyplot as plt
import numpy as np
import loguru as logger
from typing import Callable, Optional

OUTPUT_FOLDER = "/Users/test/Desktop/plot/"

line_colors = ["#8896AB", "#FFD5C2", "#C9D7F8", "#5C946E", "#4E5283"]
marker_types = ["h", "o", "^", "d", "D"]
font_size = 32

plt_graph_settings: dict[str, dict[str, float | int | str | tuple[float, float] | dict[str, float | int | str]]] = {
    "andy_theme": {
        "xtick.labelsize": 20,  # x軸刻度標籤大小
        "ytick.labelsize": 20,  # y軸刻度標籤大小
        "axes.labelsize": 20,  # 軸標籤大小
        "axes.titlesize": 20,  # 標題大小
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
def y_ticks_generator(y_data: list[list[float]]) -> list[float] | None:
    """! y_ticks_generator
    If you want to change the y ticks, you can change the y ticks here.
    Return array is the y ticks

    @param y_data: list[list[float]]: the y data
    @return list[float] | None: the y ticks, None means default setting
    """

    y_max = max([max(y) for y in y_data])
    y_min = min([min(y) for y in y_data])

    y_ticks_interval = max((y_max - y_min) / 4, y_min / 2)

    y_lower_bound = y_min if y_min != y_max else y_min - y_ticks_interval
    y_upper_bound = y_max + y_ticks_interval

    return np.arange(y_lower_bound, y_upper_bound, step=y_ticks_interval).tolist()


# 生成 x 軸刻度
def x_ticks_generator(x_data: list[float]) -> list[float] | None:
    """! x_ticks_generator
    If you want to change the x ticks, you can change the x ticks here.
    Return array is the x ticks

    @param x_data: list[float]: the x data
    @return list[float] | None: the x ticks, None means default setting
    """

    return None


# 改變 x, y 的資料
def data_preprocessing(data_x: list[float], data_y: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """! data_preprocessing
    If you want to change the data, you can change the data here.
    Be sure that the dimension of new_x and new_y is the same

    @param data_x: list[float]: the x data
    @param data_y: list[list[float]]: the y data
    @return tuple[list[float], list[list[float]]]: the new x, y data
    """

    new_data_y: list[list[float]] = []
    new_data_x: list[float] = []

    for i in range(len(data_y)):
        new_data_y.append([])

    all_unique_x: list[float] = []

    for i in range(len(data_y)):
        unique, counts = np.unique(data_y[i], return_counts=True)
        for j in range(len(unique)):
            if unique[j] not in all_unique_x:
                all_unique_x.append(float(unique[j]))

    # sort the unique x
    all_unique_x.sort()

    for i in range(len(data_y)):
        unique, counts = np.unique(data_y[i], return_counts=True)
        # print type of counts
        for j in range(len(all_unique_x)):
            # if the x is not in the data, append 0
            # else append the count of the x in the variable counts
            if all_unique_x[j] in data_y[i]:
                new_data_y[i].append(float(counts[np.where(unique == all_unique_x[j])[0][0]] / len(data_y[i])))
            else:
                new_data_y[i].append(0)

    print(new_data_y)

    new_data_x = all_unique_x

    return data_x, data_y


class Data:
    def __init__(
        self, filename: str, x_label: str, y_label: str, y_legends: list[str], output_filename: Optional[str] = None
    ):
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
        if output_filename is None:
            self.output_filename = os.path.splitext(os.path.basename(filename))[0]
        else:
            self.output_filename = output_filename
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
            logger.logger.warning(f"In file {filename}, the number of y_labels is not equal to the number of y values")

    def gen_graph(
        self, data_preprocessing: Callable[[list[float], list[list[float]]], tuple[list[float], list[list[float]]]]
    ):
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

        # ================================================================ set the y ticks

        y_ticks = y_ticks_generator(y_new)
        x_ticks = x_ticks_generator(x_new)

        # ================================================================ set the graph appearance

        ax1.tick_params(**plt_graph_settings["ax1_tick_settings"])
        ax1.xaxis.set_label_coords(**plt_graph_settings["ax1_xaxis_label_coords"])
        ax1.yaxis.set_label_coords(**plt_graph_settings["ax1_yaxis_label_coords"])

        plt.rcParams.update(plt_graph_settings["andy_theme"])
        plt.xlabel(xlabel=self.x_label, **plt_graph_settings["plt_xlabel_settings"])
        plt.ylabel(ylabel=self.y_label, **plt_graph_settings["plt_ylabel_settings"])
        plt.xticks(ticks=x_ticks, **plt_graph_settings["plt_xticks_settings"])
        plt.yticks(ticks=y_ticks, **plt_graph_settings["plt_yticks_settings"])
        plt.subplots_adjust(**plt_graph_settings["subplots_adjust_settings"])

        leg = plt.legend(self.y_legends, **plt_graph_settings["plt_leg_settings"])
        leg.get_frame().set_linewidth(plt_graph_settings["plt_leg_frame_settings"]["linewidth"])

        # ================================================================ save the graph
        plt.savefig(f"{OUTPUT_FOLDER}{self.output_filename}.png")
        plt.close()
        logger.logger.info(f"{self.filename} is saved as test.png")


def check_setting() -> bool:
    """! check_setting
    Check the setting of the plot generator

    @return bool: True if the setting is correct, False otherwise
    """

    return_value = True
    if len(line_colors) != len(marker_types):
        logger.logger.warning("The number of colors is not equal to the number of markers")
        return_value = False

    return return_value


def check_output_folder() -> bool:
    """! check_output_folder
    Check the output folder path is correct

    @return bool: True if the output folder path is correct, False otherwise
    """

    return_value = True
    try:
        if OUTPUT_FOLDER[-1] == "/":
            pass
    except IndexError:
        logger.logger.error("Output folder path should end with /")
        return_value = False

    return return_value


if __name__ == "__main__":
    if not check_setting():
        logger.logger.error("Setting check failed")
        exit(1)
    if not check_output_folder():
        logger.logger.error("Setting check failed")
        exit(1)
    else:
        logger.logger.info("Setting check passed")

    all_file_names: list[str] = [
        "/Users/test/Desktop/plot/input.txt",
    ]

    all_x_labels: list[str] = [
        "# Node",
    ]

    all_y_labels: list[str] = [
        "PSnode. num",
    ]

    all_y_legends: list[list[str]] = [
        ["PS", "GREEDY", "Q-CAST"],
    ]

    all_data: list[Data] = []

    for i in range(len(all_file_names)):
        all_data.append(
            Data(
                filename=all_file_names[i],
                x_label=all_x_labels[i],
                y_label=all_y_labels[i],
                y_legends=all_y_legends[i],
            )
        )

    for data in all_data:
        data.gen_graph(data_preprocessing)

    # data = Data(
    #     filename="/Users/test/Desktop/plot/input.txt",
    #     x_label="# Node",
    #     y_label="PSnode. num",
    #     y_legends=["PS", "GREEDY", "Q-CAST"],
    # )

    # data.gen_graph(data_preprocessing)
