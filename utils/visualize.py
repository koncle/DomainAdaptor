import PIL.Image
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea

__all__ = ['show_graphs', 'show_scatter', 'bar_plot', 'ImageOverlay', 'tSNE', 'get_twin_fig',
           'line_plot', 'LightColors', 'regression_plot', 'OringalColors', 'LighterColors']

from matplotlib.ticker import FormatStrFormatter

"""
  2D Image im_show in one graph 
"""

COLORS = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
marker_types = ['.', 'v', 's', '*', 'p', 'H', 'X', '1', '8', ]

OringalColors = ['#5A9BD5', '#1B9E78', '#FF9966', '#ff585d', '#614ad3', '#feda77']
LightColors = ["#f89588", "#7cd6cf", "#7898e1", "#efa666", "#9987ce", "#63b2ee", "#76da91", "#63b2ee", "#f8cb7f", ]
LighterColors = ['#929fff', '#9de0ff', '#ffa897', '#af87fe', '#7dc3fe', '#bb60b2', '#f47a75', '#009db2', '#0780cf']  # ,


def show_graphs(imgs, titles=None, invert=False, show=True, filename=None, figsize=(5, 5), bbox=[], colors=[], show_type='gray'):
    """  Show images in a grid manner. it will automatically get a almost squared grid to show these images.

    :param imgs: input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param titles: [str, ...], the title for every image
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    col = int(np.ceil(np.sqrt(len(imgs))))
    show_graph_with_col(imgs, max_cols=col, titles=titles, invert=invert, show=show, filename=filename, figsize=figsize, bbox=bbox,
                        colors=colors, show_type=show_type)


def show_graph_with_col(imgs, max_cols, titles=None, invert=False, show=True, filename=None, figsize=(5, 5),
                        bbox=[], colors=[], show_type='gray'):
    """ Show images in a grid manner.

    :param imgs: assume shape with [N, C, D, H, W], [N, C, H, W], [C, H, W], [N, H, W], [H, W]
             input images which dim ranges in (4, 3, 2), but only the first image (HxW) can be showed
    :param max_cols: int, max column of grid.
    :param titles: [str, ...], the title for every image
    :param show:  True or False, show or save image
    :param filename: str, if save image, specify the path
    :param figsize:  specify the output figure size
    :param bbox:  a list of ((min_x, max_x), (min_y, max_y))
    :param colors: a list of string of colors which length is the same as bbox
    """
    """
    Check size and type
    """
    if len(imgs) == 0:
        return

    length = len(imgs)
    if length < max_cols:
        max_cols = length

    # img = imgs[0]
    new_imgs = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            shape = img.shape
        elif isinstance(img, torch.Tensor):
            shape = img.size()
        elif isinstance(img, PIL.Image.Image):
            img = np.array(img).transpose(2, 0, 1)
            shape = img.shape
        else:
            raise Exception("Unknown type of imgs : {}".format(type(imgs)))
        assert 2 <= len(shape) <= 5, 'Error shape : {}'.format(shape)
        new_imgs.append(img)
    imgs = new_imgs

    """
    Plot graph
    """
    fig = plt.figure(figsize=figsize)
    max_line = int(np.ceil(length / max_cols))
    for i in range(1, length + 1):
        ax = fig.add_subplot(max_line, max_cols, i)
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if titles is not None:
            ax.set_title(titles[i - 1])

        img = imgs[i - 1]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        img = img.copy()
        img[img == -1] = 0
        color = False
        shape = img.shape
        if len(shape) == 5:
            # maybe colored image
            if shape[1] == 3:
                color = True
                img = img[0, :, 0, :, :]
            else:
                img = img[0, 0, 0, :, :]
        if len(shape) == 4:
            if shape[1] == 3:
                color = True
                img = img[0]
            else:
                img = img[0, 0]
        elif len(shape) == 3:
            if shape[0] == 3:
                color = True
            else:
                img = img[0]

        if color:
            # normalized image
            if img.min() < 0 or invert:
                img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                        np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255).astype(np.int32)
            img = img.transpose((1, 2, 0)).astype(np.int32)
            ax.imshow(img)
        else:
            if show_type == 'gray' or show_type == 'hot' or show_type is None:
                ax.imshow(img, cmap=show_type)
            elif show_type[:4] == 'hot_':
                vmin = int(show_type[4])
                vmax = int(show_type[5])
                ax.imshow(img, cmap=show_type[:3], vmin=vmin, vmax=vmax)
            else:
                ax.imshow(img, cmap=show_type)

        for i, box in enumerate(bbox):
            (min_x, max_x), (min_y, max_y) = box
            if len(colors) == len(bbox):
                color = colors[i]
            else:
                color = COLORS[i % len(COLORS)]
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(rect)

    # plt.subplots_adjust(wspace=0, hspace=0)
    color_map = False
    if color_map:
        import matplotlib as mpl
        ax = fig.add_axes([0.905, 0.1, 0.005, 0.8])
        cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', cmap='hsv')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


class ImageOverlay(object):
    def __init__(self, img, cmap='gray'):
        assert isinstance(img, np.ndarray), len(img.shape) in [2, 3]

        if len(img.shape) == 3:
            img = ((img * np.array([.229, .224, .225]).reshape(3, 1, 1) +
                    np.array([.485, .456, .406]).reshape(3, 1, 1)) * 255)
            img = img.transpose(1, 2, 0).astype(np.int32)

        fig = plt.figure(frameon=False, figsize=(5, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(img, cmap=cmap)

        self.fig = fig
        self.ax = ax

    def overlay(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        assert isinstance(mask, np.ndarray), len(mask.shape) == 2

        import cv2
        from matplotlib.patches import Polygon

        mask = (mask > 0).astype(np.uint8)
        # _, contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contour:
            self.ax.add_patch(
                Polygon(
                    c.reshape((-1, 2)),
                    fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
                )
            )
        return self

    def overlay_hole(self, mask, color=(1., 0., 0.), edgecolor='r', fill=False, linewidth=2.0, alpha=0.5):
        import cv2
        from matplotlib.path import Path

        mask = self.to_numpy(mask)
        mask = (mask > 0).astype(np.uint8)
        contour, hier = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        path_points = []
        path_move = []
        for c in contour:
            c = c.reshape(-1, 2)
            for i, p in enumerate(c):
                path_points.append(p)
                if i == 0:
                    path_move.append(Path.MOVETO)
                elif i == len(c) - 1:
                    path_move.append(Path.CLOSEPOLY)
                else:
                    path_move.append(Path.LINETO)
        from matplotlib.patches import PathPatch
        patch = PathPatch(Path(path_points, path_move), fill=fill, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
        self.ax.add_patch(patch)
        return self

    def show(self):
        plt.show()
        return self

    def save(self, save_path):
        self.fig.savefig(save_path)
        return self


def show_scatter(tsne_X, label, marker_size=2, marker_type='o', imgs=None, ax=None, texts=None):
    # imgs : 3 x H x W
    # label : N
    # tsne_X : N x 2
    if ax is None:
        ax = plt.gca()

    label = np.array(label)
    ret = None
    if imgs is None:
        # colors = np.array(sns.color_palette("husl", label.max()+1))
        # plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=plt.cm.Set1(label), s=marker_size, marker=marker_type)

        color_num = np.unique(label).max() + 1
        if color_num > 8:
            print('ranbow colors')
            cm = plt.get_cmap('gist_rainbow')
            colors = np.array([cm(1. * i / color_num) for i in range(color_num)])[label]
        else:
            label[label == 5] = 7  # 5太黄了
            colors = plt.cm.Set1(label)
        ret = plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=colors, s=marker_size, marker=marker_type)
    else:
        imgs = imgs.swapaxes(1, 2).swapaxes(2, 3)  # to channel last
        for i, (img, (x0, y0)) in enumerate(zip(imgs, tsne_X)):
            img = ((img * np.array([.229, .224, .225]).reshape(1, 1, 3) + np.array([.485, .456, .406]).reshape(1, 1, 3)) * 255).astype(np.int32)
            img = OffsetImage(img, zoom=0.2)  # 224*0.2 =
            ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
            ax.add_artist(ab)
            if texts is not None:
                offsetbox = TextArea(str(texts[i]), textprops=dict(alpha=1.0, size="smaller"))
                ab = AnnotationBbox(offsetbox, (x0, y0),
                                    xybox=(0, -27),
                                    xycoords='data',
                                    boxcoords="offset points",
                                    # arrowprops=dict(arrowstyle="->")
                                    )
                ax.add_artist(ab)
    return ret


class tSNE():
    @staticmethod
    def get_tsne_result(X, metric='euclidean', perplexity=30):
        """  Get 2D t-SNE result with sklearn

        :param X: feature with size of N x C
        :param metric: 'cosine', 'euclidean', and so on.
        :param perplexity:  the preserved local structure size
        """
        try:
            from sklearn.manifold.t_sne import TSNE
        except Exception as e:
            from sklearn.manifold._t_sne import TSNE
        tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity)
        tsne_X = tsne.fit_transform(X)
        tsne_X = (tsne_X - tsne_X.min()) / (tsne_X.max() - tsne_X.min())
        return tsne_X

    @staticmethod
    def plot_tsne(tsne_X, labels, domain_labels=None, imgs=None, texts=None, save_name=None, figsize=(10, 10), marker_size=20, label_name=None, legend=True):
        """ plot t-SNE results. All parameters are numpy format.

        Args:
            tsne_X: N x 2
            labels: N
            domain_labels: N
            imgs: N x 3 x H x W
            save_name: str
            figsize: tuple of figure size
            marker_size: size of markers
        """
        plt.figure(figsize=figsize)
        scatters = []
        if domain_labels is not None:
            # plot each domain with different shape of markers
            domains = np.unique(domain_labels)
            for d in domains:
                idx = domain_labels == d
                x_tmp = imgs[idx] if imgs is not None else None
                text_tmp = texts[idx] if texts is not None else None
                scatter = show_scatter(tsne_X[idx], labels[idx], marker_size=marker_size, marker_type=marker_types[d],
                                       imgs=x_tmp, texts=text_tmp)
                scatters.append(scatter)
        else:
            # plot simple clusters of classes with different colors
            show_scatter(tsne_X, labels, marker_size=marker_size, marker_type=marker_types[0], imgs=imgs, texts=texts)

        # plot legend
        if legend:
            each_labels = np.unique(labels)
            legend_elements = []
            for l in each_labels:
                if label_name is not None:
                    L = label_name[l]
                else:
                    L = str(l)
                if l == 5:
                    l = 7
                legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', label=L, markerfacecolor=plt.cm.Set1(l), markersize=10))
            legend2 = plt.legend(handles=legend_elements, loc='upper left')
            plt.gca().add_artist(legend2)

        if len(scatters) > 0:
            legend_elements = []
            domain_names = ['Photo', 'Art', 'Sketch', 'Cartoon']
            sizes = [20, 10, 10, 15]
            for i, (d, s) in enumerate(zip(domain_names, sizes)):
                legend_elements.append(mlines.Line2D([0], [0], marker=marker_types[i], color='w', label=d,
                                                     markerfacecolor=plt.cm.Set1([0]), markersize=s))
            legend1 = plt.legend(handles=legend_elements, loc='upper right')
            plt.gca().add_artist(legend1)

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        if save_name is not None:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()


def set_common_params(x_label=None, y_label=None, title=None, ylim=None, title_size=None,
                      x_axis=None, xticks=None, yticks=None, label_size=None, tick_size=None,
                      legend=True, legenc_loc=None, legend_size=None, ytick_format_n=None,
                      save_name=None, x_axis_name_rotate=None, grid=False, show=True,
                      legend_bbox_to_anchor=None, legend_ncol=None,
                      ):
    if grid:
        plt.grid(axis='y')
        plt.gca().set_axisbelow(True)
    if ylim is not None:
        plt.ylim(ylim)
    if ytick_format_n is not None:
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.{}f'.format(ytick_format_n)))

    plt.xticks(x_axis, xticks, size=tick_size, rotation=x_axis_name_rotate)
    plt.yticks(yticks, size=tick_size)
    plt.xlabel(x_label, size=label_size)  # 横坐标名字
    plt.ylabel(y_label, size=label_size)  # 纵坐标名字
    plt.title(title, size=title_size)
    if legend:
        if legend_ncol is None:
            plt.legend(loc=legenc_loc, fontsize=legend_size, bbox_to_anchor=legend_bbox_to_anchor)
        else:
            plt.legend(loc=legenc_loc, fontsize=legend_size, bbox_to_anchor=legend_bbox_to_anchor,ncol=legend_ncol)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()


def line_plot(data, x_axis_names=None, line_names=None,
              marker_size=35, text_size=10, line_width=3,
              text_x_offset=-0.15, text_y_offset=0.05, colors=None, line_styles=None, with_dot=True,
              x_label=None, y_label=None, title=None, ylim=None, yticks=None, figsize=(8, 5),
              title_size=15, label_size=17, tick_size=15, legenc_loc=None, legend_size=13, ytick_format_n=None,
              save_name=None, x_axis_name_rotate=None, plot_text=True, grid=False, show=True,
              legend_bbox_to_anchor=None, legend_ncol=None,
              ):
    # colors = ['#5A9BD5', '#1B9E78', '#FF9966', '#ff585d', '#614ad3', '#feda77']  # TODO : add more colors # '#5A9BD5', '#FF9966', '
    # colors = ['#7F95D1', '#FF82A9', '#3DB29F', '#7ebc59']
    if colors is None:
        colors = LightColors
    if line_styles is None:
        # line_styles = ['-', ':', '-.', '-.', '-.']
        line_styles = ['-']*10#, ':', '-.', '-.', '-.']
    # dashList = [None, None, (5,2),(2,5),(4,10)]
    dashList = [None] * len(data)
    # dashList = [None, None, (2, 3), (2, 3), (2, 3)]
    # linestyle='--', dashes=dashList[n]

    plt.figure(figsize=figsize)
    if line_names is None:
        line_names = [None] * len(data)

    x_axis = np.arange(0, len(data[0]))

    for i, (line_data, style, name, color, dash) in enumerate(zip(data, line_styles, line_names, colors, dashList)):
        x_axis_t = [x_axis[i] for i, d in enumerate(line_data) if d != 0]
        line_data_t = [d for i, d in enumerate(line_data) if d != 0]

        if with_dot:
            plt.scatter(x_axis_t, line_data_t, color=color, s=marker_size)
        if dash is None:
            plt.plot(x_axis_t, line_data_t, linestyle=style, label=name, color=color, linewidth=line_width)
        else:
            plt.plot(x_axis_t, line_data_t, linestyle=style, dashes=dash, label=name, color=color, linewidth=line_width)

        for x, y in zip(x_axis_t, line_data_t):
            if plot_text and text_size > 0:
                plt.annotate(f"{y:.2f}", (x + text_x_offset, y + text_y_offset), fontsize=text_size)

    # x_axis = np.arange(0, len(data[0]), 2)
    # x_axis_names=['{:.1f}'.format(i / 10) for i in range(0, 10, 2)]
    set_common_params(x_label, y_label, title, ylim, title_size,
                      x_axis, x_axis_names, yticks, label_size, tick_size,
                      line_names[0] is not None, legenc_loc, legend_size, ytick_format_n,
                      save_name, x_axis_name_rotate, grid, show, legend_bbox_to_anchor, legend_ncol)


def bar_plot(data, x_axis_names=None, bar_names=None,
             width_of_all_col=0.7, offset_between_bars=0.02, text_size=None, colors=None,
             x_label=None, y_label=None, title=None, ylim=None, yticks=None, figsize=(8, 5),
             title_size=15, label_size=17, tick_size=15, legenc_loc=None, legend_size=13, ytick_format_n=None,
             save_name=None, x_axis_name_rotate=None, plot_text=False, grid=False, show=True,
              legend_bbox_to_anchor=None, legend_ncol=None,):
    """
    Args:
        data: shape of (Columns, DataNum)  (e.g., 3 methods x outputs of each method)
        x_axis_names: name of each data, same shape of DataNum
        bar_names:    name of each column  (e.g., method names)
        width_of_all_col: total width of all types of data ( all_width = bar_width * columns + offset * (columns-1))
        offset_between_bars: offset between data
        title_size : size of text over the bar, choices : [ll, small, medium, large, x-large, xx-large, larger]
        ylim: min and max of y-axis
        x_label: name of all x-axis
        y_label: name of all y-axis
        figsize: figure size
        save_name ; provide file name to save

    Examples:
        >>> domains = [[53.7, 51.8, 53.9, 54.8, 55.0, 55.1],
        >>>            [55.95, 57.19, 57.65, 57.94, 57.84, 57.64],
        >>>            [55.95, 57.19, 57.76, 57.87, 57.94, 57.67]
        >>>        ]
        >>> names = [str(i) for i in ['baseline', 1, 2, 4, 8, 16]]
        >>> labels = ['AGG', 'MLDG', 'MLDG+SIB']
        >>> ylim = [50, 60]
        >>> bar_plot(domains, names, labels, ylim=ylim)

    """
    # TODO : add more colors # '#5A9BD5', '#FF9966', '
    if colors is None:
        colors = LightColors[:2] + LighterColors[0:2]
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    num_of_cols, DataNum = data.shape

    if bar_names is None:
        bar_names = [None] * num_of_cols

    offset_between_col = offset_between_bars
    width_of_one_col = width_of_all_col / num_of_cols
    # start_x = start_x  #(width_of_one_col / 2 * (num_of_cols-1)) #(width_of_all_col - width_of_one_col) / num_of_cols
    start_x = np.arange(DataNum)
    plt.figure(figsize=figsize)

    # plt.gca().axhline(y=79.44, color='black', linestyle='-', zorder=0, label='Baseline')

    if grid:
        plt.grid(axis='y')
        plt.gca().set_axisbelow(True)
    for i, (l, c, label) in enumerate(zip(data, colors, bar_names)):
        # plot one bar
        h = plt.bar(start_x + width_of_one_col * i, l,
                    width=width_of_one_col - offset_between_col, color=c, label=label, linewidth=2.0)  # s-:方形

        # plot acc text over the bar
        if plot_text:
            for j, rect in enumerate(h):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{:.2f}'.format(data[i][j]), ha='center', va='bottom', size=text_size)

    # 'll, small, medium, large, x-large, xx-large, larger'
    plt.ylim(ylim)
    if x_axis_names is None:
        x_axis_names = np.arange(DataNum)
    x_axis = np.arange(DataNum) + width_of_all_col / 2 - width_of_one_col / 2
    set_common_params(x_label, y_label, title, ylim, title_size,
                      x_axis, x_axis_names, yticks, label_size, tick_size,
                      x_axis_names is not None, legenc_loc, legend_size, ytick_format_n,
                      save_name, x_axis_name_rotate, grid, show, legend_bbox_to_anchor, legend_ncol)


def regression_plot(data, line_names=None, marker_size=None, line_width=3,
                    colors=None,  x_axis_names=None, x_label=None, y_label=None,
                    title=None, ylim=None, yticks=None, figsize=(8, 5),
                    title_size=15, label_size=17, tick_size=15, legend_size=13,
                    legenc_loc=None, ytick_format_n=None,
                    save_name=None, x_axis_name_rotate=None, grid=False, show=True):
    from sklearn.linear_model import LinearRegression

    plt.figure(figsize=figsize)
    if colors is None:
        colors = LightColors
    for i, (line_data, line_name, color) in enumerate(zip(data, line_names, colors)):
        x = np.arange(0, len(line_data)).reshape(-1, 1)
        y = line_data
        plt.scatter(x, y, color=color, s=marker_size)
        reg = LinearRegression().fit(x, y)
        # cof = reg.score(x, y)
        # print(cof)
        x = np.linspace(x.min(), x.max())
        y = reg.predict(x.reshape(-1, 1)).reshape(-1)
        plt.plot(x, y, '-', label=line_name, color=color, linewidth=line_width)

    x_axis = np.arange(0, len(data[0]))
    set_common_params(x_label, y_label, title, ylim, title_size,
                      x_axis, x_axis_names, yticks, label_size, tick_size,
                      line_names is not None, legenc_loc, legend_size, ytick_format_n,
                      save_name, x_axis_name_rotate, grid, show)


def get_ticks_by_num(x_range, tick_num=None, tick_space=None):
    if tick_num is not None:
        tick_space = (x_range[1] - x_range[0]) / (tick_num - 1) - 1e-5
    return np.arange(x_range[0], x_range[1], tick_space)


def get_twin_fig(x, y1, y2, x_range, y_range, tick_num, labels=None, y_name=None, x_name=None, title=None,
                 legend_loc='upper left', figsize=(4, 2), save_name=None):
    w = 3
    fig, ax1 = plt.subplots(figsize=figsize)

    ax2 = ax1.twinx()  # 做镜像处理
    ln1 = ax1.plot(x, y1, '--', marker='v', label=labels[0] if labels is not None else None, color='#5A9BD5', linewidth=w)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.set_yticks(np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / (tick_num - 1) - 1e-5))
    ax1.set_ylabel(y_name[0] if y_name is not None else None)  # 设置Y1轴标题

    ln2 = ax2.plot(x, y2, '-', marker='s', label=labels[1] if labels is not None else None, color='#FF9966', linewidth=w)
    ax2.plot()
    ax2.set_yticks(np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / (tick_num - 1) - 1e-5))
    ax2.grid(False)
    ax2.set_ylabel(y_name[1] if y_name is not None else None)  # 设置Y1轴标题
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    lns = ln1 + ln2  # + [ ln3]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=legend_loc)

    ax1.set_xlabel(x_name)  # 设置x轴标题
    plt.title(title)

    ax3 = ax1.twinx()
    times = [2, 3, 4, 8, 14, 28, 55]
    ln3 = ax3.bar(x, times)
    for j, rect in enumerate(ln3):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{:d}'.format(times[j]), ha='center', va='bottom', size=14)
    ax3.set_ylim([0, 100])
    ax3.set_yticks([])

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


def test_colors(colors):
    n = len(colors)
    data = [np.random.randint(0, 100, (3)) for i in range(n)]
    bar_plot(data,
             colors=colors,
             width_of_all_col=0.8,
             offset_between_bars=0.02,
             figsize=(8, 4)
             )


if __name__ == '__main__':
    test_colors(LightColors)
