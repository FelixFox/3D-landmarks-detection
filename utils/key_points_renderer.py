import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import cv2
import os
from scipy import signal
from sklearn import decomposition


class KeyPointsRenderer:
    """Renders key points animation with given path of image sequence. Created for 68 face key points.

    Attributes:
        fig (Figure):    pyplot figure
        ax (Axes): contains most of the `fig`'s elements
        frames_path (str): path, where frames' info (.txt files with points) is located
        frames_num (int): number of frames
        frames_step (int): defines step of points rendering
        lines_color (str): defines color of face's lines 

    """

    def __init__(self, frames_step=1, filter_points=True, lines_color='black'):
        """Initializes all params required for rendering.

        Note:
            `frames_path` must end WITHOUT slash (`/`) or (`\\`)

        Args: 
            frames_path(str): frames_path (str): path, where frames' info (.txt files with points) are located
            frames_step (int): defines step of points rendering
            lines_color (str): defines color of face's lines

        """
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.gca(projection='3d')
        self.frames = np.array([])
        self.frames_step = frames_step
        self.lines_color = lines_color
        self.filter_points = filter_points

        self.__init_ax()

    def __init_ax(self):
        """ Sets up params for `self.ax` (Axes instance)
        """
        self.ax.view_init(elev=90, azim=90)
        self.ax.invert_xaxis()

    def load_frames(self, frames_path):
        """Loads key points info (frames) into numpy array.

        Args:
            frames_path - path with .txt frames

        Returns:
            frames_array - numpy array with frames
            frames_num - number of frames
        """
        files = [os.path.join(frames_path, name) for name in os.listdir(
            frames_path) if name.endswith('.txt')]
        frames_array = np.array([self.read_pts68_from_txt(f) for f in files])
        return frames_array, len(frames_array)

    def render_frame(self, pts,  style='fancy',   **kwargs):
        """Draw landmarks using matpliotlib

        Parameters:
            pts (np.array): 3-dimensional numpy array with key points

        """

        if not type(pts) in [tuple, list]:
            pts = [pts]
        for i in range(len(pts)):
            if style == 'simple':
                plt.plot(pts[i][0, :], pts[i][1, :], pts[i]
                         [2, :], 'o', markersize=4, color='g')

            elif style == 'fancy':
                alpha = 0.8
                markersize = 4
                lw = 1.5
                color = kwargs.get('color', 'w')
                markeredgecolor = kwargs.get('markeredgecolor', 'black')

                nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

                # close eyes and mouths
                def plot_close(i1, i2): return plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                        [pts[i][2, i1], pts[i][2, i2]], color=color, lw=lw, alpha=alpha - 0.1)
                plot_close(41, 36)
                plot_close(47, 42)
                plot_close(59, 48)
                plot_close(67, 60)

                for ind in range(len(nums) - 1):
                    l, r = nums[ind], nums[ind + 1]
                    plt.plot(pts[i][0, l:r], pts[i][1, l:r], pts[i][2, l:r],
                             color=color, lw=lw, alpha=alpha - 0.1)

                    plt.plot(pts[i][0, l:r], pts[i][1, l:r], pts[i][2, l:r], marker='o', linestyle='None', markersize=markersize,
                             color=color,
                             markeredgecolor=markeredgecolor, alpha=alpha)

        plt.axis('off')
        plt.tight_layout()

    def read_pts68_from_txt(self, filepath):
        """Retrieves points from given txt

        Parameters:
            filepath - path of .txt file with points

        Returns:
            3-dimensional numpy array with points (coordinates of  type `float`)
        """
        with open(filepath) as fp:
            lines = fp.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            pts68 = [list(map(float, num_list)) for num_list in lines]
            return np.array(pts68)

    def __animate(self, frame):
        """The function to call at each frame. The first argument will be the next value in frames.

        Parameters:
            frame(int): number of current frame
        """
        self.ax.clear()
        self.render_frame(self.frames[frame], color=self.lines_color)

    def animate_key_points_from_txts(self, frames_path, interval=50):
        """The function for animation creation form given path with txt frames

        Parameters:
            interval(int): Delay between frames in milliseconds. Defaults to 50.
            frames_path (str): Path to frames (withot slash in the end of the string)
        """
        self.frames,  frames_num = self.load_frames(frames_path)
        if self.filter_points:
            self.apply_filter()

        ani = animation.FuncAnimation(self.fig, self.__animate, frames=range(
            0, frames_num, self.frames_step), interval=interval, blit=False, save_count=100)
        ani._start()
        plt.show()

    def animate_key_points_from_frames(self, frames, interval=50):
        """The function for animation creation form given numpy array with frames. It should be with shape
            (num_frames, pts_dim, pts_num)

        Parameters:
            interval(int): Delay between frames in milliseconds. Defaults to 50.
            frames (np.array): Numpy array with key points (frames)
        """
        self.frames = frames
        frames_num = len(frames)
        if self.filter_points:
            self.apply_filter()
            

        ani = animation.FuncAnimation(self.fig, self.__animate, frames=range(
            0, frames_num, self.frames_step), interval=interval, blit=False, save_count=100)
        ani._start()
        plt.show()

    def apply_filter(self, num_components=30, filterHalfLength=2):
        """Build a 3D Shape Model by fitting a Multivariate Gaussian (PCA); Project the original shapes onto our Shape Model and Reconstruct;
            Applies temporal filter to frames, using PCA method.  Also applies normalization. 
           Remember that PCA is essentially fitting a multivariate gaussian distribution with a low rank covariance matrix.

        """
        key_pts = self.__get_shapes_model(num_components=num_components)
        temporal_filter = np.ones((1, 1, 2*filterHalfLength+1))
        temporal_filter = temporal_filter / temporal_filter.sum()
        start_tile_block = np.tile(key_pts[:, :, 0][:, :, np.newaxis], [1, 1, filterHalfLength])
        end_tile_block = np.tile(key_pts[:, :, -1][:, :, np.newaxis], [1, 1, filterHalfLength])
        key_pts_padded = np.dstack((start_tile_block, key_pts, end_tile_block))
        key_pts_filtered = signal.convolve(key_pts_padded, temporal_filter, mode='valid', method='fft')

        self.frames = np.swapaxes(key_pts_filtered, 0, 2)

    def __get_shapes_model(self, num_components):
        """Build a 3d shape model by fitting a Multivariate Gaussian (PCA), 
        project the original shapes onto our Shape Model and Reconstruct
        """
        key_pts_norm, scale_factors, mean_coords = self.__normalise_shapes()
        norm_shapes_table = np.reshape(key_pts_norm, [68*3, key_pts_norm.shape[2]]).T
        shapesModel = decomposition.PCA(n_components=num_components, whiten=True, random_state=1).fit(norm_shapes_table)
        # convert to matrix form
        key_pts_norm_table = np.reshape(key_pts_norm, [68*3, key_pts_norm.shape[2]]).T
        # project onto shapes model and reconstruct
        key_pts_norm_table_rec = shapesModel.inverse_transform(shapesModel.transform(key_pts_norm_table))
        # convert back to shapes (numKeypoint, num_dims, numFrames)
        key_pts_norm_rec = np.reshape(key_pts_norm_table_rec.T, [68, 3, key_pts_norm.shape[2]])
        # transform back to image coords
        key_pts = self.__transform_shape_back2image_coords(key_pts_norm_rec, scale_factors, mean_coords)
        return key_pts

    def __transform_shape_back2image_coords(self, norm_shapes, scale_factors, mean_coords):
        """shapes_im_coords_rec = TransformShapeBackToImageCoords(norm_shapes, scale_factors, mean_coords)"""
        (num_points, num_dims, _) = norm_shapes.shape
        # move back to the correct scale
        shapes_centered = norm_shapes * \
            np.tile(scale_factors, [num_points, num_dims, 1])
        # move back to the correct location
        shapes_im_coords = shapes_centered + \
            np.tile(mean_coords, [num_points, 1, 1])
        return shapes_im_coords

    def __normalise_shapes(self):
        """in order to compare apples to apples (or in this case, shapes to shapes),
           we need first to normalize the shapes in and manually remove the things that
           we don't care about (in this case, we want to disregard translation and scale differences between shapes, 
           and model only the shape's shape :-) )
        """
        shapes_im_coords = np.swapaxes(self.frames, 0, 2)
        (num_points, num_dims, _) = shapes_im_coords.shape
        """norm_shapes, scale_factors, mean_coords  = NormlizeShapes(shapes_im_coords)"""
        # calc mean coords and subtract from shapes
        mean_coords = shapes_im_coords.mean(axis=0)

        shapes_centered = np.zeros(shapes_im_coords.shape)
        tiled = np.tile(mean_coords, [num_points, 1, 1])
        shapes_centered = shapes_im_coords - \
            np.tile(mean_coords, [num_points, 1, 1])

        # calc scale factors and divide shapes
        scale_factors = np.sqrt((shapes_centered**2).sum(axis=1)).mean(axis=0)
        norm_shapes = np.zeros(shapes_centered.shape)
        norm_shapes = shapes_centered / \
            np.tile(scale_factors, [num_points, num_dims, 1])

        return norm_shapes, scale_factors, mean_coords


