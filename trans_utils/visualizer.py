
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, repeat

# ax = plt.gca()


class Visualizer:

    @staticmethod
    def _show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    @staticmethod
    def _show_points(coords, labels, ax, marker_size=375):
        # points [(x,y), .... ]
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        # marker
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1)   
        
    @staticmethod
    def _show_box(box, ax):
        # box (x0, y0, x1, y1)
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    @staticmethod
    def reverse_points(points):
        return points[:, ::-1]

    def add_points(self, points, ax, color="green"):
        """
            Input points: (y,x), 
                we reverse the points from yx to xy for display
        """
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*', s=300, edgecolor='white', linewidth=1)

    def display_ori_data(self, bundle):
        # landmark: (x, y)
        img = bundle['img'][0]
        points = [p for k, p in bundle.items() if k.startswith("landmark_")]
        points = np.array(points)

        plt.figure(figsize=(20,20), dpi=100)
        plt.imshow(img, cmap="gray")
        self.add_points(points, plt.gca())
        plt.savefig("./tmp.jpg")

    def display(self, bundle, save_name=None):
        img = bundle['img']
        # points = rearrange(points, "c n -> n c") if points.shape[0] == 2 else points
        # points = np.concatenate([points, [[20,200]]], axis=0)
        # print("image.shape: ", img.shape)
        plt.figure(figsize=(20,20), dpi=100)
        plt.imshow(img, cmap="gray")
        
        mask = bundle.get('mask', None)
        if mask is not None:
            self._show_mask(mask, plt.gca())        
        
        points = bundle.get('landmarks', None)
        if points is not None:
            points = self.reverse_points(points)
            self.add_points(points, plt.gca())

        gt = bundle.get("gt", None)
        if gt is not None:
            # gt = rearrange(gt, "c n -> n c") if gt.shape[0] == 2 else gt
            gt = self.reverse_points(gt)
            self.add_points(gt, plt.gca(), color="red")

        save_name = "./tmp.jpg" if save_name is None else save_name
        plt.savefig(save_name)
    