
import numpy as np
from scipy.integrate import quad
from scipy import stats
import os
import glob
from PIL import Image
from visualizer import Visualizer

def swap_x_and_y(points):
    # 交换 x 和 y 坐标
    swapped_points = np.array([(point[1], point[0]) for point in points])
    return swapped_points

def fit_curve(points, degree=4):
    # 拟合多项式曲线 (输入y得到x)
    coefficients = np.polyfit(points[:, 0], points[:, 1], degree)
    return np.poly1d(coefficients)

def find_intersection(curve, line):
    # 找到曲线和直线的交点
    # curve 是曲线的多项式函数，line 是直线的系数 (m, n)
    intersection_x = np.roots(curve -  [line[1], -line[0]])
    intersection_y = curve(intersection_x)
    return intersection_x[0], intersection_y[0]

def curve_length(curve, x1, x2):
    # 计算曲线在区间 [x1, x2] 上的长度
    integrand = lambda x: np.sqrt(1 + (curve.deriv()(x))**2)
    length, _ = quad(integrand, x1, x2)
    return length

def main():
    # 示例数据点
    points = np.array([(1, 2), (2, 3), (3, 4), (4, 5)])

    # 交换 x 和 y 坐标
    swapped_points = swap_x_and_y(points)

    # 拟合三阶曲线
    curve = fit_curve(swapped_points)

    # 直线的系数 (假设直线为 y = mx + n)
    line_coefficients = (0.5, 1)

    # 找到曲线和直线的交点
    intersection_x, _ = find_intersection(curve, line_coefficients)

    # 计算第一个点到交点的曲线长度
    first_point_x = swapped_points[0][0]
    length = curve_length(curve, first_point_x, intersection_x)

    print("曲线系数:", curve.coefficients)
    print("直线系数:", line_coefficients)
    print("交点坐标:", (intersection_x, curve(intersection_x)))
    print("第一个点到交点的曲线长度:", length)


def tmp(anal_points, tumor_point, rectum_mask):
    def find_left_right_coordinates(vector):
        indices = np.where(vector >= 1)[0]
        if len(indices) == 0:
            return None, None
        left_coordinate = indices[0]
        right_coordinate = indices[-1]
        return (left_coordinate + right_coordinate) / 2

    tumor_point_y = tumor_point[0]

    center_points = []
    for i in range(rectum_mask.shape[0]):
        line = rectum_mask[i,:]
        if line.sum() >= 2:# i >= tumor_point_y and 
            center_points.append([i, find_left_right_coordinates(line)])

    curve_points = np.concatenate([center_points, [tumor_point]], axis=0)
    curve = fit_curve(curve_points)
    
    y1, x1 = zip(*anal_points)
    slope1, intercept1, _, _, _ = stats.linregress(y1, x1)

    # intersection = find_intersection(curve, (slope1, intercept1))
    # tumor_point_proj = (tumor_point_y, curve(tumor_point_y))
    anal_mid_point = (np.array(y1).mean(), curve(np.array(y1).mean()))
    
    length = curve_length(curve, tumor_point_y, np.array(y1).mean())
    
    curve_line_points = []
    for c in center_points:
        curve_line_points.append([c[0], np.clip(curve(c[0]), 0, rectum_mask.shape[1])])
    print(center_points)

    line_points = []
    for c in np.arange(min(y1), max(y1)):
        line_points.append([c, slope1*c+intercept1])

    return length, curve_line_points, center_points, anal_mid_point, line_points


def calc_anal_ring_to_margin(ring, margin):
    def calculate_distance(point1, point2):
        # 使用 NumPy 计算两点之间的欧几里得距离
        distance = np.linalg.norm(point1 - point2)
        return distance
    distance_collect = []
    for p in margin:
        distance_collect.append(calculate_distance(p, ring))
    return np.array(distance_collect).mean()


class AnallinePredictor:
    def __init__(self) -> None:
        self.spacing = 0.875

    def main(self):
        img_dir = "/home1/zhhli/sagjpg/"
        mask_dir = "/home1/zhhli/sagmask/"
        dirpath = "/home1/quanquan/code/landmark/code/runs/ssl_probmap/debug3/"
        paths = glob.glob(os.path.join(dirpath, "visuals", "*"))[2:]
        
        # /home1/quanquan/code/landmark/code/runs/ssl_probmap/debug3/visuals/11182324_11/
        # /home1/quanquan/code/landmark/code/runs/ssl_probmap/debug3/visuals/11182324_11/pred_lanrmarks.npy

        for p in paths:
            landmark_path = os.path.join(p, "pred_landmarks.npy")
            name = p.split("/")[-1]
            img_path = os.path.join(img_dir, name+".jpg")
            mask_path = os.path.join(mask_dir, name+".npy")
            if not os.path.exists(mask_path):
                print("Ignore ", mask_path)
                continue

            anal_points = np.load(landmark_path)
            mask = np.load(mask_path)

            tumor = np.int32(mask==2)
            rectum = np.int32(mask==1) + np.int32(mask==2)

            def find_left_right_coordinates(vector):
                indices = np.where(vector >= 1)[0]
                if len(indices) == 0:
                    return None, None
                left_coordinate = indices[0]
                right_coordinate = indices[-1]
                return left_coordinate, right_coordinate
            
            tumor_point_y = find_left_right_coordinates(tumor.sum(axis=1))[-1]
            tumor_point_x = np.array(find_left_right_coordinates(tumor[tumor_point_y,:])).mean()

            length, curve_line_points , curve_points, intersection, line_points = tmp(anal_points[:2], [tumor_point_y,tumor_point_x], rectum)
            print(length * self.spacing)

            length2 = calc_anal_ring_to_margin(intersection, anal_points[2:])
            print(length2 * self.spacing)

            img = Image.open(img_path).convert("L")
            # import ipdb; ipdb.set_trace()
            bundle = {"img": img, "landmarks": np.concatenate([anal_points, [[tumor_point_y,tumor_point_x]], np.array([intersection])]), "gt": np.array(curve_line_points)}
            # bundle = {"img": img, "landmarks": np.array([intersection]), "gt": np.array(curve_line_points)}
            # bundle = {"img": img, "landmarks": np.array(curve_line_points), "gt": np.array(line_points)}

            break
        self.visualize(bundle)
        # import ipdb; ipdb.set_trace()

    def visualize(self, bundle): # img, anal_points, tumor_point, curve_points
        vis = Visualizer()
        vis.display(bundle, save_name="./tumor2anus.jpg")


if __name__ == "__main__":
    pre = AnallinePredictor()
    pre.main()
