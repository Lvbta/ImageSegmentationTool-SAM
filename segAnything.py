import numpy as np
import torch
import cv2
import sys
from osgeo import gdal, ogr, osr
from shapely.geometry import Polygon
from shapely.wkb import dumps
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('ggplot')


class ImageSegmentation:
    def __init__(self, image_path, sam_checkpoint, model_type='vit_b', device='cpu'):
        self.image_path = image_path
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self.geo_transform, self.proj = self.get_geoinfo()
        self.sam = self.load_sam_model()
        self.predictor = self.init_predictor()

    def get_geoinfo(self):
        dataset = gdal.Open(self.image_path)
        geo_transform = dataset.GetGeoTransform()
        proj = dataset.GetProjection()
        dataset = None  # 关闭
        return geo_transform, proj

    def read_image_chunk(self, x_off, y_off, x_size, y_size):
        dataset = gdal.Open(self.image_path)
        image = dataset.ReadAsArray(x_off, y_off, x_size, y_size)
        dataset = None  # 关闭
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))  # GDAL reads in (bands, height, width) format
        else:
            image = np.stack([image] * 3, axis=-1)  # If it's a single-band image, stack to (height, width, 3)
        return image

    def load_sam_model(self):
        sys.path.append("..")
        from segment_anything import sam_model_registry

        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return sam

    def init_predictor(self):
        from segment_anything import SamPredictor

        predictor = SamPredictor(self.sam)
        return predictor

    def predict(self, mode='single_point', input_points=None, input_labels=None, input_box=None, multimask_output=None):
        if mode == 'single_point':
            assert input_points is not None and input_labels is not None, "Points and labels are required for single point mode."
            x, y = input_points[0]
            chunk_size = 512  # or any appropriate size
            x_off = max(x - chunk_size // 2, 0)
            y_off = max(y - chunk_size // 2, 0)
            x_size = y_size = chunk_size

            image_chunk = self.read_image_chunk(x_off, y_off, x_size, y_size)
            self.predictor.set_image(image_chunk)

            adjusted_points = [(x - x_off, y - y_off)]
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(adjusted_points),
                point_labels=np.array(input_labels),
                multimask_output=multimask_output,
            )
        elif mode == 'multi_point':
            assert input_points is not None and input_labels is not None, "Points and labels are required for multi point mode."
            # Determine bounding box of all points
            x_min = min(p[0] for p in input_points)
            y_min = min(p[1] for p in input_points)
            x_max = max(p[0] for p in input_points)
            y_max = max(p[1] for p in input_points)
            margin = 256  # or any appropriate margin
            x_off = max(x_min - margin, 0)
            y_off = max(y_min - margin, 0)
            x_size = min(x_max - x_min + 2 * margin, 2048)
            y_size = min(y_max - y_min + 2 * margin, 2048)

            image_chunk = self.read_image_chunk(x_off, y_off, x_size, y_size)
            self.predictor.set_image(image_chunk)

            adjusted_points = [(x - x_off, y - y_off) for x, y in input_points]
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(adjusted_points),
                point_labels=np.array(input_labels),
                multimask_output=multimask_output,
            )
        elif mode == 'box':
            assert input_box is not None, "Box coordinates are required for box mode."
            x_min, y_min, x_max, y_max = input_box[0]
            margin = 256  # or any appropriate margin
            x_off = max(x_min - margin, 0)
            y_off = max(y_min - margin, 0)
            x_size = min(x_max - x_min + 2 * margin, 2048)
            y_size = min(y_max - y_min + 2 * margin, 2048)

            image_chunk = self.read_image_chunk(x_off, y_off, x_size, y_size)
            self.predictor.set_image(image_chunk)

            adjusted_box = [(x_min - x_off, y_min - y_off, x_max - x_off, y_max - y_off)]
            masks, scores, logits = self.predictor.predict(
                box=np.array(adjusted_box).reshape(1, -1),
                multimask_output=multimask_output,
            )
        else:
            raise ValueError("Mode must be 'single_point', 'multi_point', or 'box'.")

        return masks, scores, x_off, y_off

    def masks_to_polygons(self, masks, x_off, y_off):
        polygons = []
        for mask in masks:
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if len(contour.shape) == 2 and len(contour) >= 3:  # valid polygon
                    geo_contour = [self.pixel_to_geo(x + x_off, y + y_off) for x, y in contour]
                    polygon = Polygon(geo_contour)
                    if polygon.is_valid:
                        polygons.append(polygon)
        return polygons

    def pixel_to_geo(self, x, y):
        geox = self.geo_transform[0] + x * self.geo_transform[1] + y * self.geo_transform[2]
        geoy = self.geo_transform[3] + x * self.geo_transform[4] + y * self.geo_transform[5]
        return geox, geoy

    def save_polygons_gdal(self, polygons, output_shp):
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(output_shp)

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromWkt(self.proj)  # 使用图像的投影信息

        layer = data_source.CreateLayer("segmentation", spatial_ref, ogr.wkbPolygon)
        layer_defn = layer.GetLayerDefn()

        for i, polygon in enumerate(polygons):
            feature = ogr.Feature(layer_defn)
            geom_wkb = dumps(polygon)  # 将Shapely几何对象转换为WKB
            ogr_geom = ogr.CreateGeometryFromWkb(geom_wkb)  # 从WKB创建OGR几何对象
            feature.SetGeometry(ogr_geom)
            feature.SetField("id", i + 1)
            layer.CreateFeature(feature)
            feature = None

        data_source = None

    def show_masks(self, mode, masks, scores,x_off, y_off, input_point, input_label, image):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            if mode == 'box':
                self.show_box(np.array(input_point[0]), plt.gca(), x_off, y_off)
            else:
                self.show_points(np.array(input_point), np.array(input_label), plt.gca(), x_off, y_off)
            plt.title(f"{mode}模式 {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('on')
            plt.show()

    def show_mask(self, mask, ax, x_off=0, y_off=0):
        mask_resized = np.zeros((mask.shape[0] + y_off, mask.shape[1] + x_off), dtype=np.uint8)
        mask_resized[y_off:y_off + mask.shape[0], x_off:x_off + mask.shape[1]] = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour[:, :, 0] += x_off
            contour[:, :, 1] += y_off
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='lime', linewidth=2)

    def show_points(self, points, labels, ax, x_off, y_off):
        for point, label in zip(points, labels):
            x, y = point
            x -= x_off  
            y -= y_off  
            ax.scatter(x, y, c='red', marker='o', label=f'Label: {label}')

    @staticmethod
    def show_box(box, ax, x_off, y_off):
        x0, y0 = box[0]-x_off, box[1]-y_off
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == '__main__':
    # Usage
    image_path = r'./data/sentinel2.tif'
    sam_checkpoint = "./model/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"
    output_shp = r'./result/segmentation_results.shp'
    # # 预测模式
    # seg_mode = 'single_point'
    # # # 模型参数
    # input_points = [[1248, 1507]]
    # single_label = [1]

    # # 预测模式
    # seg_mode = 'multi_point'
    # # 模型参数
    # input_points = [[389, 1041],[411, 1094]]
    # single_label = [1, 1]

    # 预测模式
    seg_mode = 'box'
    # 模型参数
    input_box = [[0, 951, 1909, 2383]]
    single_label = [1]


    # 实例化类
    segmenter = ImageSegmentation(image_path, sam_checkpoint, model_type, device)

    # # 调用segAnything模型
    # masks, scores, x_off, y_off = segmenter.predict(mode=seg_mode, input_points=input_points,
    #                                                     input_labels=single_label, multimask_output=False)
    # box
    masks, scores, x_off, y_off = segmenter.predict(mode=seg_mode, input_box=input_box,
                                                    input_labels=single_label, multimask_output=True)

    # 模型预测结果转矢量多边形
    polygons = segmenter.masks_to_polygons(masks, x_off, y_off)

    # 保存为shp
    segmenter.save_polygons_gdal(polygons, output_shp)

    # 可视化
    image_chunk = segmenter.read_image_chunk(x_off, y_off, 512, 512)
    # segmenter.show_masks(seg_mode, masks, scores, x_off, y_off, input_points, single_label, image_chunk)
    # box
    segmenter.show_masks(seg_mode, masks, scores, x_off, y_off, input_box, single_label, image_chunk)


