<<<<<<< HEAD
以下是针对您的代码编写的中文README文件，用于指导用户如何有效地使用您的图像分割工具。

---

# 图像分割工具

本项目提供了一个图像分割工具，利用 Segment Anything Model (SAM) 对大规模的卫星或航拍图像进行分割。该工具支持通过单点、多点或边界框输入进行图像分割，并将分割结果保存为 shapefile，以便进一步进行地理空间分析。

## 功能特点

- **单点分割**：支持基于单个点的输入进行分割。
- **多点分割**：支持使用多个点进行分割。
- **边界框分割**：支持在指定的边界框内进行分割。
- **地理空间集成**：使用 GDAL 读取地理空间图像，并将分割的掩膜转换为多边形。
- **Shapefile 导出**：将分割结果保存为 shapefile，方便与 GIS 工具集成。
- **可视化**：在原始图像上可视化分割结果，便于验证和分析。

## 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/yourusername/ImageSegmentationTool.git
   cd ImageSegmentationTool
   ```

2. **下载SAM权重：**
	- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
	- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
	- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

	
3. **安装所需的依赖：**
   ```bash
   pip install -r requirements.txt
   ```

4. **设置环境变量：**
   - 代码内已设置 `KMP_DUPLICATE_LIB_OK` 变量，以避免冲突。


## 使用方法

### 步骤 1：准备数据

- **图像**：确保您拥有地理参考的卫星或航拍图像，格式为 TIFF。
- **SAM 模型检查点**：下载 SAM 模型检查点文件，并将其放置在项目目录中。

### 步骤 2：配置参数

在脚本中设置以下参数：

- `image_path`: 您的地理参考图像文件的路径（例如 `./sentinel2.tif`）。
- `sam_checkpoint`: 您的 SAM 模型检查点文件的路径（例如 `./sam_vit_b_01ec64.pth`）。
- `model_type`: 用于分割的模型类型（`vit_b`、`vit_l` 等）。
- `device`: 用于运行模型的设备（`cpu` 或 `cuda`）。
- `output_shp`: 保存输出 shapefile 的路径。

### 步骤 3：运行分割

选择分割模式并指定必要的输入点或边界框：

- **单点模式**：
  ```python
  seg_mode = 'single_point'
  input_points = [[1248, 1507]]
  single_label = [1]
  ```

- **多点模式**：
  ```python
  seg_mode = 'multi_point'
  input_points = [[389, 1041],[411, 1094]]
  single_label = [1, 1]
  ```

- **边界框模式**：
  ```python
  seg_mode = 'box'
  input_box = [[0, 951, 1909, 2383]]
  single_label = [1]
  ```

### 步骤 4：执行脚本

运行脚本以进行分割：

```bash
python main.py
```

### 步骤 5：可视化并保存结果

分割的掩膜将被可视化，多边形将作为 shapefile 保存到指定位置。

## 示例

使用边界框对图像进行分割，脚本配置如下：

```python
# 边界框模式示例配置
seg_mode = 'box'
input_box = [[0, 951, 1909, 2383]]
single_label = [1]

segmenter = ImageSegmentation(image_path, sam_checkpoint, model_type, device)
masks, scores, x_off, y_off = segmenter.predict(mode=seg_mode, input_box=input_box, input_labels=single_label, multimask_output=True)
polygons = segmenter.masks_to_polygons(masks, x_off, y_off)
segmenter.save_polygons_gdal(polygons, output_shp)
segmenter.show_masks(seg_mode, masks, scores, x_off, y_off, input_box, single_label, image_chunk)
```

## 故障排除

如果您遇到任何问题，请确保以下几点：

- 所有依赖项均已正确安装。
- 图像和检查点路径正确无误。
- 分割模式和输入点/边界框已正确配置。

## 许可证

本项目基于 MIT 许可证进行许可。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- **Segment Anything Model (SAM)**：本项目使用 SAM 模型进行图像分割。

---

如有需要，您可以根据项目的具体要求进一步修改 README 的内容！
=======
# ImageSegmentationTool-SAM
An interactive annotation case developed based on SAM for remote sensing image annotation, which can generate corresponding segmentation results based on point, multi-point, and rectangular box prompts, and convert the recognition results into vector data shp.
>>>>>>> 8e1f2cdd4ff3661a961e99c2dca1f1a1c917073a
