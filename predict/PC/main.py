# main.py

import os
import warnings
import numpy as np
import torch
import xarray as xr
import rioxarray
import pystac_client
import planetary_computer
import matplotlib.pyplot as plt
from torchvision import transforms
from odc.stac import load
import time
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from tqdm import tqdm
# 从单独的文件中导入模型加载函数
from model import get_model
# 忽略 rasterio 库中特定的"未地理参考"警告
# warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
# --- 参数设置 ---
AOI_BBOX = [109.59, 40.20, 109.80, 40.31]
TIME_RANGE = "2024-10-01/2024-12-31"
PATCH_SIZE = 512
OUTPUT_GEOTIFF_PATH = "prediction_mask.tif"


# --- 函数定义 ---
def get_sentinel2_data(bbox, time_range):
    """
    使用 odc-stac 直接从云端读取并裁剪数据，避免下载完整影像。
    这是最高效的方法。
    """
    print("🛰️  Searching for Sentinel-2 L2A imagery...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No items found for the given BBOX and time range ({time_range}).")

    print(f"✅ Found {len(items)} scenes. Loading and clipping directly from cloud via odc-stac...")

    data = load(
        items,
        bands=["B04", "B03", "B02"],
        bbox=bbox,
        crs="EPSG:3857",
        resolution=10,
        patch_url=planetary_computer.sign,
        chunks={},
    )

    # data 是一个Dataset对象。我们先选择第一个时间点并触发计算。
    data_ds = data.isel(time=0).compute()

    # 1. 将 Dataset 转换为一个单一的 DataArray。
    #    B04, B03, B02 会被堆叠到一个新的维度 'variable' 上。
    data_da = data_ds.to_dataarray(dim="variable")

    # 2. 为确保后续处理的颜色通道顺序正确 (RGB)，我们使用 .sel() 显式选择顺序。
    clipped_data = data_da.sel(variable=["B04", "B03", "B02"])

    # 现在 clipped_data 是一个 DataArray，可以安全地访问 .shape 属性
    print(f"✅ Clipping and loading complete. Final shape: {clipped_data.shape}")

    return clipped_data


def predict_single_tile(model, tile_numpy, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tile_float = np.clip(tile_numpy / 3000, 0, 1).astype(np.float32)
    tile_transposed = np.transpose(tile_float, (1, 2, 0))

    with torch.no_grad():
        input_tensor = transform(tile_transposed).unsqueeze(0).to(device)
        output = model(input_tensor)['out']
        prediction_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        return prediction_mask.astype(np.uint8)


def save_geotiff(geo_array, filepath):
    print(f"💾 Saving georeferenced prediction to {filepath}...")
    try:
        geo_array.rio.write_nodata(0, inplace=True)
        geo_array.rio.to_raster(filepath, compress='LZW', dtype='uint8')
        print(f"✅ Save complete. You can now open '{filepath}' in QGIS or ArcGIS.")
    except Exception as e:
        print(f"❌ Error saving GeoTIFF: {e}")


# --- 主执行函数 ---
if __name__ == "__main__":
    try:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {DEVICE}")

        geo_referenced_data = get_sentinel2_data(AOI_BBOX, TIME_RANGE)

        print("📋 Creating an empty canvas for stitched predictions...")
        stitched_prediction = xr.DataArray(
            np.zeros((geo_referenced_data.y.size, geo_referenced_data.x.size), dtype=np.uint8),
            coords={"y": geo_referenced_data.y, "x": geo_referenced_data.x},
            dims=("y", "x"),
        )
        stitched_prediction.rio.write_crs(geo_referenced_data.rio.crs, inplace=True)
        stitched_prediction.rio.write_transform(geo_referenced_data.rio.transform(), inplace=True)

        model = get_model(device=DEVICE)

        print(f"🤖 Preparing to predict with patch size {PATCH_SIZE}x{PATCH_SIZE}...")
        height = geo_referenced_data.y.size
        width = geo_referenced_data.x.size

        patch_coords = []
        for y_start in range(0, height, PATCH_SIZE):
            for x_start in range(0, width, PATCH_SIZE):
                if (y_start + PATCH_SIZE <= height) and (x_start + PATCH_SIZE <= width):
                    patch_coords.append((y_start, x_start))

        # --- 2. 核心修改：在循环中加入重试逻辑 ---
        for y_start, x_start in tqdm(patch_coords, desc="Processing Patches"):
            y_end = y_start + PATCH_SIZE
            x_end = x_start + PATCH_SIZE

            tile_data = None
            max_retries = 5  # 最多重试5次
            for attempt in range(max_retries):
                try:
                    # 尝试读取数据，这是唯一可能出错的地方
                    tile_data = geo_referenced_data[:, y_start:y_end, x_start:x_end].values
                    # 如果成功，就跳出重试循环
                    break
                except RasterioIOError as e:
                    # 如果捕获到IO错误，打印警告并等待
                    wait_time = (attempt + 1) * 2  # 等待时间逐渐增加
                    tqdm.write(
                        f"⚠️ Rasterio IO error on patch ({y_start}, {x_start}). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

            # 如果重试了5次仍然失败，则跳过这个图块
            if tile_data is None:
                tqdm.write(f"❌ Failed to read patch ({y_start}, {x_start}) after {max_retries} attempts. Skipping.")
                continue

            # --- 如果读取成功，则继续执行预测和拼接 ---
            prediction_mask = predict_single_tile(model, tile_data, DEVICE)
            stitched_prediction[y_start:y_end, x_start:x_end] = prediction_mask

        # print(f"✅ Prediction and stitching complete. Processed a total of {len(patch_coords)} patches.")

        save_geotiff(stitched_prediction, OUTPUT_GEOTIFF_PATH)

        # (可选) 显示样本...
        # ... (显示部分代码保持不变) ...
        print("🖼️  Displaying a sample tile and its prediction for quick preview...")
        # 显示第一个patch
        sample_tile = geo_referenced_data[:, 0:PATCH_SIZE, 0:PATCH_SIZE].values
        sample_mask = stitched_prediction[0:PATCH_SIZE, 0:PATCH_SIZE].values

        BRIGHTNESS_ADJUST_VALUE = 3000  # 尝试调低这个值，比如 2500, 2000, 1800
        sample_tile_display = np.clip(sample_tile / BRIGHTNESS_ADJUST_VALUE, 0, 1) * 255
        sample_tile_display = sample_tile_display.astype(np.uint8)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(np.transpose(sample_tile_display, (1, 2, 0)))
        ax1.set_title("Original Image Patch (Sample)")
        ax1.axis('off')

        ax2.imshow(sample_mask)
        ax2.set_title("Predicted Mask (Sample)")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        import traceback

        traceback.print_exc()
        print("---------------------------")
