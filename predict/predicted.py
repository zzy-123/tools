from osgeo import gdal
import numpy as np
from functools import wraps
import torch
from model.UPerNet import UPerNet
from glob import glob
import os
from tqdm import tqdm

def blockwise_processor(block_size=256, use_blocks=True, dtype=gdal.GDT_Byte):
    """
    分块处理装饰器，用于对大栅格数据进行分块处理。
    现在每次读取的数据包括所有波段，形状为 [bands, x_block_size, y_block_size]。
    :param block_size: 分块大小
    :param dtype: 输出数据类型
    """
    def decorator(process_fn):
        @wraps(process_fn)
        def wrapper(src_filename, dst_filename, dst_bands=1,*args, **kwargs):
            # 打开源文件
            src_ds = gdal.Open(src_filename, gdal.GA_ReadOnly)
            if src_ds is None:
                raise ValueError(f"无法打开源文件: {src_filename}")

            # 获取栅格的基本信息
            cols = src_ds.RasterXSize
            rows = src_ds.RasterYSize
            bands = src_ds.RasterCount
            geotransform = src_ds.GetGeoTransform()
            projection = src_ds.GetProjection()

            # 创建目标文件
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(dst_filename, cols, rows, dst_bands, dtype, options=['COMPRESS=LZW'])
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(projection)

            if use_blocks:
                # 按块读取和处理（包括所有波段）
                for i in range(0, rows, block_size):
                    for j in range(0, cols, block_size):
                        # 确定当前块的大小（处理边界情况）
                        x_block_size = min(block_size, cols - j)
                        y_block_size = min(block_size, rows - i)

                        # 读取所有波段的块数据，形状为 [bands, y_block_size, x_block_size]
                        block_data = np.zeros((bands, y_block_size, x_block_size), dtype=np.float32)
                        for b in range(bands):
                            band = src_ds.GetRasterBand(b + 1)  # 波段从1开始
                            block_data[b, :, :] = band.ReadAsArray(j, i, x_block_size, y_block_size)

                        # 调用用户定义的处理函数
                        processed_block_data = process_fn(block_data, *args, **kwargs)

                        # 写入目标文件中的所有波段
                        if len(processed_block_data.shape)==2:
                            dst_band = dst_ds.GetRasterBand(1)
                            # # 消除小斑块
                            # gdal.SieveFilter(srcBand=dst_band, maskBand=None, dstBand=dst_band,
                            #                  threshold=100,
                            #                  connectedness=8)
                            dst_band.WriteArray(processed_block_data, j, i)
                        else:
                            for b in range(dst_bands):
                                dst_band = dst_ds.GetRasterBand(b + 1)
                                dst_band.WriteArray(processed_block_data[b, :, :], j, i)

            else:
                # 读取整个栅格数据，形状为 [bands, rows, cols]
                full_data = np.zeros((bands, rows, cols), dtype=np.float32)
                for b in range(bands):
                    band = src_ds.GetRasterBand(b + 1)
                    full_data[b, :, :] = band.ReadAsArray()

                # 调用用户定义的处理函数
                processed_full_data = process_fn(full_data, *args, **kwargs)

                # 写入整个数据到目标文件
                # 写入目标文件中的所有波段
                if len(processed_full_data.shape) == 2:
                    dst_band = dst_ds.GetRasterBand(1)
                    # 消除小斑块
                    # gdal.SieveFilter(srcBand=dst_band, maskBand=None, dstBand=dst_band,
                    #                  threshold=100,
                    #                  connectedness=8)
                    dst_band.WriteArray(processed_full_data)
                else:
                    for b in range(dst_bands):
                        dst_band = dst_ds.GetRasterBand(b + 1)
                        dst_band.WriteArray(processed_full_data[b, :, :])

            # 确保数据写入完成
            dst_ds.FlushCache()

            # 关闭数据集
            src_ds = None
            dst_ds = None

        return wrapper
    return decorator

def load_net(model_path, device='cuda'):
    # 加载模型
    net = UPerNet(3, [3, 4, 6, 3], num_class=1)
    net.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    net.to(device)
    net.eval()
    return net

@blockwise_processor(use_blocks=False)
def predict_block( block_data, net, device='cuda', threshold=0.5, mask_multiplier=1):
    # 转换为张量
    x = torch.from_numpy(block_data).to(device)
    x = x.unsqueeze(0).float()

    # 预测
    with torch.no_grad():
        pred = net(x)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().cpu().numpy()
        # 阈值0.5二值化
        pred = (pred > threshold).astype(np.int_)

    return pred*mask_multiplier

def predict_batch(image_dir,mask_dir,net,device="cuda",threshold=0.5, mask_multiplier=1):
    image_list = glob(image_dir+'/*.tif')
    for i in tqdm(range(len(image_list))):
        src_filename = image_list[i]
        dst_filename = os.path.join(mask_dir,os.path.basename(src_filename))
        predict_block(src_filename, dst_filename,net=net, device=device, threshold=threshold, mask_multiplier=mask_multiplier)


if __name__ == '__main__':
    # 设置文件路径
    # src_filename = '/mnt/d/ProgramData/GEE_PV/tiles/Dortmund_satellite_0/Dortmund_satellite_0_tile_2_8.tif'
    # dst_filename = '/mnt/d/ProgramData/predicted/Dortmund_satellite_0_tile_2_8_mask.tif'
    model_path = './checkpoint/checkpoint_UPerNet_STC_BCEloss_bz16_HRPVS.pt'

    # 加载模型
    net = load_net(model_path)

    # 预测
    # predict_block(src_filename, dst_filename,net=net, device='cuda', threshold=0.5, mask_multiplier=255)

    # print('预测完成！')
    image_dir = '/mnt/d/ProgramData/GEE_PV/tiles/*'
    mask_dir = '/mnt/d/ProgramData/predicted/'
    print("开始预测:")
    predict_batch(image_dir,mask_dir,net=net,device='cuda',threshold=0.5, mask_multiplier=255)
    print("预测完成！")


