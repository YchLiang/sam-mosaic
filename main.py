from sam_mosaic import segment_with_params
import os


if __name__ == "__main__":

    def main():
        imput_path = "./tif/ArcGIS_卫星_z18_20260612_085913.tif"
        output_dir = "./result/ArcGIS_卫星_z18_20260612_085913_param_optmized2/"
        os.makedirs(output_dir, exist_ok=True)
        
        """Run segmentation with custom parameters."""
        result = segment_with_params(
            input_path=imput_path,
            output_dir=output_dir,
            tile_size= 1000,
            checkpoint="./checkpoints/sam2.1_hiera_large.pt",
            roi_mask=None,  # 可选：提供 SHP 文件路径以限制分割区域
            # === 速度优化 ===
            max_passes=50,           # 原 None。限制最大 pass 数，避免过多无效 pass 导致时间过长
            target_coverage=95.0,    # 原 99.0。高覆盖 tile 提前退出
            points_per_side=48,      # 原 64。减少约 44% 点数，单 pass 加速，但不太影响 mask 产出
            threshold_step=0.02,     # 原 0.01。阈值下降快一倍，减少无效空 pass（0 mask 的 pass）
        )

        print(f"\nResults:")
        print(f"  Segments: {result.n_segments}")
        print(f"  Coverage: {result.coverage:.1f}%")
        print(f"  Time: {result.processing_time:.1f}s")
        print(f"  Labels: {result.labels_path}")
        print(f"  Shapefile: {result.shapefile_path}")

    main()