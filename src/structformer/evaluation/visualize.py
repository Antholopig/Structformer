import open3d as o3d
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from src.structformer.utils import load_json_file

def visualize(point_path:str,save_path:str,scene_path:str,interact:bool=False):
    # 读取点云
    pointcloud=o3d.io.read_point_cloud(point_path)
    to_visualize = [pointcloud]

    # 从文件中读取桌子和坐标系，并与点云一起显示

    loaded_scene =None
    if os.path.exists(scene_path):
        loaded_scene = o3d.io.read_triangle_mesh(scene_path)
        to_visualize.append(loaded_scene)
        
    if args.interact:
        o3d.visualization.draw_geometries(to_visualize) 
    else:
        # 创建可视化对象
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)  # 不显示窗口
        vis.add_geometry(pointcloud)  # 添加点云
        vis.add_geometry(loaded_scene)  # 添加加载的桌子和坐标系

        # 设置俯视图的摄像机参数
        ctr = vis.get_view_control()
        # 获取当前窗口大小
        #window_width, window_height = vis.get_render_option().width, vis.get_render_option().height

        parameters = ctr.convert_to_pinhole_camera_parameters()
        # 创建一个新的 extrinsic 矩阵并设置俯视视角
        new_extrinsic = np.eye(4)  # 初始化为单位矩阵
        new_extrinsic[:3, 3] = [0, 0, 2]  # 设置相机位置 (x=0, y=0, z=2)
        new_extrinsic[:3, :3] = np.eye(3)  # 设置视角垂直向下
        parameters.extrinsic = new_extrinsic
        ctr.convert_from_pinhole_camera_parameters(parameters)

        # 保存俯视图
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()

        print(f"俯视图已保存到 {save_path}")


def compare_visualize(original_path:str, modified_path:str,index_path:str,visual_id:str,instruction=True):
    # Load the images using cv2
    origin_image = cv2.imread(original_path)
    current_image = cv2.imread(modified_path)
    origin_image = origin_image[50:-50,300:-300,:]
    current_image = current_image[50:-50,300:-300,:]
    if instruction:
        index_file = load_json_file(index_path)
        natural_language_instruction = index_file[visual_id]
    # Get image dimensions
    height, width, _ = origin_image.shape

    # Create a blank canvas that is twice the width and taller to include text below the images
    text_height = 50  # Height for the text area
    canvas_height = height + text_height
    canvas_width = width * 2
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White canvas

    # Place the original and modified images on the canvas
    canvas[:height, :width, :] = origin_image
    canvas[:height, width:width*2, :] = current_image

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # Adjust this value for font size
    font_thickness = 5
    color = (0, 0, 0)  # Black color for text

    # Calculate text positions (centered below each image)
    before_text_position = (width // 2 - 50, height + 30)  # x, y for "before"
    after_text_position = (width + width // 2 - 50, height + 30)  # x, y for "after"
    natural_instruction_position = (0, 30)

    # Add text labels
    cv2.putText(canvas, "before", before_text_position, font, font_scale, color, font_thickness)
    cv2.putText(canvas, "after", after_text_position, font, font_scale, color, font_thickness)
    if instruction:
        cv2.putText(canvas, natural_language_instruction, natural_instruction_position, font, font_scale, color, font_thickness)

    # Save the resulting image
    output_path = modified_path.replace(".png", "_compare.png")
    cv2.imwrite(output_path, canvas)
    print(f"Comparison image saved at: {output_path}")

def compare_pointclouds(original_path, modified_path):
    """
    比较原版点云和修改后的点云差异
    """
    # 读取点云
    original_pcd = o3d.io.read_point_cloud(original_path)
    modified_pcd = o3d.io.read_point_cloud(modified_path)
    
    # 获取点和颜色
    original_points = np.asarray(original_pcd.points)
    modified_points = np.asarray(modified_pcd.points)
    original_colors = np.asarray(original_pcd.colors)
    modified_colors = np.asarray(modified_pcd.colors)

    # 比较点的差异
    point_diff = np.linalg.norm(original_points - modified_points, axis=1)
    avg_point_diff = np.mean(point_diff)  # 平均差异
    max_point_diff = np.max(point_diff)  # 最大差异

    # 比较颜色的差异（如果有）
    if original_colors.size > 0 and modified_colors.size > 0:
        color_diff = np.linalg.norm(original_colors - modified_colors, axis=1)
        avg_color_diff = np.mean(color_diff)
        max_color_diff = np.max(color_diff)
    else:
        avg_color_diff = max_color_diff = None

    print(f"点云差异:")
    print(f"平均点距离: {avg_point_diff}")
    print(f"最大点距离: {max_point_diff}")
    if avg_color_diff is not None:
        print(f"平均颜色差异: {avg_color_diff}")
        print(f"最大颜色差异: {max_color_diff}")
    else:
        print("点云没有颜色数据。")



def compress_compare_images(source_folder, output_zip):
    """
    查找指定文件夹下所有以 '_compare.png' 结尾的文件，并压缩到一个 ZIP 文件中。
    
    :param source_folder: 需要查找的文件夹路径
    :param output_zip: 输出压缩文件的路径
    """
    import zipfile
    import os
    from pathlib import Path
    source_folder = Path(source_folder)
    output_zip = Path(output_zip)

    # 检查文件夹是否存在
    if not source_folder.exists():
        print(f"错误: 文件夹 '{source_folder}' 不存在！")
        return

    # 查找所有结尾为 '_compare.png' 的文件
    compare_files = list(source_folder.rglob("*_compare.png"))

    if not compare_files:
        print(f"在文件夹 '{source_folder}' 中没有找到任何 '_compare.png' 文件。")
        return

    # 创建 ZIP 文件并添加文件
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in compare_files:
            # 将文件添加到 ZIP 中，保留相对路径
            arcname = file.relative_to(source_folder)
            zipf.write(file, arcname)
            print(f"已添加: {file}")

    print(f"成功压缩 {len(compare_files)} 个文件到 '{output_zip}' 中。")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize the point cloud")
    parser.add_argument("--visual_fold", 
                        default="visualize",
                        help='the show path of the point cloud', type=str)
    
    parser.add_argument("--visual_id", 
                        default="00553435",
                        help='the show path of the point cloud', type=str)
    parser.add_argument("--view", 
                        default="top",
                        help='from which view to look at the point cloud', type=str)
    parser.add_argument("--interact", type=bool,default=False,
                        help='from which view to look at the point cloud')
    parser.add_argument("--origin", type=bool,default=False,
                    help='visualize origin photo')
    parser.add_argument("--no_instruction", type=bool,default=False,
                    help='visualize origin photo')


    args = parser.parse_args()
    #compress_compare_images(args.visual_fold,"./compressed.zip")
    #exit()
    visual_fold = args.visual_fold
    visual_id = args.visual_id
    visual_ids = []
    if visual_id=="":
        ply_files = list(Path(visual_fold).glob("*.ply"))
        if not ply_files:
            exit()
        visual_ids = [ply_file.stem for ply_file in ply_files if "_original" not in str(ply_file)]
    else:
        visual_ids = [visual_id]
    for visual_id in visual_ids:
        point_path = os.path.join(visual_fold,f"{visual_id}.ply")
        save_path = os.path.join(visual_fold,f"{visual_id}.png")
        scene_path = os.path.join(visual_fold,f"{visual_id}.obj")
        index_path = os.path.join(visual_fold,"index.json")

        visualize(point_path=point_path,save_path=save_path,scene_path=scene_path)
        
        if args.origin:
            origin_point_path = point_path.replace(".ply","_original.ply")
            origin_save_path = origin_point_path.replace(".ply",".png")
            origin_scene_path = origin_point_path.replace(".ply",".obj")
            visualize(point_path=origin_point_path,save_path=origin_save_path,scene_path=origin_scene_path)
            #compare_pointclouds(point_path,args.show_path)
            compare_visualize(origin_save_path,save_path,index_path,visual_id,not args.no_instruction)
            
    