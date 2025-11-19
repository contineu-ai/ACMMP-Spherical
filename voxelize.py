import open3d as o3d
import numpy as np

def clean_and_voxelize_ply(ply_path, voxel_size=0.05):
    """
    Voxelize and clean a point cloud PLY file while preserving colors
    
    Args:
        ply_path: Path to the PLY file
        voxel_size: Size of each voxel (smaller = more detail, larger file)
    
    Returns:
        cleaned_pcd: Cleaned voxelized point cloud with colors
    """
    # Load the point cloud
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded: {len(pcd.points)} points")
    print(f"Has colors: {pcd.has_colors()}")
    
    # Voxel downsample (this preserves colors automatically)
    print(f"Voxelizing with voxel_size={voxel_size}...")
    voxelized_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"After voxelization: {len(voxelized_pcd.points)} points")
    print(f"Has colors: {voxelized_pcd.has_colors()}")
    
    # Clean the voxelized point cloud
    print("Cleaning voxelized point cloud...")
    
    # Remove statistical outliers
    cleaned_pcd, ind = voxelized_pcd.remove_statistical_outlier(
        nb_neighbors=20, 
        std_ratio=2.0
    )
    print(f"After outlier removal: {len(cleaned_pcd.points)} points")
    
    # Remove radius outliers (removes isolated voxels)
    cleaned_pcd, ind = cleaned_pcd.remove_radius_outlier(
        nb_points=5, 
        radius=voxel_size * 3
    )
    print(f"After radius outlier removal: {len(cleaned_pcd.points)} points")
    print(f"Final has colors: {cleaned_pcd.has_colors()}")
    
    return cleaned_pcd
    


def save_voxel_grid(pcd, output_path):
    """Save cleaned point cloud to file"""
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved cleaned point cloud to {output_path}")

# Example usage
if __name__ == "__main__":
    # Input PLY file path
    input_ply = "/home/kanao/sahil/ACMMP-Spherical/mahindra_af6/ACMMP/ACMM_model_cpp11_compatible.ply"
    
    # Clean and voxelize the point cloud PLY file
    # Smaller voxel_size = more detail (try 0.01-0.1)
    cleaned_pcd= clean_and_voxelize_ply(input_ply, voxel_size=0.001)
    
    # Visualize the cleaned voxelized point cloud
    print("Visualizing cleaned voxelized point cloud...")
    o3d.visualization.draw_geometries([cleaned_pcd])
    
    # Optional: Save the cleaned point cloud
    save_voxel_grid(cleaned_pcd, "output_cleaned_voxels.ply")
    
    # Optional: Get point cloud information
    print(f"\nCleaned voxelized point cloud info:")
    print(f"  Total points: {len(cleaned_pcd.points)}")