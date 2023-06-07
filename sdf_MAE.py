import open3d as o3d
from pysdf import SDF
from sklearn.metrics import mean_absolute_error

# Read ground-truth PointCloud
stl_pcd = o3d.io.read_point_cloud('Points/stl/stl065_total.ply')
# Compute a triangle mesh from a PointCloud
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(stl_pcd)
f_gt=SDF(poisson_mesh[0].vertices, poisson_mesh[0].triangles)

# Uniform surface point sampling on GT surface
random_surface_points = f_gt.sample_surface(100000)

# Load reconstructed mesh
data_mesh = o3d.io.read_triangle_mesh('exp/dtu_scan65/wmask/meshes/00300000.ply')
f=SDF(data_mesh.vertices, data_mesh.triangles)


mae = mean_absolute_error(f_gt(random_surface_points), f(random_surface_points))
print("MAE:", mae)


