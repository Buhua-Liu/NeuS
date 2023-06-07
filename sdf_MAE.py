import open3d as o3d
from pysdf import SDF
from sklearn.metrics import mean_absolute_error
from models.fields import SDFNetwork
import torch

# Read ground-truth PointCloud
stl_pcd = o3d.io.read_point_cloud('Points/stl/stl065_total.ply')
# Compute a triangle mesh from a PointCloud
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(stl_pcd)
f_gt=SDF(poisson_mesh[0].vertices, poisson_mesh[0].triangles)

# Uniform surface point sampling on GT surface
random_surface_points = f_gt.sample_surface(100000)

# # Load reconstructed mesh
# data_mesh = o3d.io.read_triangle_mesh('exp/dtu_scan65/wmask/meshes/00300000.ply')
# f=SDF(data_mesh.vertices, data_mesh.triangles)

# mae = mean_absolute_error(f_gt(random_surface_points), f(random_surface_points)) # 108.21
# print("MAE:", mae)

device = torch.device('cuda')

sdf_network = SDFNetwork(d_out = 257,
        d_in = 3,
        d_hidden = 256,
        n_layers = 8,
        skip_in = [4],
        multires = 6,
        bias = 0.5,
        scale = 1.0,
        geometric_init = True,
        weight_norm = True
        ).to(device)

model_path = 'exp/dtu_scan65/wmask/checkpoints/ckpt_300000.pth'
checkpoint = torch.load(model_path, map_location=device)
sdf_network.load_state_dict(checkpoint['sdf_network_fine'])

mae = mean_absolute_error(f_gt(random_surface_points), sdf_network.sdf(torch.tensor(random_surface_points, device=device)).cpu().detach().numpy())
print("MAE:", mae) # 173.03


