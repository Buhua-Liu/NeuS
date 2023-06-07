import pyrender
import os
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
from pyrender.constants import RenderFlags

import trimesh
import trimesh.creation
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def inter_pose(cameras, cam_id, cam_id_new, ratio):
    intrinsics, pose_0 = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id)] @ cameras['scale_mat_0'.format(cam_id)])[:3, :4])
    intrinsics, pose_1 = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id_new)] @ cameras['scale_mat_0'.format(cam_id)])[:3, :4])
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    key_rots = [rot_0, rot_1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    # pose[:3, 3] = (pose_0 * (1.0 - 0.5) + pose_1 * 0.5)[:3, 3]
    pose[:3, 3] = (pose_0 * (1.0 - ratio) + pose_1 * ratio)[:3, 3]
    pose = np.linalg.inv(pose @ np.linalg.inv(cameras['scale_mat_0'.format(cam_id)]))
    return pose



def visualize(scan, cam_id, cam_id_new=-1, ratio=-1):
    model_trimesh.visual.vertex_colors = np.ones_like(model_trimesh.vertices) * (np.array([255, 255, 255]) / 255.0)
    model_center = model_trimesh.vertices.mean(axis=0)

    cameras = np.load('./data/{}/cameras_sphere.npz'.format(scan))

    model = Mesh.from_trimesh(model_trimesh)

    direc_l = DirectionalLight(color=np.ones(3), intensity=3.0)
    direc_l_2 = DirectionalLight(color=np.ones(3), intensity=1.0)
    intrinsics, pose = load_K_Rt_from_P(None, (cameras['world_mat_{}'.format(cam_id)])[:3, :4])

    if cam_id_new >= 0:
         pose = inter_pose(cameras, cam_id, cam_id_new, ratio)
    print(intrinsics)

    fake_H = intrinsics[1, 2] * 2.0
    fake_W = intrinsics[0, 2] * 2.0

    yfov = np.arctan((fake_H / 2.0) / intrinsics[1, 1] * ext) * 2.0

    print(yfov)
    print(pose)

    cam = PerspectiveCamera(yfov=yfov)
    cam_pose = pose
    cam_pose = np.linalg.inv(cam_pose)
    cam_pose = np.linalg.inv(cam_pose)
    cam_pose = cam_pose @ np.diag([1.0, -1.0, -1.0, 1.0])

    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=np.array([1.0, 1.0, 1.0]))
    node = Node(mesh=model, translation=np.array([0.0, 0.0, 0.0]))
    scene.add_node(node)

    # light pose
    model_pos = model_center[:]
    cam_pos = pose[:3, 3]
    offset = (model_pos - cam_pos)
    dis = np.sqrt((offset**2).sum())
    pro = np.linalg.inv(pose[:3, :3])
    cam_x_vector, cam_up_vector, cam_z_vector = pro[0], pro[1], pro[2]

    light_pos = model_pos - cam_up_vector * dis
    light_z_vector = -cam_up_vector
    light_x_vector = cam_x_vector
    light_y_vector = cam_z_vector
    light_pro = np.stack([light_x_vector, light_y_vector, light_z_vector], axis=0)
    light_pose = np.diag([1.0, 1.0, 1.0, 1.0])
    light_pose[:3, :3] = np.linalg.inv(light_pro)
    light_pose[:3, 3] = light_pos

    direc_l_node = scene.add(direc_l, pose=light_pose[:])
    direc_l_node_2 = scene.add(direc_l_2, pose=cam_pose)

    cam_node = scene.add(cam, pose=cam_pose)
    # v = Viewer(scene)

    r = OffscreenRenderer(viewport_width=fake_W * ext, viewport_height=fake_H * ext)
    color, depth = r.render(scene, flags=RenderFlags.OFFSCREEN)
    return color, depth, fake_H, fake_W


# --------------------------
scan_id = 65

H = 1200
W = 1600
ext = 1.5

model_trimesh = trimesh.load(f'exp/dtu_scan{scan_id}/wmask/meshes/00300000.ply')


for i in range(0, 60):
    start_idx = 9
    end_idx = 10
    image, depth, fake_H, fake_W = visualize(scan=f'dtu_scan{scan_id}', cam_id=start_idx, cam_id_new=end_idx, ratio=(np.sin(((i / 60) - 0.5) * np.pi) + 1.0) * 0.5)
    # image, depth, fake_H, fake_W = visualize()

    image = image[int(fake_H * ext * 0.5 - fake_H * 0.5): int(fake_H * ext * 0.5 - fake_H * 0.5) + H, int(fake_W * ext * 0.5 - fake_W * 0.5): int(fake_W * ext * 0.5 - fake_W * 0.5) + W]
    depth = depth[int(fake_H * ext * 0.5 - fake_H * 0.5): int(fake_H * ext * 0.5 - fake_H * 0.5) + H, int(fake_W * ext * 0.5 - fake_W * 0.5): int(fake_W * ext * 0.5 - fake_W * 0.5) + W]
    image = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)
    depth[np.where(depth < 1e-5)] = depth.max()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth = depth.astype(np.uint8)

    depth = cv.applyColorMap(depth[:, :, None], cv.COLORMAP_WINTER)
    cv.imwrite('./vis_results/tmp_depth.png', depth)
    cv.imwrite('./vis_results/video/{}.png'.format(i), image)