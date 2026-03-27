import bpy
import numpy as np
import os

save_dir = "/home/tanmay/thesis/UniPhy/save_dir/interacting_custom_high_lr_updated_network/"
TRAJ_PATH = os.path.join(save_dir, "trajectory.npy")
MAT_IDS_PATH = os.path.join(save_dir, "material_ids.npy")
OUTPUT_PATH = "/home/tanmay/thesis/renders/render_colored_new_viewpt_custom_high_lr_updated_network.mp4"

# LOAD DATA
traj = np.load(TRAJ_PATH)        # (T, P, 3)
mat_ids = np.load(MAT_IDS_PATH)  # (P,)

T, P, _ = traj.shape
num_materials = int(mat_ids.max()) + 1

print(f"Loaded trajectory: {traj.shape}")
print(f"Loaded material IDs: {mat_ids.shape}, num materials = {num_materials}")

# CLEAR SCENE
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# CREATE MESH
mesh = bpy.data.meshes.new("ParticlesMesh")
obj = bpy.data.objects.new("Particles", mesh)
bpy.context.collection.objects.link(obj)

bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# initial frame
verts = traj[0].tolist()
mesh.from_pydata(verts, [], [])
mesh.update()

# ADD MATERIAL ID ATTRIBUTE
if "material_id" not in mesh.attributes:
    attr = mesh.attributes.new(name="material_id", type='FLOAT', domain='POINT')

for i in range(P):
    mesh.attributes["material_id"].data[i].value = float(mat_ids[i])

# FRAME UPDATE
def update_particles(scene):
    frame = min(scene.frame_current, T - 1)
    positions = traj[frame]

    mesh.clear_geometry()
    mesh.from_pydata(positions.tolist(), [], [])

    # re-add attribute after geometry reset
    attr = mesh.attributes.new(name="material_id", type='FLOAT', domain='POINT')
    for i in range(P):
        attr.data[i].value = float(mat_ids[i])

    mesh.update()

bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(update_particles)

# SCENE SETTINGS
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = T - 1

# CAMERA
cam_data = bpy.data.cameras.new("Camera")
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

import mathutils

# Center of your scene (compute from trajectory)
center = traj.mean(axis=(0, 1))  # (3,)

cam.location = (
    center[0] - 0.75,
    center[1] - 1.5,
    center[2] + 1.5
)

# Make camera look at center
direction = mathutils.Vector(center) - cam.location
rot_quat = direction.to_track_quat('-Z', 'Y')
cam.rotation_euler = rot_quat.to_euler()

# LIGHT
light_data = bpy.data.lights.new(name="light", type='POINT')
light = bpy.data.objects.new(name="light", object_data=light_data)
bpy.context.collection.objects.link(light)
light.location = (4, -4, 6)

# CREATE MATERIAL
mat = bpy.data.materials.new(name="ParticleMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

nodes.clear()

output = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
attr_node = nodes.new("ShaderNodeAttribute")

attr_node.attribute_name = "material_id"
attr_node.attribute_type = 'INSTANCER'

# Map material_id → color
color_ramp = nodes.new("ShaderNodeValToRGB")

# Setup colors (edit these if you want)
colors = [
    (0.2, 0.4, 1.0, 1.0),  # blue
    (1.0, 0.3, 0.3, 1.0),  # red
    (0.3, 1.0, 0.3, 1.0),  # green
    (1.0, 1.0, 0.3, 1.0),  # yellow
]

ramp = color_ramp.color_ramp

# Ensure correct number of elements
while len(ramp.elements) < num_materials:
    ramp.elements.new(1.0)

# If too many, remove extras (but keep at least 1)
while len(ramp.elements) > num_materials:
    ramp.elements.remove(ramp.elements[-1])

# Assign positions + colors
for i, elem in enumerate(ramp.elements):
    elem.position = i / max(1, num_materials - 1)
    elem.color = colors[i % len(colors)]

# link nodes
links.new(attr_node.outputs["Fac"], color_ramp.inputs["Fac"])
links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])
links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

# GEOMETRY NODES SETUP
geo_mod = obj.modifiers.new(name="ParticlesGN", type='NODES')

node_group = bpy.data.node_groups.new("ParticlesGN", "GeometryNodeTree")
geo_mod.node_group = node_group

nodes = node_group.nodes
links = node_group.links
nodes.clear()

# nodes
group_in = nodes.new("NodeGroupInput")
group_out = nodes.new("NodeGroupOutput")
node_group.inputs.new("NodeSocketGeometry", "Geometry")
node_group.outputs.new("NodeSocketGeometry", "Geometry")

mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
ico_sphere = nodes.new("GeometryNodeMeshIcoSphere")
set_material = nodes.new("GeometryNodeSetMaterial")

ico_sphere.inputs["Radius"].default_value = 0.01

# layout
group_in.location = (-400, 0)
mesh_to_points.location = (-200, 0)
instance_on_points.location = (0, 0)
ico_sphere.location = (0, -200)
set_material.location = (200, 0)
group_out.location = (400, 0)

# links
capture_attr = nodes.new("GeometryNodeCaptureAttribute")
capture_attr.location = (-50, 0)
capture_attr.data_type = 'FLOAT'
capture_attr.domain = 'POINT'
# capture_attr.inputs[1].default_value = 0.0  # will override via field

# Named attribute node (to read material_id)
named_attr = nodes.new("GeometryNodeInputNamedAttribute")
named_attr.location = (-250, -200)
named_attr.data_type = 'FLOAT'
named_attr.inputs["Name"].default_value = "material_id"

# connections
links.new(group_in.outputs[0], mesh_to_points.inputs[0])
links.new(mesh_to_points.outputs[0], capture_attr.inputs[0])
links.new(named_attr.outputs[0], capture_attr.inputs[1])
links.new(capture_attr.outputs[0], instance_on_points.inputs[0])
links.new(ico_sphere.outputs[0], instance_on_points.inputs[2])
links.new(instance_on_points.outputs[0], set_material.inputs[0])
links.new(set_material.outputs[0], group_out.inputs[0])

# assign material
set_material.inputs["Material"].default_value = mat

# RENDER SETTINGS
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'

scene.render.filepath = OUTPUT_PATH
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'

scene.render.resolution_x = 1280
scene.render.resolution_y = 1280

# RENDER
print("Starting render...")
bpy.ops.render.render(animation=True)
print("Done!")