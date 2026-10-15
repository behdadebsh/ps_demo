# Release script. Read compressed files from hologram_data.
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import pickle
import sys
import time
from pathlib import Path

from demo_io import (
    PUBLIC_DATA,
    load_triangle_surface,
    prepare_ventilation_animation,
    scale_points_about_centre,
    validate_public_bundle,
)
from demo_config import CUSTOM_MULTIMODAL_REGIONS
from multimodal_viewer import MultimodalViewer, RegionSpec


def comparison_specs():
    """Return private configured regions, or a runnable public-data example."""

    if CUSTOM_MULTIMODAL_REGIONS:
        specs = []
        for item in CUSTOM_MULTIMODAL_REGIONS:
            volume = item.get("volume")
            specs.append(
                RegionSpec(
                    name=str(item["name"]),
                    modality=str(item["modality"]).upper() if item.get("modality") else None,
                    volume_path=Path(volume).expanduser().resolve() if volume else None,
                    surface_paths=[
                        Path(path).expanduser().resolve()
                        for path in item.get("surfaces", [])
                    ],
                )
            )
        return specs

    return [
        RegionSpec(
            "Public CT image",
            modality="CT",
            volume_path=PUBLIC_DATA / "raw_3D_RAS.npz",
        ),
        RegionSpec(
            "Public PLY lobes",
            surface_paths=[
                PUBLIC_DATA / f"{lobe}.ply"
                for lobe in ("RUL", "RML", "RLL", "LUL", "LLL")
            ],
        ),
    ]

if "--check" in sys.argv:
    summary = validate_public_bundle()
    print("Public hologram demo data is valid:")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    raise SystemExit(0)

ps.set_print_prefix("ABI Lungs & Respiratory Group 2025\n")
ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")

# Register lobe surface meshes
meshes = []
lobes = ['RUL','RML','RLL','LUL','LLL']
for lobe in lobes:
    meshes.append(load_triangle_surface(PUBLIC_DATA / f'{lobe}.ply'))

# Read in grown airways
data = np.load(PUBLIC_DATA / "tree.npz")
nodes = data["nodes"]
edges = data["edges"]
radius = data["radius"]

# === SCENE 1: tree growing into lobes ===
with (PUBLIC_DATA / "growing.pkl").open("rb") as f:
    frames = pickle.load(f)

# === SCENE 2: lung ventilating ===
data = np.load(PUBLIC_DATA / 'ventilation.npz')
coordinates = data['acinus']
raw_vols_by_frame = data['dvdt']

# Keep the physiological values for the volume label, but exaggerate their
# deviation around the temporal mean so point size and colour read clearly in
# a live presentation. Geometry also expands from 90% to 110% according to the
# measured total-volume breathing phase.
VENTILATION_VALUE_GAIN = 3.0
BREATH_SCALE_MIN = 0.90
BREATH_SCALE_MAX = 1.10
vols_by_frame, raw_total_volume, breath_scale = prepare_ventilation_animation(
    raw_vols_by_frame,
    value_gain=VENTILATION_VALUE_GAIN,
    scale_min=BREATH_SCALE_MIN,
    scale_max=BREATH_SCALE_MAX,
)
ventilation_centre = np.mean(coordinates, axis=0)
vmin, vmax = np.percentile(vols_by_frame, (1.0, 99.0))

# Frame the shared animation camera from the checked-in lobe geometry.
all_lobe_vertices = np.concatenate([vertices for vertices, _faces in meshes], axis=0)
scene_low = np.min(all_lobe_vertices, axis=0)
scene_high = np.max(all_lobe_vertices, axis=0)
centre = (scene_low + scene_high) * 0.5
spin_radius = np.max(scene_high - scene_low) * 1.1

# --- CLEAR MEMORY: delete variables not needed after callback ---
del data, all_lobe_vertices, scene_low, scene_high, lobe
import gc
gc.collect()

# == CALLBACK FUNCTION FOR ANIMATION ==
# SPIN variables
axis = 'Z-axis' # initial camera revolving axis
spin_f = 0 # spin curr frame
n_spin = 500 # spin total frames
elevation = centre[2]
ps.look_at((centre[0],centre[0]-spin_radius,centre[2]),centre) # not sure y it doesn't automatically centre cam now. add this to centre cam
spin = False

# GROW variables
tree=None
meshes_ps=[] # retained across callback frames while the lobes fade
grow_f = 0
n_frames_grow = len(frames)
n_grow = n_frames_grow + 10
transparency = np.linspace(0.5,0,n_frames_grow)
grow = False

# VENTILATION variables
pts=None
n_vent = len(raw_vols_by_frame)
vent_f = 0
vent = False

# Shared variables
existing = False # SHARED variable
comparison = False
comparison_viewer = None


def clear_comparison():
    global comparison_viewer
    if comparison_viewer is not None:
        comparison_viewer.clear_scene()
        comparison_viewer = None


def breathing_positions(points, frame):
    """Expand tissue or airway geometry with the same measured breath phase."""

    return scale_points_about_centre(
        points, ventilation_centre, breath_scale[frame]
    )

def callback():
    # vars for Spin
    global spin_f, n_spin, spin_radius, centre, spin, elevation, axis, up_dir
    # vars for Growing
    global grow, grow_f, n_grow, n_frames_grow, frames, tree, transparency, meshes_ps
    # vars for Ventilation
    global coordinates, pts, nodes,edges,radius
    global vent, n_vent, update_vent, vent_f, vols_by_frame, pts
    # shared vars
    global existing
    global comparison, comparison_viewer

    update_spin = False
    update_grow = False
    update_vent = False

    psim.PushItemWidth(100)
    _, spin = psim.Checkbox("Spin", spin)
    if spin: # Advance the frame
        if grow or vent or comparison:
            ps.set_view_projection_mode('perspective')
        update_spin = True
        spin_f = (spin_f + 1) % n_spin
    if update_spin:
        angle = 2 * np.pi * spin_f / n_spin  # full revolution
        if axis == "Revolve front":
            ps.set_up_dir("z_up")
            ps.set_front_dir("neg_y_front")
            # Revolve around Y axis (not really but this might be worth recording)
            x = centre[0] + spin_radius * np.cos(angle)
            y = elevation  # keep camera elevated along Y
            z = centre[2] + spin_radius * np.sin(angle)
            up_dir = (0,0,1)
        elif axis=='Z-axis':
            ps.set_up_dir("z_up")
            ps.set_front_dir("neg_y_front")
            # Revolve around Z axis (turntable)
            x = centre[0] + spin_radius * np.cos(angle)
            y = centre[1] + spin_radius * np.sin(angle)
            z = elevation
            up_dir = (0,0,1)
        elif axis=='Y-axis':
            ps.set_up_dir("neg_y_up")
            ps.set_front_dir("z_front")
            x = centre[0] + spin_radius * np.cos(angle)
            y = elevation
            z = centre[2] + spin_radius * np.sin(angle)
            up_dir = (0,-1,0)
        camera_pos = (x, y, z)
        # set camera to look at the object
        # ps.look_at(camera_pos,centre,fly_to=True)
        ps.look_at_dir(camera_pos, centre, up_dir=up_dir, fly_to=False) # fly_to must be False or it'll revolve weird
    psim.SameLine()
    axes_options = ["Z-axis", "Y-axis", "Revolve front"]
    changed = psim.BeginCombo("Revolve around", axis)
    if changed:
        for val in axes_options:
            _, selected = psim.Selectable(val, axis == val)
            if selected:
                axis = val
        psim.EndCombo()
    if(psim.TreeNode("Spin Settings")):
        slider_elevation, elevation = psim.SliderFloat("Elevation", elevation, centre[2]-spin_radius, centre[2]+spin_radius)
        psim.SameLine()
        slider_rpm, n_spin = psim.SliderFloat("Speed", n_spin, 200, 700)
        update_spin = update_spin or slider_elevation or slider_rpm
        psim.TreePop()

    psim.PopItemWidth()
    psim.Separator()
    
    changed_grow, grow = psim.Checkbox("Growing", grow)
    if changed_grow:
        clear_comparison()
        comparison = False
        ps.remove_all_structures()
        tree=pts=None
        meshes_ps=[]
        existing = False
        if grow:
            ps.set_navigation_style('free')
            vent=False
            update_vent=False
        if not grow:
            update_grow = False
    if grow: # SCENE 1 button
        update_grow = True
        grow_f = (grow_f + 1) % n_grow # starts from 1. goes to n_grow-1. then resets to 0.
        time.sleep(0.05)
        
        if not existing:
            ps.set_view_projection_mode('perspective') # not sure if setting it everytime slows it slightly. looks ok for now
            existing = True
            for (lobe,mesh) in zip(lobes,meshes):
                vertices, faces = mesh
                surf = ps.register_surface_mesh(f'{lobe}',vertices,faces,smooth_shade=True, transparency=transparency[0],enabled=True)
                meshes_ps.append(surf) # store to set ignore planes later
            
            tree = ps.register_curve_network("upper airway",frames[0]['nodes'],frames[0]['edges'],color=[1.0,0.8,0.8],radius=0.018) # register upper airway
            tree.add_scalar_quantity("radius", frames[0]['radius'], defined_on='nodes')
            tree.set_node_radius_quantity("radius") # nodes become visible in 'orthographic' projection
        
        if grow_f>=n_frames_grow: # once done growing tree, show tissue units
            update_grow = False
            for m in meshes_ps:
                m.set_transparency(0)
            pts = ps.register_point_cloud("tissue units", np.array(coordinates),color=[255/255,50/255,255/255],enabled=True,radius=0.01) # Register terminal units

        if grow_f==0:
            ps.remove_all_structures() # clear memory
            tree=pts=None
            meshes_ps=[]
            existing = False
            update_grow=False
    if update_grow: # Grow tree
        tree = ps.register_curve_network("airways",frames[grow_f]['nodes'],frames[grow_f]['edges'],color=[1.0,0.8,0.8],radius=0.018)
        tree.add_scalar_quantity("radius", frames[grow_f]['radius'], defined_on='nodes', enabled=False)
        tree.set_node_radius_quantity("radius")

        for m in meshes_ps: # meshes fading away
            m.set_transparency(transparency[grow_f])
    if(psim.TreeNode("Manual")):
        slider_grow, grow_f = psim.SliderInt("Current Frame",grow_f,0,n_frames_grow-1)
        if slider_grow:
            if vent or comparison: # avoid replacing another active scene
                psim.TreePop() # close TreeNode to avoid complains
                return # ignore the click or else it'll seg fault
            grow=False # stop autoplay
            for m in meshes_ps:
                m.set_transparency(0) # hide meshes
            tree = ps.register_curve_network("airways",frames[grow_f]['nodes'],frames[grow_f]['edges'],color=[1.0,0.8,0.8],radius=0.018)
            tree.add_scalar_quantity("radius", frames[grow_f]['radius'], defined_on='nodes', enabled=False)
            tree.set_node_radius_quantity("radius")
        psim.TreePop()

    changed_vent, vent = psim.Checkbox("Ventilation",vent)
    if changed_vent:
        clear_comparison()
        comparison = False
        ps.remove_all_structures()
        tree=pts=None
        meshes_ps=[]
        existing = False
        if vent:
            ps.set_navigation_style('free')
            grow=False
            update_grow=False
        if not vent:
            update_vent = False
    if vent: # SCENE 2 button
        update_vent = True
        vent_f = (vent_f+1) % n_vent
        time.sleep(0.05)
        if not existing:
            ps.set_view_projection_mode("perspective")
            existing=True
            pts = ps.register_point_cloud(
                "tissue units",
                breathing_positions(coordinates, vent_f),
                enabled=True,
                radius=0.014,
            )
            pts.add_scalar_quantity("volume",vols_by_frame[vent_f],vminmax=(vmin,vmax),enabled=True,cmap='jet')
            pts.set_point_radius_quantity("volume")

            tree = ps.register_curve_network(
                "airways",
                breathing_positions(nodes, vent_f),
                edges,
                color=[1,0.8,0.8],
                radius=0.018,
            )
            tree.add_scalar_quantity("radius",radius)
            tree.set_node_radius_quantity("radius")

    if update_vent:
        pts.update_point_positions(breathing_positions(coordinates, vent_f))
        tree.update_node_positions(breathing_positions(nodes, vent_f))
        pts.add_scalar_quantity("volume", vols_by_frame[vent_f],vminmax=(vmin,vmax),enabled=True,cmap='jet')
        pts.set_point_radius_quantity("volume")
        psim.TextUnformatted(
            f"Lung volume: {raw_total_volume[vent_f]/10**6:.2f} L"
        )
        psim.TextUnformatted(
            f"Visual emphasis: {VENTILATION_VALUE_GAIN:.0f}x values, "
            f"{breath_scale[vent_f]:.2f}x expansion"
        )

    psim.Separator()
    changed_comparison, comparison = psim.Checkbox("CT / MR / surfaces", comparison)
    if changed_comparison:
        clear_comparison()
        ps.remove_all_structures()
        tree = pts = None
        meshes_ps=[]
        existing = False
        if comparison:
            grow = vent = False
            update_grow = update_vent = False
            ps.set_navigation_style("free")
            ps.set_view_projection_mode("perspective")
            comparison_viewer = MultimodalViewer(
                comparison_specs(), layout="side-by-side"
            )
            comparison_viewer.register_scene()
            ps.reset_camera_to_home_view()
    if comparison and comparison_viewer is not None:
        comparison_viewer.draw_controls(psim, show_title=False)

ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.set_background_color([0,0,0])
if "--smoke-comparison" in sys.argv:
    comparison = True
    comparison_viewer = MultimodalViewer(comparison_specs(), layout="side-by-side")
    comparison_viewer.register_scene()
    ps.reset_camera_to_home_view()
if "--smoke-ventilation" in sys.argv:
    vent = True
if "--smoke-growing" in sys.argv:
    grow = True
smoke_requested = any(
    option in sys.argv
    for option in (
        "--smoke",
        "--smoke-comparison",
        "--smoke-ventilation",
        "--smoke-growing",
    )
)
smoke_frames = 3 if "--smoke-growing" in sys.argv else 1 if smoke_requested else None
ps.show(smoke_frames)
