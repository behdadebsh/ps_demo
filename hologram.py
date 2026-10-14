# Release script. Read compressed files from hologram_data.
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import pickle
import sys
import time, math

from demo_io import PUBLIC_DATA, load_triangle_surface, validate_public_bundle

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

# === SCENE 1: extracting upper airway + lobe geometries from 3D image ===
# Register 3D image volume
data = np.load(PUBLIC_DATA / 'raw_3D_RAS.npz')
spacing = data['spacing']
arr_img = data['array']
shape = arr_img.shape # in (x,y,z)
swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]

bound_low = np.array((0, 0, 0))
bound_high = np.array((shape[2]*spacing[0], shape[1]*spacing[1], -shape[0]*spacing[2]))
centre = ((bound_low+bound_high)/2) # set target to centre of block (for turntable revolution)

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

# === SCENE 2: tree growing into lobes ===
with (PUBLIC_DATA / "growing.pkl").open("rb") as f:
    frames = pickle.load(f)

# === SCENE 3: lung ventilating
data = np.load(PUBLIC_DATA / 'ventilation.npz')
coordinates = data['acinus']
vols_by_frame = data['dvdt']

# --- CLEAR MEMORY: delete variables not needed after callback ---
del data, arr_img, bound_high, bound_low, lobe
import gc
gc.collect()

# == CALLBACK FUNCTION FOR ANIMATION ==
# SPIN variables
axis = 'Z-axis' # initial camera revolving axis
spin_f = 0 # spin curr frame
n_spin = 500 # spin total frames
spin_radius = shape[1]*spacing[1]
elevation = centre[2]
ps.look_at((centre[0],centre[0]-spin_radius,centre[2]),centre) # not sure y it doesn't automatically centre cam now. add this to centre cam
spin = False

# EXTRACTION variables
ext_f = 0
n_extract = 250
slice_t = np.linspace(-1, shape[1]*spacing[1], n_extract)
slice_t2 = np.linspace(0, -shape[0]*spacing[0], n_extract)
pos = slice_t[0]
pos2 = slice_t2[0]
extract = False

# GROW variables
tree=None
grow_f = 0
n_frames_grow = len(frames)
n_grow = n_frames_grow + 10
transparency = np.linspace(0.5,0,n_frames_grow)
grow = False

# VENTILATION variables
pts=None
n_vent = len(vols_by_frame)
vent_f = 0
vent = False

# Shared variables
existing = False # SHARED variable
existing_plane = False

def callback():
    # vars for Spin
    global spin_f, n_spin, spin_radius, centre, spin, elevation, axis, up_dir
    # vars for Extraction
    global ext_f, n_extract, slice_t, slice_t2, extract, lobes, meshes, swap_arr, shape, spacing
    global meshes_ps, upper_airway, img_block,img_block2
    global cor_plane_pos,cor_plane_neg,ax_plane_neg,ax_plane_pos, pos, pos2
    # vars for Growing
    global grow, grow_f, n_grow, n_frames_grow, frames, tree, transparency, tree
    # vars for Ventilation
    global coordinates, pts, nodes,edges,radius
    global vent, n_vent, update_vent, vent_f, vols_by_frame, pts
    # shared vars
    global existing, existing_plane

    update_spin = False
    update_extract = False
    update_grow = False
    update_vent = False
    vmin = math.floor(np.min(vols_by_frame))
    vmax = math.ceil(np.max(vols_by_frame))

    psim.PushItemWidth(100)
    _, spin = psim.Checkbox("Spin", spin)
    if spin: # Advance the frame
        if extract:
            ps.set_view_projection_mode('orthographic') # switch from default 'perspective' projection to orthographic projection
        elif grow or vent:
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

    psim.Separator()
    if(psim.TreeNode('Planes')):
        slider_cor_plane_neg, pos = psim.SliderFloat("Cor Plane", pos, centre[1]-centre[1],centre[1]+centre[1])
        psim.SameLine()
        slider_ax_plane_neg, pos2 = psim.SliderFloat("Ax Plane", pos2, centre[2]-centre[2],centre[2]+centre[2])
        if slider_cor_plane_neg or slider_ax_plane_neg:
            extract=False # stop autoplay
            ps.remove_last_scene_slice_plane() # Remove prev slice plane
            ps.remove_last_scene_slice_plane() # Remove prev slice plane
            ps.remove_last_scene_slice_plane() # Remove prev slice plane
            ps.remove_last_scene_slice_plane() # Remove prev slice plane

            # Create updated planes
            cor_plane_neg = ps.add_scene_slice_plane()
            cor_plane_neg.set_pose((0., pos+10, 0.), (0., -1., 0.))
            cor_plane_pos = ps.add_scene_slice_plane()
            cor_plane_pos.set_pose((0., pos, 0.), (0., 1., 0.))

            ax_plane_neg = ps.add_scene_slice_plane()
            ax_plane_neg.set_pose((0., 0., pos2+10), (0., 0., -1.))
            ax_plane_pos = ps.add_scene_slice_plane()
            ax_plane_pos.set_pose((0., 0., pos2), (0., 0., 1.))
            existing_plane=True

            if tree is not None:
                tree.set_ignore_slice_plane(ax_plane_neg,True)
                tree.set_ignore_slice_plane(cor_plane_pos,True)
            if pts is not None:
                pts.set_ignore_slice_plane(ax_plane_neg,True)
                pts.set_ignore_slice_plane(cor_plane_pos,True)
            
        psim.TreePop()
    psim.PopItemWidth()
    psim.Separator()

    changed_ext, extract = psim.Checkbox("Extraction", extract)
    if changed_ext:
        ps.remove_all_structures()
        tree=pts=None
        existing = False
        if extract:
            grow=vent=False # make sure other scenes off
            update_grow=update_vent=False
            ps.set_navigation_style('turntable')
    if not extract:
        update_extract = False
    if extract: # SCENE 1 button
        ps.set_view_projection_mode('orthographic') # removes laggy initial transition
        update_extract = True
        ext_f = (ext_f + 1) % n_extract
        pos = slice_t[ext_f]
        pos2 = slice_t2[ext_f]

        if not existing: # at the beginning, register all structures
            existing = True
            meshes_ps = []
            for (lobe,mesh) in zip(lobes,meshes):
                vertices, faces = mesh
                surf = ps.register_surface_mesh(f'{lobe}',vertices,faces,smooth_shade=True, transparency=0.5)
                meshes_ps.append(surf) # store to set ignore planes later

            upper_airway = ps.register_curve_network("upper airway",frames[0]['nodes'],frames[0]['edges'],color=[1.0,0.8,0.8],radius=0.01) # register upper airway
            upper_airway.add_scalar_quantity("radius", frames[0]['radius'], defined_on='nodes')
            upper_airway.set_node_radius_quantity("radius")

            img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0), # register 3D image
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=True)
            img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',enabled=True,cmap='gray')

            img_block2 = ps.register_volume_grid("image2", (shape[2],shape[1],shape[0]),bound_low=(0,0,0), # maybe better to register slice by slice?
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=True)
            img_block2.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',enabled=True,cmap='gray')
    if update_extract: # Animate the plane sliding along the scene
        ps.remove_last_scene_slice_plane() # Remove prev slice plane
        ps.remove_last_scene_slice_plane() # Remove prev slice plane
        ps.remove_last_scene_slice_plane() # Remove prev slice plane
        ps.remove_last_scene_slice_plane() # Remove prev slice plane

        # Create updated planes
        cor_plane_neg = ps.add_scene_slice_plane()
        cor_plane_neg.set_pose((0., pos+10, 0.), (0., -1., 0.))
        cor_plane_pos = ps.add_scene_slice_plane()
        cor_plane_pos.set_pose((0., pos, 0.), (0., 1., 0.))

        ax_plane_neg = ps.add_scene_slice_plane()
        ax_plane_neg.set_pose((0., 0., pos2+10), (0., 0., -1.))
        ax_plane_pos = ps.add_scene_slice_plane()
        ax_plane_pos.set_pose((0., 0., pos2), (0., 0., 1.))
        existing_plane=True

        img_block.set_ignore_slice_plane(ax_plane_neg,True)
        img_block.set_ignore_slice_plane(ax_plane_pos,True)
        img_block2.set_ignore_slice_plane(cor_plane_neg,True)
        img_block2.set_ignore_slice_plane(cor_plane_pos,True)

        for m in meshes_ps: # Set ignore meshes
            m.set_ignore_slice_plane(cor_plane_pos,True)
            m.set_ignore_slice_plane(ax_plane_neg,True)

        upper_airway.set_ignore_slice_plane(cor_plane_pos,True)
        upper_airway.set_ignore_slice_plane(ax_plane_neg,True)
    
    changed_grow, grow = psim.Checkbox("Growing", grow)
    if changed_grow:
        ps.remove_all_structures()
        tree=pts=None
        existing = False
        if grow:
            ps.set_navigation_style('free')
            if existing_plane:
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                existing_plane=False
            extract=vent=False
            update_extract=update_vent=False
        if not grow:
            update_grow = False
    if grow: # SCENE 2 button
        update_grow = True
        grow_f = (grow_f + 1) % n_grow # starts from 1. goes to n_grow-1. then resets to 0.
        time.sleep(0.05)
        
        if not existing:
            ps.set_view_projection_mode('perspective') # not sure if setting it everytime slows it slightly. looks ok for now
            existing = True
            meshes_ps = []
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
            if extract or vent: # if clicked on manual slider while other stuff was activated
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
        ps.remove_all_structures()
        tree=pts=None
        existing = False
        if vent:
            ps.set_navigation_style('free')
            if existing_plane:
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                ps.remove_last_scene_slice_plane()
                existing_plane=False

            extract=grow=False
            update_extract=update_grow=False
        if not vent:
            update_vent = False
    if vent: # SCENE 3 button
        update_vent = True
        vent_f = (vent_f+1) % n_vent
        time.sleep(0.05)
        if not existing:
            ps.set_view_projection_mode("perspective")
            existing=True
            pts = ps.register_point_cloud("tissue units",np.array(coordinates),enabled=True,radius=0.01) # Visualise tissue units
            pts.add_scalar_quantity("volume",vols_by_frame[0],vminmax=(vmin,vmax),enabled=True,cmap='jet')
            pts.set_point_radius_quantity("volume")

            tree = ps.register_curve_network("airways",nodes,edges,color=[1,0.8,0.8],radius=0.018)
            tree.add_scalar_quantity("radius",radius)
            tree.set_node_radius_quantity("radius")

            # Add tidal vol label
            psim.TreeNode("Lung volume")
            psim.TextUnformatted(f"Lung volume: {np.sum(vols_by_frame[vent_f])/10**6:.2f} L")
    if update_vent:
        pts.add_scalar_quantity("volume", vols_by_frame[vent_f],vminmax=(vmin,vmax),enabled=True,cmap='jet')
        pts.set_point_radius_quantity("volume")
        psim.TextUnformatted(f"Lung volume: {np.sum(vols_by_frame[vent_f])/10**6:.2f} L")

ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.set_background_color([0,0,0])
ps.show(1 if "--smoke" in sys.argv else None)
