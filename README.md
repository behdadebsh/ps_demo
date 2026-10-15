# ps_demo

Scripts to visualise and animate lung models with [Polyscope](https://polyscope.run/py/).
All demos now run with the public compressed data in `hologram_data`. Where the
original private `ps_demo_data` files are available, the legacy scripts continue
to prefer those source files.

## Setup

```bash
git clone https://github.com/behdadebsh/ps_demo.git
cd ps_demo
python -m venv venv-ps
```

Activate the environment and install the dependencies:

```bash
# macOS or Linux
source venv-ps/bin/activate

# Windows PowerShell
venv-ps\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file lists only the four direct dependencies. PyVista and VTK
are not required; STL/PLY files are loaded with Trimesh. This avoids the
platform- and Python-version-specific VTK wheel conflict in the old frozen
requirements file.

Before opening a window, verify the installation and public data:

```bash
python multimodal_viewer.py --check
python hologram.py --check
```

If an older environment already contains the previous pinned VTK/PyVista
packages, create a new virtual environment rather than trying to repair that
environment in place.

## CT/MR and surface comparison demo

`multimodal_viewer.py` loads any number of named regions. A region may contain
one CT or MR image and one or more STL/PLY surfaces. PLY is a first-class input:
use `--ply`, `--surface`, or the existing `--stl` spelling for either mesh
format. Every region and surface
has a unique Polyscope structure name, so files with the same basename do not
replace each other.

Side-by-side mode is the default. It translates complete named regions along
the X axis so they do not overlap; surfaces assigned to `CT` or `MRI` move with
that image. Switch the **Layout** control to `overlay` to restore every file's
native patient/world coordinates. CT and MR visibility, the three anatomical
slices, slice indices, individual surfaces, and surface opacity are all
controlled independently in the demo panel.

Load NIfTI, MHA, or another SimpleITK-readable volume file:

```bash
python multimodal_viewer.py \
  --ct "CT=data/ct.nii.gz" \
  --mr "MRI=data/mri.nii.gz" \
  --ply "CT=data/ct_lung.ply" \
  --stl "MRI=data/mr_lung.stl"
```

A CT or MR path may also be a DICOM series directory. Repeat `--stl` (or its
alias `--surface`) to add more surfaces. Use a new name to give a surface its
own layout slot:

```bash
python multimodal_viewer.py \
  --ct "CT=data/ct_dicom" \
  --mr "MRI=data/mr_dicom" \
  --stl "Airway=data/airway.stl" \
  --layout side-by-side
```

For a reproducible demo, put the configuration in JSON. Paths are resolved
relative to the manifest:

```json
{
  "layout": "side-by-side",
  "regions": [
    {
      "name": "CT",
      "modality": "CT",
      "volume": "data/ct.nii.gz",
      "surfaces": ["data/ct_lung.stl", "data/airway.stl"]
    },
    {
      "name": "MRI",
      "modality": "MR",
      "volume": "data/mri.nii.gz",
      "surfaces": ["data/mr_lung.stl"]
    }
  ]
}
```

Run it with:

```bash
python multimodal_viewer.py --manifest path/to/demo.json
```

Running `python multimodal_viewer.py` without arguments opens the public lung
PLY surfaces already included in this repository.

## Consolidated animation and comparison demo

`hologram.py` contains the **Growing**, **Ventilation**, and **Perfusion**
animations plus a **CT / MR / surfaces** scene. The image/surface scene embeds the same
comparison controls described above, including
side-by-side/overlay layout, per-image visibility, slice selection, and
per-surface visibility and opacity. The old Surprise scene remains removed.

Ventilation keeps the original simulated lung volume for the displayed litre
value, while using a 3x deviation gain for point colour/radius and a
data-driven 0.90x to 1.10x tissue expansion. This makes inspiration and
expiration readable during a presentation without relabelling the amplified
visual as a physiological measurement. These presentation constants are near
the top of `hologram.py` if a subtler or stronger effect is preferred.

## Disease-case perfusion

The **Perfusion** scene displays a stationary pulmonary arterial tree using
the supplied element radii. Its 30,933 terminal points animate through the 50
samples of one cardiac cycle: colour shows signed terminal flow and point size
shows flow magnitude. Neither the vessel nodes nor the terminal coordinates
move during the cycle. Because the source table has no timestamps, the UI
reports normalized cardiac phase from 0% to 100% rather than inventing a time
unit.

Terminal flow is mapped by connectivity, not by assuming that node and element
numbers match. Each terminal node is associated with its single incident
vessel element, then that element's row is selected from the time-dependent
flow table. As a validation, the mean of every terminal waveform is compared
with the supplied steady-state terminal flow when the dataset is built.

The public repository contains only the anonymized, derived
`hologram_data/pulmonary_perfusion_disease_case.npz`. It retains the complete
tree geometry, element radii, terminal mapping, terminal pressure, steady flow,
and the supplied terminal flow samples, but excludes the original filenames
and raw text files. The original flow units are preserved rather than guessed
or relabeled.

For visibility, the viewer uses the square root of radius to draw distal
vessels while retaining the original radius array as the **vessel radius**
quantity. Rebuild the anonymized file from equivalent EX/text inputs with:

```bash
python perfusion_io.py \
  --nodes data/vessel_tree.exnode \
  --elements data/vessel_tree.exelem \
  --radii data/vessel_radius.exelem \
  --terminals data/terminal_response.exnode \
  --flows data/element_flows.txt \
  --output hologram_data/pulmonary_perfusion_disease_case.npz
```

The consolidated scene runs immediately with the checked-in CT NPZ and five
PLY lobe files. To use private data later, edit only
`CUSTOM_MULTIMODAL_REGIONS` in `demo_config.py`. Add one dictionary per layout
slot; files listed in the same dictionary move together, while separate
dictionaries are kept apart in side-by-side mode. CT and MR may point to a
SimpleITK-readable image or DICOM directory, and `surfaces` may freely mix STL
and PLY paths.

```python
CUSTOM_MULTIMODAL_REGIONS = [
    {
        "name": "CT",
        "modality": "CT",
        "volume": r"C:\data\ct.nii.gz",
        "surfaces": [r"C:\data\ct_lung.ply", r"C:\data\airway.stl"],
    },
    {
        "name": "MRI",
        "modality": "MR",
        "volume": r"C:\data\mri.nii.gz",
        "surfaces": [r"C:\data\mri_lung.ply"],
    },
]
```

Run the combined application with `python hologram.py`, then select
**CT / MR / surfaces** in the Polyscope controls.

## Bringing the demo into LungViZ

Treat this script as the interaction prototype, not as a library dependency.
LungViZ already loads every STL/PLY selection as an isolated geometry region,
so that part should stay in its existing `RegionState` model. The reusable
change is to replace LungViZ's single `ct_volume`/`ct_slices` slot with a list
of medical-image region states containing a modality label, volume, slice
states, visibility, and layout transform. Its NIfTI and DICOM readers can then
return a modality-neutral `MedicalVolume` (keeping `CTVolume` as a compatibility
alias if needed).

The side-by-side offset calculation in this demo can be moved into LungViZ as a
small layout helper. Apply the resulting translation to the medical-image
slices and to any geometry region explicitly assigned to the same group. Keep
`overlay` as the identity transform so registered CT, MR, and segmentations
remain in patient/world coordinates. This avoids making LungViZ import code
from a demo repository and lets the same state model support more modalities
later.

## Included demos

Each command below runs with the checked-in public data. The animation and mesh
scripts transparently use equivalent arrays from `hologram_data` when the
original private files are absent.

```bash
python hologram.py
python animate_breath.py
python animate_grow_tree.py
python animate_grow_tree_mesh.py
python visualise_model.py
python visualise_tree_mesh.py
```

To build `polyscope-py` from source instead of using the published package:

```bash
git clone https://github.com/nmwsharp/polyscope-py.git
cd polyscope-py
git submodule update --init --recursive
pip install .
```

Run the non-graphical demo tests with:

```bash
python -m unittest discover -s tests -v
```

Maintainers can also perform a one-frame graphical smoke test without leaving a
window open:

```bash
python multimodal_viewer.py --smoke
python hologram.py --smoke
python hologram.py --smoke-growing
python hologram.py --smoke-ventilation
python hologram.py --smoke-perfusion
python hologram.py --smoke-comparison
python animate_breath.py --smoke
python animate_grow_tree.py --smoke
python animate_grow_tree_mesh.py --smoke
python visualise_model.py --smoke
python visualise_tree_mesh.py --smoke
python cam_control.py --smoke
```
