"""Local paths for the consolidated animation + comparison demo.

Leave this list empty to run entirely from ``hologram_data``. When private
files are available, add one dictionary per independent layout region. STL and
PLY surfaces can be mixed in the same region.
"""

CUSTOM_MULTIMODAL_REGIONS = [
    # {
    #     "name": "CT",
    #     "modality": "CT",
    #     "volume": r"C:\path\to\ct.nii.gz",  # or a DICOM directory
    #     "surfaces": [
    #         r"C:\path\to\ct_lung.ply",
    #         r"C:\path\to\ct_airway.stl",
    #     ],
    # },
    # {
    #     "name": "MRI",
    #     "modality": "MR",
    #     "volume": r"C:\path\to\mri.nii.gz",
    #     "surfaces": [r"C:\path\to\mri_lung.ply"],
    # },
]
