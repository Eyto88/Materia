"""Write a synthetic SEM-style grain micrograph (FEI-style pixel size tag) to sample_data/."""
import os, sys
import numpy as np
import tifffile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labstudio.core.micro_analysis import make_voronoi_grains

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "sample_data")
img, _, _ = make_voronoi_grains((900, 1200), n_grains=180, seed=3, boundary_width=3, noise=8)
bar = np.full((80, img.shape[1]), 20, np.uint8)  # SEM-style data bar
full = np.vstack([img.astype(np.uint8), bar])
ini = "[Scan]\r\nPixelWidth=5e-08\r\nPixelHeight=5e-08\r\n[Beam]\r\nHV=15000\r\n"
tifffile.imwrite(os.path.join(out, "NiTi_SEM_grains_synthetic.tif"), full,
                 extratags=[(34682, "s", 0, ini, True)])
print("written")
