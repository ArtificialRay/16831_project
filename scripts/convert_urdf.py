import os
import sys
from pathlib import Path

# PROOF: write a file immediately so we know the script executed
proof_path = Path("/tmp/isaacsim_script_ran.txt")
proof_path.write_text("script started\n")

def log(msg: str):
    # Write to terminal AND to file
    try:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()
    except Exception:
        pass
    with open("/tmp/convert_urdf_log.txt", "a") as f:
        f.write(msg + "\n")

log("SCRIPT STARTED")

try:
    log("Kit is already running (launched via isaacsim --exec). Not creating SimulationApp().")
    log(f"Python exe: {sys.executable}")
    log(f"CWD: {os.getcwd()}")
    log(f"ARGV: {sys.argv}")

    # Use paths relative to script location
    script_dir = Path(__file__).parent.parent.resolve()
    urdf_path = script_dir / "assets/robots/piper_description.urdf"
    out_dir = script_dir / "assets/robots/piper_description"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"URDF: {urdf_path} exists={urdf_path.exists()}")
    log(f"OUT : {out_dir}")

    # Import after Kit is live so pxr/omni are available
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        output_dir=str(out_dir),
    )
    converter = UrdfConverter(cfg)

    usd_path = getattr(converter, "usd_path", None)
    log(f"converter.usd_path = {usd_path}")

    if usd_path and Path(usd_path).exists():
        log(f"✅ USD exists: {usd_path}")
    else:
        log("⚠️ USD not found at converter.usd_path. Listing any USD files in output dir:")
        for p in out_dir.rglob("*.usd"):
            log(f"  found: {p}")
        for p in out_dir.rglob("*.usda"):
            log(f"  found: {p}")
        for p in out_dir.rglob("*.usdc"):
            log(f"  found: {p}")

except Exception as e:
    log(f"EXCEPTION: {repr(e)}")
    raise
finally:
    log("DONE")
