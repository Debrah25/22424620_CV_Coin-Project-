# move heic files from the phone into data/raw_originals then convert+rename into data/raw as jpg.
# only needs to be run once. uses sips (built into macOS) to do the heic->jpg conversion.

import subprocess
import shutil
from pathlib import Path

RAW = Path("data/raw")
ORIG = Path("data/raw_originals")

# old filename -> new clean name (category matches assignment brief)
RENAME_MAP = {
    "A1_baseline_3same.heic": "A1_baseline.jpg",
    "A2_baseline_5same.heic": "A2_baseline.jpg",
    "A3_baseline_3different_spaceout.heic": "A3_baseline.jpg",
    "A4_baseline_5mixed.heic": "A4_baseline.jpg",

    "B6_Illum_Dim indoor light.HEIC": "B1_dim.jpg",
    "B7_Illum_Harsh direct sunlight.heic": "B2_harsh.jpg",
    "B8_Illum_Side-lit.heic": "B3_sidelit.jpg",
    "B9_Illum_yellow:warm bulb.heic": "B4_warmbulb.jpg",
    "B10_Illum_Flash On.heic": "B5_flash.jpg",

    "C11_backvar_wood-grain surface.heic": "C1_wood.jpg",
    "C12_backvar_Patterned cloth.heic": "C2_cloth.jpg",
    "C13_backvar_Dark surface.heic": "C3_dark.jpg",
    "C14_backvar_printed page.heic": "C4_newspaper.jpg",

    "D15_touch_3 coins touching edge-to-edge in a row.heic": "D1_touch.jpg",
    "D16_Touch_edges touching.heic": "D2_touch.jpg",
    "D17_touch_coinoverallap.HEIC": "D3_touch.jpg",
    "D18_Touch_mixed case.heic": "D4_touch.jpg",

    "E19_sacelvar_coins fill frame.heic": "E1_close.jpg",
    "E20_scalevar_Camera-at-normal.heic": "E2_normal.jpg",
    "E21_Scalevar_camera-far.heic": "E3_far.jpg",

    "F22_Advers_Slightly out of focus.heic": "F1_outoffocus.jpg",
    "F23_Adver_Motion blur.heic": "F2_motionblur.jpg",
    "f24_Advers_Clutter Backg.heic": "F3_cluttered.jpg",
    "F24_Adver_Verypoorlights.heic": "F4_verydark.jpg",
}

# keep a note of what each original description actually meant, for the report
NOTES = {
    "A1_baseline.jpg": "3 same coins",
    "A2_baseline.jpg": "5 same coins",
    "A3_baseline.jpg": "3 different coins, spaced out",
    "A4_baseline.jpg": "5 mixed coins",
    "B1_dim.jpg": "dim indoor light",
    "B2_harsh.jpg": "harsh direct sunlight",
    "B3_sidelit.jpg": "side-lit",
    "B4_warmbulb.jpg": "yellow/warm bulb",
    "B5_flash.jpg": "flash on",
    "C1_wood.jpg": "wood-grain surface",
    "C2_cloth.jpg": "patterned cloth",
    "C3_dark.jpg": "dark surface",
    "C4_newspaper.jpg": "printed page",
    "D1_touch.jpg": "3 coins touching edge-to-edge in a row",
    "D2_touch.jpg": "edges touching",
    "D3_touch.jpg": "coins overlapping",
    "D4_touch.jpg": "mixed touching case",
    "E1_close.jpg": "coins fill frame",
    "E2_normal.jpg": "camera at normal distance",
    "E3_far.jpg": "camera far",
    "F1_outoffocus.jpg": "slightly out of focus",
    "F2_motionblur.jpg": "motion blur",
    "F3_cluttered.jpg": "cluttered background",
    "F4_verydark.jpg": "very poor light",
}


def main():
    ORIG.mkdir(exist_ok=True)
    manifest_lines = ["original_file,new_file,notes"]

    for old_name, new_name in RENAME_MAP.items():
        old_path = RAW / old_name
        if not old_path.exists():
            print("missing:", old_name)
            continue

        # move heic original out of the way first
        orig_path = ORIG / old_name
        shutil.move(str(old_path), str(orig_path))

        # convert to jpg straight into data/raw with the new name
        new_path = RAW / new_name
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(orig_path), "--out", str(new_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"{old_name} -> {new_name}")
        manifest_lines.append(f"{old_name},{new_name},{NOTES.get(new_name, '')}")

    manifest_path = RAW / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
