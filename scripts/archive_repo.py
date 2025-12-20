import argparse
import os
import zipfile


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            fp = os.path.join(root, file)
            arcname = os.path.relpath(fp, path)
            ziph.write(fp, arcname)


def main():
    parser = argparse.ArgumentParser(description="Create a ZIP archive of the repository.")
    parser.add_argument("--src", default=".", help="Source directory to zip (repo root)")
    parser.add_argument("--out", default="infoRates_repo.zip", help="Zip file output path")
    args = parser.parse_args()

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(args.src, zipf)
    print(f"Created archive: {args.out}")


if __name__ == "__main__":
    main()
