
from pathlib import Path
import gdown

DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1V8TqCMwcei2QIphqMYxA7tpSe-Ok3aVG"
)

VIDEO_DIR = Path(__file__).resolve().parent / "data" / "videos"


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading accessible files from:")
    print(DRIVE_FOLDER_URL)
    print(f"Destination: {VIDEO_DIR}")

    downloaded = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(VIDEO_DIR),
        quiet=False,
    )

    if downloaded:
        print("Download process completed.")
    else:
        print(
            "No files were downloaded. Check that the Google Drive "
            "folder is accessible and its files can be downloaded."
        )


if __name__ == "__main__":
    main()
