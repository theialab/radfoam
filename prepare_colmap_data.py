import argparse
import os
import warnings

import pycolmap
from PIL import Image
import tqdm


def main(args):
    """Minimal script to run COLMAP on a directory of images."""

    data_dir = args.data_dir
    reconstruction_dir = os.path.join(data_dir, "sparse")

    if os.path.exists(reconstruction_dir):
        raise ValueError("Reconstruction directory already exists")

    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError("data_dir must contain an 'images' directory")

    database_path = os.path.join(data_dir, "database.db")
    if os.path.exists(database_path):
        raise ValueError("Database file already exists")

    database = pycolmap.Database(database_path)

    pycolmap.extract_features(
        database_path,
        images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=args.camera_model,
    )

    print(f"Imported {database.num_images} images to {database_path}")

    pycolmap.match_exhaustive(database_path)

    print(f"Feature matching completed")

    os.makedirs(reconstruction_dir)

    reconstructions = pycolmap.incremental_mapping(
        database_path,
        image_path=images_dir,
        output_path=reconstruction_dir,
    )

    if len(reconstructions) > 1:
        warnings.warn("Multiple reconstructions found")

    reconstruction = reconstructions[0]

    os.makedirs(os.path.join(data_dir, "images_2"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images_4"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images_8"), exist_ok=True)

    print("Downsampling images")

    for image in tqdm.tqdm(list(reconstruction.images.values())):
        image_1_path = os.path.join(images_dir, image.name)
        image_2_path = os.path.join(data_dir, "images_2", image.name)
        image_4_path = os.path.join(data_dir, "images_4", image.name)
        image_8_path = os.path.join(data_dir, "images_8", image.name)

        pil_image = Image.open(image_1_path)
        pil_image_2 = pil_image.resize(
            (pil_image.width // 2, pil_image.height // 2),
            resample=Image.LANCZOS,
        )
        pil_image_4 = pil_image.resize(
            (pil_image.width // 4, pil_image.height // 4),
            resample=Image.LANCZOS,
        )
        pil_image_8 = pil_image.resize(
            (pil_image.width // 8, pil_image.height // 8),
            resample=Image.LANCZOS,
        )

        pil_image_2.save(image_2_path)
        pil_image_4.save(image_4_path)
        pil_image_8.save(image_8_path)

        pil_image.close()
        pil_image_2.close()
        pil_image_4.close()
        pil_image_8.close()

    print("Exporting point cloud")

    reconstruction.export_PLY(os.path.join(data_dir, "point_cloud.ply"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--camera_model", type=str, default="OPENCV")
    args = parser.parse_args()
    main(args)
