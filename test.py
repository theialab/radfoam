import numpy as np
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import psnr
import radfoam


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def test(args, pipeline_args, model_args, optimizer_args, dataset_args):
    checkpoint = args.config.replace("/config.yaml", "")
    os.makedirs(os.path.join(checkpoint, "test"), exist_ok=True)
    device = torch.device(args.device)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )
    test_ray_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rays, batch_size=1, shuffle=False
    )
    test_rgb_batch_fetcher = radfoam.BatchFetcher(
        test_data_handler.rgbs, batch_size=1, shuffle=False
    )

    # Setting up model
    model = RadFoamScene(args=model_args, device=device)

    model.load_pt(f"{checkpoint}/model.pt")

    def test_render(
        test_data_handler, ray_batch_fetcher, rgb_batch_fetcher
    ):
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            rays[:, 0, 0].cuda(), points, model.aabb_tree
        )

        psnr_list = []
        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]
                output, _, _, _, _ = model(ray_batch, start_points[i])

                # White background
                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean()
                psnr_list.append(img_psnr)
                torch.cuda.synchronize()

                error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                rgb_output = np.uint8(rgb_output.cpu() * 255)
                rgb_batch = np.uint8(rgb_batch.cpu() * 255)

                im = Image.fromarray(
                    np.concatenate([rgb_output, rgb_batch, error], axis=1)
                )
                im.save(
                    f"{checkpoint}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}.png"
                )

        average_psnr = sum(psnr_list) / len(psnr_list)

        f = open(f"{checkpoint}/metrics.txt", "w")
        f.write(f"Average PSNR: {average_psnr}")
        f.close()

        return average_psnr

    test_render(
        test_data_handler, test_ray_batch_fetcher, test_rgb_batch_fetcher
    )


def main():
    parser = configargparse.ArgParser()

    model_params = ModelParams(parser)
    dataset_params = DatasetParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)

    # Add argument to specify a custom config file
    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    # Parse arguments
    args = parser.parse_args()

    test(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
