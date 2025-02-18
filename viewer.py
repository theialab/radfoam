import numpy as np
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def viewer(args, pipeline_args, model_args, optimizer_args, dataset_args):
    checkpoint = args.config.replace("/config.yaml", "")
    device = torch.device(args.device)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    # Define viewer settings
    viewer_options = {
        "camera_pos": test_data_handler.viewer_pos,
        "camera_up": test_data_handler.viewer_up,
        "camera_forward": test_data_handler.viewer_forward,
    }

    # Setting up model
    model = RadFoamScene(
        args=model_args, device=device, attr_dtype=torch.float16
    )

    model.load_pt(f"{checkpoint}/model.pt")

    def viewer_init(viewer):
        model.update_viewer(viewer)

    model.show(viewer_init, **viewer_options)


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

    viewer(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
