import os
import numpy as np
from PIL import Image
import configargparse
import warnings

warnings.filterwarnings("ignore")

import torch

from data_loader import DataHandler
from configs import *
from radfoam_model.scene import RadFoamScene
import radfoam


seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def benchmark(args, pipeline_args, model_args, optimizer_args, dataset_args):
    checkpoint = args.config.replace("/config.yaml", "")
    os.makedirs(os.path.join(checkpoint, "test"), exist_ok=True)
    device = torch.device(args.device)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test", downsample=min(dataset_args.downsample)
    )

    # Setting up model
    model = RadFoamScene(
        args=model_args, device=device, attr_dtype=torch.float16
    )

    model.load_pt(f"{checkpoint}/model.pt")

    points, attributes, point_adjacency, point_adjacency_offsets = (
        model.get_trace_data()
    )
    self_point_inds = torch.zeros_like(point_adjacency.long())
    scatter_inds = point_adjacency_offsets[1:-1].long()
    self_point_inds.scatter_add_(0, scatter_inds, torch.ones_like(scatter_inds))
    self_point_inds = torch.cumsum(self_point_inds, dim=0)
    self_points = points[self_point_inds]

    adjacent_points = points[point_adjacency.long()]
    adjacent_offsets = adjacent_points - self_points
    adjacent_offsets = torch.cat(
        [adjacent_offsets, torch.zeros_like(adjacent_offsets[:, :1])], dim=1
    ).to(torch.half)

    c2w = test_data_handler.c2ws
    width, height = test_data_handler.img_wh
    fy = test_data_handler.fy

    cameras = []
    positions = []

    for i in range(c2w.shape[0]):
        if i % 8 == 0:
            position = c2w[i, :3, 3].contiguous()
            fov = float(2 * np.arctan(height / (2 * fy)))

            right = c2w[i, :3, 0].contiguous()
            up = -c2w[i, :3, 1].contiguous()
            forward = c2w[i, :3, 2].contiguous()

            positions.append(position)

            camera = {
                "position": position,
                "forward": forward,
                "right": right,
                "up": up,
                "fov": fov,
                "width": width,
                "height": height,
                "model": "pinhole",
            }
            cameras.append(camera)

    n_frames = len(cameras)

    positions = torch.stack(positions, dim=0).to(device)
    start_points = radfoam.nn(points, model.aabb_tree, positions)

    output = torch.zeros(
        (n_frames, height, width), dtype=torch.uint32, device=device
    )

    torch.cuda.synchronize()

    # warmup
    for i in range(n_frames):
        model.pipeline.trace_benchmark(
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            adjacent_offsets,
            cameras[i],
            start_points[i],
            output[i],
            weight_threshold=0.05,
        )

    torch.cuda.synchronize()
    n_reps = 5
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(n_reps):
        for i in range(n_frames):
            model.pipeline.trace_benchmark(
                points,
                attributes,
                point_adjacency,
                point_adjacency_offsets,
                adjacent_offsets,
                cameras[i],
                start_points[i],
                output[i],
                weight_threshold=0.05,
            )

    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()

    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event)
    framerate = n_reps * n_frames / (total_time / 1000.0)

    print(f"Total time: {total_time} ms")
    print(f"FPS: {framerate}")


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

    benchmark(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
