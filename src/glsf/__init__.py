"""GPU-accelerated 3D structure factor via WebGPU."""

import numpy as np
import wgpu

_SHADER = """
struct Params {
    N: u32,
    num_k: u32,
}

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> k_vecs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.num_k) {
        return;
    }

    let kx = k_vecs[idx * 3u];
    let ky = k_vecs[idx * 3u + 1u];
    let kz = k_vecs[idx * 3u + 2u];

    var sum_cos: f32 = 0.0;
    var sum_sin: f32 = 0.0;

    for (var j: u32 = 0u; j < params.N; j = j + 1u) {
        let px = points[j * 3u];
        let py = points[j * 3u + 1u];
        let pz = points[j * 3u + 2u];
        let phase = kx * px + ky * py + kz * pz;
        sum_cos = sum_cos + cos(phase);
        sum_sin = sum_sin + sin(phase);
    }

    output[idx] = (sum_cos * sum_cos + sum_sin * sum_sin) / f32(params.N);
}
"""

_device = None
_pipeline = None


def _ensure_device():
    global _device, _pipeline
    if _device is not None:
        return
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    _device = adapter.request_device_sync()
    shader = _device.create_shader_module(code=_SHADER)
    _pipeline = _device.create_compute_pipeline(
        layout="auto",
        compute={"module": shader, "entry_point": "main"},
    )


def sf3d(points, k_vecs):
    """Compute the 3D structure factor S(k) for each k-vector.

    Args:
        points: (N, 3) float32 array of particle positions.
        k_vecs: (Nk, 3) float32 array of wave vectors.

    Returns:
        (Nk,) float32 array of S(k) values.
    """
    _ensure_device()

    pts = np.ascontiguousarray(points, dtype=np.float32).ravel()
    kvs = np.ascontiguousarray(k_vecs, dtype=np.float32).ravel()
    N = len(pts) // 3
    Nk = len(kvs) // 3

    points_buf = _device.create_buffer_with_data(data=pts, usage=wgpu.BufferUsage.STORAGE)
    k_vecs_buf = _device.create_buffer_with_data(data=kvs, usage=wgpu.BufferUsage.STORAGE)
    output_buf = _device.create_buffer(size=Nk * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    params_buf = _device.create_buffer_with_data(
        data=np.array([N, Nk], dtype=np.uint32), usage=wgpu.BufferUsage.UNIFORM
    )

    bind_group = _device.create_bind_group(
        layout=_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": points_buf}},
            {"binding": 1, "resource": {"buffer": k_vecs_buf}},
            {"binding": 2, "resource": {"buffer": output_buf}},
            {"binding": 3, "resource": {"buffer": params_buf}},
        ],
    )

    encoder = _device.create_command_encoder()
    pass_ = encoder.begin_compute_pass()
    pass_.set_pipeline(_pipeline)
    pass_.set_bind_group(0, bind_group)
    pass_.dispatch_workgroups((Nk + 255) // 256, 1, 1)
    pass_.end()
    _device.queue.submit([encoder.finish()])

    result = np.frombuffer(_device.queue.read_buffer(output_buf), dtype=np.float32).copy()
    k_sq = np.sum(np.asarray(k_vecs, dtype=np.float32) ** 2, axis=1)
    result[k_sq < 1e-10] = 0.0
    return result
