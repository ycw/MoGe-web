const gpu_state = {
  tested: false,
  device: null,
  pipeline: null,
  bind_group_layout: null
}

const WGSL_SHADER = /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  threshold_cos: f32,
  smoothing_kernel_radius: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src_normals: array<f32>;
@group(0) @binding(2) var<storage, read> src_masks: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_pixels: array<u32>;

const DIR_TO_ME: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let x = global_id.x;
  let y = global_id.y;
  let w = params.width;
  let h = params.height;

  if (x >= w || y >= h) {
    return;
  }

  let idx = y * w + x;
  let c_idx = idx * 3u;

  let center_normal = vec3<f32>(
    src_normals[c_idx],
    -src_normals[c_idx + 1u],
    -src_normals[c_idx + 2u]
  );

  var sum_normal = vec3<f32>(0.0);
  var total_weight = 0.0;
  let r = params.smoothing_kernel_radius;

  let diff_cos = 1.0 - params.threshold_cos;
  let sigma_sq = 2.0 * diff_cos * diff_cos;

  for (var dy = -r; dy <= r; dy++) {
    let ny = u32(clamp(i32(y) + dy, 0, i32(h) - 1));
    for (var dx = -r; dx <= r; dx++) {
      let nx = u32(clamp(i32(x) + dx, 0, i32(w) - 1));
      let n_idx = (ny * w + nx) * 3u;

      let sample_normal = vec3<f32>(
        src_normals[n_idx],
        -src_normals[n_idx + 1u],
        -src_normals[n_idx + 2u]
      );

      let cos_theta = dot(center_normal, sample_normal);
      if (cos_theta < params.threshold_cos) {
        continue;
      }

      let diff = 1.0 - cos_theta;
      let spatial_weight = exp(-(diff * diff) / sigma_sq);

      sum_normal += sample_normal * spatial_weight;
      total_weight += spatial_weight;
    }
  }

  var norm = center_normal;
  if (total_weight > 0.0) {
    norm = normalize(sum_normal);
  }

  let mask = src_masks[idx];
  let final_normal = normalize(mix(DIR_TO_ME, norm, mask));

  let r_val = u32(clamp((final_normal.x + 1.0) * 127.5, 0.0, 255.0));
  let g_val = u32(clamp((final_normal.y + 1.0) * 127.5, 0.0, 255.0));
  let b_val = u32(clamp((final_normal.z + 1.0) * 127.5, 0.0, 255.0));
  let a_val = 255u;

  dst_pixels[idx] = (a_val << 24u) | (b_val << 16u) | (g_val << 8u) | r_val;
}
`

async function get_gpu_device() {
  if (gpu_state.tested) return gpu_state.device
  gpu_state.tested = true

  if (!navigator.gpu) return null

  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return null

    gpu_state.device = await adapter.requestDevice()
    init_compute_pipeline(gpu_state.device)
    return gpu_state.device
  } catch (e) {
    gpu_state.device = null
    return null
  }
}

function init_compute_pipeline(device) {
  const module = device.createShaderModule({ code: WGSL_SHADER })

  gpu_state.bind_group_layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
    ]
  })

  gpu_state.pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [gpu_state.bind_group_layout] }),
    compute: { module, entryPoint: 'main' }
  })
}

async function to_canvas_gpu({
  device,
  normal_tensor,
  mask_tensor,
  smoothing_threshold_deg = 8,
  smoothing_kernel_radius = 7
}) {
  const sh = normal_tensor.dims[1]
  const sw = normal_tensor.dims[2]
  const pixel_count = sw * sh

  const threshold_cos = Math.cos((smoothing_threshold_deg * Math.PI) / 180.0)
  const params_buf = new ArrayBuffer(16)
  const u32_view = new Uint32Array(params_buf)
  const f32_view = new Float32Array(params_buf)
  const i32_view = new Int32Array(params_buf)

  u32_view[0] = sw
  u32_view[1] = sh
  f32_view[2] = threshold_cos
  i32_view[3] = smoothing_kernel_radius

  const uniform_buf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(uniform_buf, 0, params_buf)

  const normal_buf = device.createBuffer({
    size: normal_tensor.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(normal_buf, 0, normal_tensor.data)

  const mask_buf = device.createBuffer({
    size: mask_tensor.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  })
  device.queue.writeBuffer(mask_buf, 0, mask_tensor.data)

  const output_buf_size = pixel_count * 4
  const output_buf = device.createBuffer({
    size: output_buf_size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })

  const readback_buf = device.createBuffer({
    size: output_buf_size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  })

  const bind_group = device.createBindGroup({
    layout: gpu_state.bind_group_layout,
    entries: [
      { binding: 0, resource: { buffer: uniform_buf } },
      { binding: 1, resource: { buffer: normal_buf } },
      { binding: 2, resource: { buffer: mask_buf } },
      { binding: 3, resource: { buffer: output_buf } }
    ]
  })

  const encoder = device.createCommandEncoder()
  const pass = encoder.beginComputePass()
  pass.setPipeline(gpu_state.pipeline)
  pass.setBindGroup(0, bind_group)
  pass.dispatchWorkgroups(Math.ceil(sw / 16), Math.ceil(sh / 16))
  pass.end()

  encoder.copyBufferToBuffer(output_buf, 0, readback_buf, 0, output_buf_size)
  device.queue.submit([encoder.finish()])

  await readback_buf.mapAsync(GPUMapMode.READ)
  const pixel_array = new Uint8ClampedArray(readback_buf.getMappedRange())

  const canvas = new OffscreenCanvas(sw, sh)
  canvas.getContext('2d').putImageData(new ImageData(pixel_array, sw, sh), 0, 0)

  readback_buf.unmap()

  uniform_buf.destroy()
  normal_buf.destroy()
  mask_buf.destroy()
  output_buf.destroy()
  readback_buf.destroy()

  return canvas
}

function to_canvas_cpu({
  normal_tensor,
  mask_tensor,
  smoothing_threshold_deg = 8,
  smoothing_kernel_radius = 7
}) {
  const sh = normal_tensor.dims[1]
  const sw = normal_tensor.dims[2]
  const src_normals = normal_tensor.data
  const src_masks = mask_tensor.data

  const canvas = new OffscreenCanvas(sw, sh)
  const img_data = new ImageData(sw, sh)
  const dst_pixels = img_data.data

  const threshold_cos = Math.cos((smoothing_threshold_deg * Math.PI) / 180.0)
  const diff_cos = 1.0 - threshold_cos
  const sigma_sq = 2.0 * diff_cos * diff_cos
  const r = smoothing_kernel_radius

  // Helper functions (aligns 1:1 with WGSL built-in functions)
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
  const mix = (a, b, t) => [
    a[0] * (1 - t) + b[0] * t,
    a[1] * (1 - t) + b[1] * t,
    a[2] * (1 - t) + b[2] * t
  ]
  const length = (v) => Math.hypot(v[0], v[1], v[2])
  const normalize = (v) => {
    const len = Math.max(0.0001, length(v))
    return [v[0] / len, v[1] / len, v[2] / len]
  }

  // Camera-facing default normal vector (+Z towards viewer)
  const DIR_TO_ME = [0.0, 0.0, 1.0]

  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const idx = y * sw + x
      const c_idx = idx * 3

      // Convert MoGe space to OpenGL space: flip y (down -> up), flip z (forward -> backward)
      const center_normal = [
        src_normals[c_idx],
        -src_normals[c_idx + 1],
        -src_normals[c_idx + 2]
      ]

      // Bilateral filtering
      let sum_normal = [0.0, 0.0, 0.0]
      let total_weight = 0.0

      for (let dy = -r; dy <= r; dy++) {
        const ny = Math.min(Math.max(y + dy, 0), sh - 1)
        for (let dx = -r; dx <= r; dx++) {
          const nx = Math.min(Math.max(x + dx, 0), sw - 1)
          const n_idx = (ny * sw + nx) * 3

          const sample_normal = [
            src_normals[n_idx],
            -src_normals[n_idx + 1],
            -src_normals[n_idx + 2]
          ]

          const cos_theta = dot(center_normal, sample_normal)
          if (cos_theta < threshold_cos) continue

          const diff = 1.0 - cos_theta
          const spatial_weight = Math.exp(-(diff * diff) / sigma_sq)

          sum_normal[0] += sample_normal[0] * spatial_weight
          sum_normal[1] += sample_normal[1] * spatial_weight
          sum_normal[2] += sample_normal[2] * spatial_weight
          total_weight += spatial_weight
        }
      }

      // Normalize bilateral output
      const norm = total_weight > 0.0 ? normalize(sum_normal) : center_normal

      // Lerp toward camera default normal based on confidence mask
      const mask = src_masks[idx]
      const final_normal = normalize(mix(DIR_TO_ME, norm, mask))

      // Mapping (-1..1 -> 0..255 RGB)
      const out_idx = idx * 4
      dst_pixels[out_idx]     = (final_normal[0] + 1.0) * 127.5
      dst_pixels[out_idx + 1] = (final_normal[1] + 1.0) * 127.5
      dst_pixels[out_idx + 2] = (final_normal[2] + 1.0) * 127.5
      dst_pixels[out_idx + 3] = 255
    }
  }

  canvas.getContext('2d').putImageData(img_data, 0, 0)
  return canvas
}

export async function normal_tensor_to_canvas({
  normal_tensor,
  mask_tensor,
  smoothing_threshold_deg = 8,
  smoothing_kernel_radius = 7
}) {
  const device = await get_gpu_device()

  if (device) {
    return {
      backend: 'gpu',
      normal_canvas: await to_canvas_gpu({
        device,
        normal_tensor,
        mask_tensor,
        smoothing_threshold_deg,
        smoothing_kernel_radius
      })
    }
  }

  return {
    backend: 'cpu',
    normal_canvas: to_canvas_cpu({
      normal_tensor,
      mask_tensor,
      smoothing_threshold_deg,
      smoothing_kernel_radius
    })
  }
}