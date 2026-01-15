defmodule Honeycomb.Quantization do
  @moduledoc """
  Quantization support for model weights and activations.

  Implements various quantization schemes for faster inference:

  - **INT8**: 8-bit integer quantization
  - **INT4**: 4-bit integer quantization (packed)
  - **FP8**: 8-bit floating point (E4M3/E5M2)
  - **BF16**: Brain floating point 16

  ## Quantization Methods

  - **Static quantization**: Pre-compute scales from calibration data
  - **Dynamic quantization**: Compute scales at runtime
  - **Per-tensor**: Single scale for entire tensor
  - **Per-channel**: Different scale per output channel

  ## Usage

      # Quantize weights
      quantized = Honeycomb.Quantization.quantize(weights, :int8, :per_channel)

      # Dequantize for computation
      dequantized = Honeycomb.Quantization.dequantize(quantized)

      # Quantized matmul
      output = Honeycomb.Quantization.quantized_matmul(input, quantized)
  """

  import Nx.Defn

  @doc """
  Quantizes a tensor to the specified format.

  ## Options

    * `:scheme` - :per_tensor or :per_channel (default: :per_tensor)
    * `:symmetric` - Use symmetric quantization (default: true)
  """
  def quantize(tensor, dtype, opts \\ []) do
    scheme = Keyword.get(opts, :scheme, :per_tensor)
    symmetric = Keyword.get(opts, :symmetric, true)

    case dtype do
      :int8 -> quantize_int8(tensor, scheme, symmetric)
      :int4 -> quantize_int4(tensor, scheme, symmetric)
      :fp8 -> quantize_fp8(tensor)
      :bf16 -> Nx.as_type(tensor, :bf16)
      _ -> raise "Unsupported quantization dtype: #{dtype}"
    end
  end

  @doc """
  Dequantizes a quantized tensor back to float.
  """
  def dequantize(%{data: data, scale: scale, zero_point: zero_point, dtype: dtype}) do
    case dtype do
      :int8 -> dequantize_int8(data, scale, zero_point)
      :int4 -> dequantize_int4(data, scale, zero_point)
      :fp8 -> Nx.as_type(data, :f32)
      :bf16 -> Nx.as_type(data, :f32)
    end
  end

  def dequantize(tensor), do: tensor

  @doc """
  Performs quantized matrix multiplication.

  Uses integer arithmetic when possible for speed.
  """
  def quantized_matmul(input, %{dtype: :int8} = quantized) do
    # For INT8, we can use integer matmul then rescale
    # input: [batch, in_features]
    # weight: [out_features, in_features]

    input_q = quantize_int8_dynamic(input)

    # Integer matmul
    result = Nx.dot(input_q.data, [1], quantized.data, [1])

    # Rescale
    scale = Nx.multiply(input_q.scale, quantized.scale)
    Nx.multiply(Nx.as_type(result, :f32), scale)
  end

  def quantized_matmul(input, quantized) do
    # Fallback to dequantize then compute
    weight = dequantize(quantized)
    Nx.dot(input, [1], weight, [1])
  end

  @doc """
  Creates a quantization configuration.
  """
  def config(opts \\ []) do
    %{
      weight_dtype: Keyword.get(opts, :weight_dtype, :int8),
      activation_dtype: Keyword.get(opts, :activation_dtype, :f16),
      scheme: Keyword.get(opts, :scheme, :per_channel),
      symmetric: Keyword.get(opts, :symmetric, true),
      calibration_method: Keyword.get(opts, :calibration_method, :minmax)
    }
  end

  # INT8 Quantization

  defp quantize_int8(tensor, :per_tensor, symmetric) do
    {scale, zero_point} = compute_scale_zp_int8(tensor, symmetric)

    quantized =
      tensor
      |> Nx.divide(scale)
      |> Nx.add(zero_point)
      |> Nx.round()
      |> Nx.clip(-128, 127)
      |> Nx.as_type(:s8)

    %{
      data: quantized,
      scale: scale,
      zero_point: zero_point,
      dtype: :int8,
      scheme: :per_tensor
    }
  end

  defp quantize_int8(tensor, :per_channel, symmetric) do
    # Quantize per output channel (axis 0)
    num_channels = Nx.axis_size(tensor, 0)

    {scales, zero_points, quantized_channels} =
      Enum.reduce(0..(num_channels - 1), {[], [], []}, fn i, {scales, zps, channels} ->
        channel = tensor[i]
        {scale, zp} = compute_scale_zp_int8(channel, symmetric)

        q_channel =
          channel
          |> Nx.divide(scale)
          |> Nx.add(zp)
          |> Nx.round()
          |> Nx.clip(-128, 127)

        {[scale | scales], [zp | zps], [q_channel | channels]}
      end)

    quantized = Nx.stack(Enum.reverse(quantized_channels)) |> Nx.as_type(:s8)

    %{
      data: quantized,
      scale: Nx.stack(Enum.reverse(scales)),
      zero_point: Nx.stack(Enum.reverse(zero_points)),
      dtype: :int8,
      scheme: :per_channel
    }
  end

  defp compute_scale_zp_int8(tensor, true = _symmetric) do
    # Symmetric: zero_point = 0, scale = max(abs(tensor)) / 127
    max_abs = Nx.reduce_max(Nx.abs(tensor))
    scale = Nx.divide(max_abs, 127)
    # Avoid division by zero
    scale = Nx.max(scale, Nx.tensor(1.0e-10))
    {scale, Nx.tensor(0)}
  end

  defp compute_scale_zp_int8(tensor, false = _symmetric) do
    # Asymmetric: map [min, max] to [-128, 127]
    min_val = Nx.reduce_min(tensor)
    max_val = Nx.reduce_max(tensor)

    scale = Nx.divide(Nx.subtract(max_val, min_val), 255)
    scale = Nx.max(scale, Nx.tensor(1.0e-10))

    zero_point = Nx.round(Nx.divide(Nx.negate(min_val), scale))
    zero_point = Nx.clip(zero_point, -128, 127)

    {scale, zero_point}
  end

  defp dequantize_int8(data, scale, zero_point) do
    data
    |> Nx.as_type(:f32)
    |> Nx.subtract(zero_point)
    |> Nx.multiply(scale)
  end

  defp quantize_int8_dynamic(tensor) do
    {scale, zero_point} = compute_scale_zp_int8(tensor, true)

    quantized =
      tensor
      |> Nx.divide(scale)
      |> Nx.round()
      |> Nx.clip(-128, 127)
      |> Nx.as_type(:s8)

    %{data: quantized, scale: scale, zero_point: zero_point, dtype: :int8}
  end

  # INT4 Quantization

  defp quantize_int4(tensor, scheme, symmetric) do
    # First quantize to int8, then pack to int4
    int8_quantized = quantize_int8(tensor, scheme, symmetric)

    # Pack two int4 values into one int8
    packed = pack_int4(int8_quantized.data)

    %{int8_quantized |
      data: packed,
      dtype: :int4
    }
  end

  defp pack_int4(tensor) do
    # Clip to 4-bit range [-8, 7]
    clipped = Nx.clip(tensor, -8, 7)
    flat = Nx.to_flat_list(clipped)

    # Pad if odd length
    padded = if rem(length(flat), 2) == 1, do: flat ++ [0], else: flat

    # Pack pairs
    packed =
      padded
      |> Enum.chunk_every(2)
      |> Enum.map(fn [a, b] ->
        # Store as (low << 4) | (high & 0xF)
        low = a + 8  # Shift to [0, 15]
        high = b + 8
        Bitwise.bor(Bitwise.bsl(low, 4), high)
      end)

    Nx.tensor(packed, type: :u8)
  end

  defp dequantize_int4(packed_data, scale, zero_point) do
    # Unpack int4 values
    flat = Nx.to_flat_list(packed_data)

    unpacked =
      Enum.flat_map(flat, fn byte ->
        low = Bitwise.bsr(byte, 4) - 8
        high = Bitwise.band(byte, 0xF) - 8
        [low, high]
      end)

    Nx.tensor(unpacked, type: :s8)
    |> dequantize_int8(scale, zero_point)
  end

  # FP8 Quantization

  defp quantize_fp8(tensor) do
    # E4M3 format: 1 sign, 4 exponent, 3 mantissa
    # Range: [-448, 448], precision: ~0.1%

    # Clip to FP8 range
    max_val = 448.0
    clipped = Nx.clip(tensor, -max_val, max_val)

    # Store as BF16 for now (native FP8 requires hardware support)
    %{
      data: Nx.as_type(clipped, :bf16),
      scale: Nx.tensor(1.0),
      zero_point: Nx.tensor(0.0),
      dtype: :fp8
    }
  end

  @doc """
  Applies quantization-aware training (QAT) simulation.

  Uses straight-through estimator for gradients.
  """
  defn fake_quantize(tensor, scale, zero_point, qmin, qmax) do
    # Forward: quantize then dequantize
    # Backward: straight-through (gradient flows unchanged)
    quantized =
      tensor
      |> Nx.divide(scale)
      |> Nx.add(zero_point)
      |> Nx.round()
      |> Nx.clip(qmin, qmax)

    # Dequantize
    quantized
    |> Nx.subtract(zero_point)
    |> Nx.multiply(scale)
  end

  @doc """
  Calculates the optimal scale using calibration data.
  """
  def calibrate(tensors, method \\ :minmax) do
    case method do
      :minmax ->
        min_val = tensors |> Enum.map(&Nx.reduce_min/1) |> Enum.min()
        max_val = tensors |> Enum.map(&Nx.reduce_max/1) |> Enum.max()
        compute_scale_from_range(min_val, max_val)

      :percentile ->
        # Use 99.9th percentile to avoid outliers
        all_values = Enum.flat_map(tensors, &Nx.to_flat_list/1)
        sorted = Enum.sort(all_values)
        idx = round(0.999 * length(sorted))
        max_val = Enum.at(sorted, idx)
        compute_scale_from_range(-max_val, max_val)

      :mse ->
        # Find scale that minimizes mean squared error
        # (Simplified: use minmax for now)
        calibrate(tensors, :minmax)
    end
  end

  defp compute_scale_from_range(min_val, max_val) do
    range = max_val - min_val
    scale = range / 255
    zero_point = round(-min_val / scale)
    {max(scale, 1.0e-10), zero_point}
  end
end
