defmodule Honeycomb.CLI do
  @moduledoc """
  Shared CLI argument parsing for Honeycomb mix tasks.

  Centralizes argument parsing to avoid duplication across
  serve, benchmark, and profile tasks.
  """

  @serving_args [
    model: [
      type: :string,
      required: true,
      doc: "HuggingFace model repo to use"
    ],
    chat_template: [
      type: :string,
      required: true,
      doc: "Chat template to use"
    ],
    hf_auth_token: [
      type: :string,
      doc: "HuggingFace auth token for private repos"
    ],
    max_sequence_length: [
      type: :integer,
      default: 512,
      doc: "Maximum sequence length"
    ],
    # New production options
    batch_size: [
      type: :integer,
      default: 1,
      doc: "Maximum batch size"
    ],
    batch_timeout: [
      type: :integer,
      default: 50,
      doc: "Batch timeout in milliseconds"
    ],
    quantization: [
      type: {:in, ["none", "int8", "int4", "fp8"]},
      default: "none",
      doc: "Weight quantization mode"
    ],
    tensor_parallel: [
      type: :integer,
      default: 1,
      doc: "Number of GPUs for tensor parallelism"
    ],
    enable_prefix_caching: [
      type: :boolean,
      default: true,
      doc: "Enable prefix caching"
    ],
    enable_chunked_prefill: [
      type: :boolean,
      default: false,
      doc: "Enable chunked prefill"
    ],
    max_kv_cache_blocks: [
      type: :integer,
      default: 1000,
      doc: "Maximum KV cache blocks"
    ],
    port: [
      type: :integer,
      default: 4000,
      doc: "HTTP server port"
    ]
  ]

  @doc """
  Parses command line arguments for serving configuration.

  Returns `{:ok, config}` or `{:error, reason}`.
  """
  def parse_args(args) do
    case parse_raw_args(args, []) do
      {:ok, parsed} ->
        validate_and_build_config(parsed)

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Applies parsed configuration to the application environment.
  """
  def apply_config(config) do
    serving_config = [
      model: config.model,
      chat_template: config.chat_template,
      auth_token: config.hf_auth_token,
      sequence_length: config.max_sequence_length,
      batch_size: config.batch_size,
      quantization: parse_quantization(config.quantization),
      tensor_parallel: config.tensor_parallel
    ]

    Application.put_env(:honeycomb, Honeycomb.Serving, serving_config)

    # Engine configuration
    engine_config = [
      enable_prefix_caching: config.enable_prefix_caching,
      enable_chunked_prefill: config.enable_chunked_prefill,
      max_kv_cache_blocks: config.max_kv_cache_blocks,
      batch_timeout: config.batch_timeout
    ]

    Application.put_env(:honeycomb, Honeycomb.Engine, engine_config)

    :ok
  end

  @doc """
  Returns usage help text.
  """
  def usage do
    """
    Usage: mix honeycomb.[serve|benchmark|profile] [options]

    Required options:
      --model=REPO           HuggingFace model repository
      --chat-template=NAME   Chat template name (e.g., phi3, llama3, chatml)

    Optional options:
      --hf-auth-token=TOKEN       HuggingFace auth token for gated models
      --max-sequence-length=N     Maximum sequence length (default: 512)
      --batch-size=N              Maximum batch size (default: 1)
      --batch-timeout=MS          Batch timeout in ms (default: 50)
      --quantization=MODE         Quantization: none|int8|int4|fp8 (default: none)
      --tensor-parallel=N         Number of GPUs (default: 1)
      --enable-prefix-caching     Enable prefix caching (default: true)
      --enable-chunked-prefill    Enable chunked prefill (default: false)
      --max-kv-cache-blocks=N     KV cache blocks (default: 1000)
      --port=PORT                 HTTP port (default: 4000)

    Examples:
      mix honeycomb.serve --model=microsoft/Phi-3-mini-4k-instruct --chat-template=phi3
      mix honeycomb.serve --model=meta-llama/Llama-3-8B-Instruct --chat-template=llama3 --quantization=int8
    """
  end

  # Private helpers

  defp parse_raw_args([], acc), do: {:ok, acc}

  defp parse_raw_args(["--help" | _], _acc), do: {:help, usage()}
  defp parse_raw_args(["-h" | _], _acc), do: {:help, usage()}

  defp parse_raw_args(["--" <> arg | rest], acc) do
    case parse_arg(arg) do
      {:ok, key, value} ->
        parse_raw_args(rest, Keyword.put(acc, key, value))

      {:error, _} = error ->
        error
    end
  end

  defp parse_raw_args([_unknown | rest], acc) do
    # Skip unknown args
    parse_raw_args(rest, acc)
  end

  defp parse_arg(arg) do
    case String.split(arg, "=", parts: 2) do
      [key, value] ->
        key = key |> String.replace("-", "_") |> String.to_atom()
        {:ok, key, parse_value(key, value)}

      [key] ->
        key = key |> String.replace("-", "_") |> String.to_atom()
        {:ok, key, true}
    end
  end

  defp parse_value(key, value) when key in [:max_sequence_length, :batch_size, :batch_timeout, :tensor_parallel, :max_kv_cache_blocks, :port] do
    String.to_integer(value)
  end

  defp parse_value(key, "true") when key in [:enable_prefix_caching, :enable_chunked_prefill] do
    true
  end

  defp parse_value(key, "false") when key in [:enable_prefix_caching, :enable_chunked_prefill] do
    false
  end

  defp parse_value(_key, value), do: value

  defp validate_and_build_config(parsed) do
    defaults = Enum.map(@serving_args, fn {key, opts} ->
      {key, Keyword.get(opts, :default)}
    end) |> Enum.reject(fn {_, v} -> is_nil(v) end)

    config = Keyword.merge(defaults, parsed) |> Map.new()

    cond do
      is_nil(config[:model]) ->
        {:error, "Missing required argument: --model"}

      is_nil(config[:chat_template]) ->
        {:error, "Missing required argument: --chat-template"}

      true ->
        {:ok, config}
    end
  end

  defp parse_quantization("none"), do: nil
  defp parse_quantization("int8"), do: :int8
  defp parse_quantization("int4"), do: :int4
  defp parse_quantization("fp8"), do: :fp8
  defp parse_quantization(_), do: nil
end
