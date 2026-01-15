defmodule Honeycomb.Telemetry do
  @moduledoc """
  Telemetry and metrics for Honeycomb LLM serving.

  Provides comprehensive observability including:

  - **Request metrics**: Latency, TTFT, throughput
  - **System metrics**: Queue depth, memory usage, GPU utilization
  - **Model metrics**: Tokens/second, batch sizes, cache hit rates

  ## Events

  The following telemetry events are emitted:

  ### Request Events
  - `[:honeycomb, :request, :start]` - Request received
  - `[:honeycomb, :request, :stop]` - Request completed
  - `[:honeycomb, :request, :exception]` - Request failed

  ### Generation Events
  - `[:honeycomb, :generation, :token]` - Token generated
  - `[:honeycomb, :generation, :prefill]` - Prefill completed
  - `[:honeycomb, :generation, :decode]` - Decode step completed

  ### System Events
  - `[:honeycomb, :scheduler, :iteration]` - Scheduler iteration
  - `[:honeycomb, :kv_cache, :allocate]` - KV cache allocation
  - `[:honeycomb, :kv_cache, :evict]` - KV cache eviction

  ## Usage

      # Attach handlers
      Honeycomb.Telemetry.attach_default_handlers()

      # Or use with your own handlers
      :telemetry.attach("my-handler", [:honeycomb, :request, :stop], &MyModule.handle_event/4, nil)
  """

  require Logger

  @request_start [:honeycomb, :request, :start]
  @request_stop [:honeycomb, :request, :stop]
  @request_exception [:honeycomb, :request, :exception]

  @generation_token [:honeycomb, :generation, :token]
  @generation_prefill [:honeycomb, :generation, :prefill]
  @generation_decode [:honeycomb, :generation, :decode]

  @scheduler_iteration [:honeycomb, :scheduler, :iteration]

  @kv_cache_allocate [:honeycomb, :kv_cache, :allocate]
  @kv_cache_evict [:honeycomb, :kv_cache, :evict]

  # Public API

  @doc """
  Attaches default telemetry handlers for logging.
  """
  def attach_default_handlers do
    handlers = [
      {"honeycomb-request-handler", [@request_start, @request_stop, @request_exception], &handle_request_event/4},
      {"honeycomb-generation-handler", [@generation_token, @generation_prefill, @generation_decode], &handle_generation_event/4},
      {"honeycomb-system-handler", [@scheduler_iteration, @kv_cache_allocate, @kv_cache_evict], &handle_system_event/4}
    ]

    Enum.each(handlers, fn {name, events, handler} ->
      :telemetry.attach_many(name, events, handler, nil)
    end)

    :ok
  end

  @doc """
  Detaches all default handlers.
  """
  def detach_default_handlers do
    :telemetry.detach("honeycomb-request-handler")
    :telemetry.detach("honeycomb-generation-handler")
    :telemetry.detach("honeycomb-system-handler")
    :ok
  end

  # Event emission functions

  @doc """
  Emits a request start event.
  """
  def request_start(metadata \\ %{}) do
    :telemetry.execute(@request_start, %{system_time: System.system_time()}, metadata)
  end

  @doc """
  Emits a request stop event with duration.
  """
  def request_stop(start_time, metadata \\ %{}) do
    duration = System.monotonic_time() - start_time
    :telemetry.execute(@request_stop, %{duration: duration}, metadata)
  end

  @doc """
  Emits a request exception event.
  """
  def request_exception(kind, reason, stacktrace, metadata \\ %{}) do
    :telemetry.execute(@request_exception, %{}, Map.merge(metadata, %{
      kind: kind,
      reason: reason,
      stacktrace: stacktrace
    }))
  end

  @doc """
  Emits a token generation event.
  """
  def token_generated(metadata \\ %{}) do
    :telemetry.execute(@generation_token, %{system_time: System.system_time()}, metadata)
  end

  @doc """
  Emits a prefill completion event.
  """
  def prefill_complete(duration, num_tokens, metadata \\ %{}) do
    measurements = %{
      duration: duration,
      num_tokens: num_tokens,
      tokens_per_second: num_tokens / (duration / 1_000_000_000)
    }
    :telemetry.execute(@generation_prefill, measurements, metadata)
  end

  @doc """
  Emits a decode step event.
  """
  def decode_step(duration, batch_size, metadata \\ %{}) do
    measurements = %{
      duration: duration,
      batch_size: batch_size
    }
    :telemetry.execute(@generation_decode, measurements, metadata)
  end

  @doc """
  Emits a scheduler iteration event.
  """
  def scheduler_iteration(measurements, metadata \\ %{}) do
    :telemetry.execute(@scheduler_iteration, measurements, metadata)
  end

  @doc """
  Emits a KV cache allocation event.
  """
  def kv_cache_allocate(num_blocks, metadata \\ %{}) do
    :telemetry.execute(@kv_cache_allocate, %{num_blocks: num_blocks}, metadata)
  end

  @doc """
  Emits a KV cache eviction event.
  """
  def kv_cache_evict(num_blocks, metadata \\ %{}) do
    :telemetry.execute(@kv_cache_evict, %{num_blocks: num_blocks}, metadata)
  end

  @doc """
  Spans a function execution with telemetry.
  """
  def span(event_prefix, metadata, fun) when is_function(fun, 0) do
    start_time = System.monotonic_time()
    :telemetry.execute(event_prefix ++ [:start], %{system_time: System.system_time()}, metadata)

    try do
      result = fun.()
      duration = System.monotonic_time() - start_time
      :telemetry.execute(event_prefix ++ [:stop], %{duration: duration}, metadata)
      result
    rescue
      e ->
        duration = System.monotonic_time() - start_time
        :telemetry.execute(event_prefix ++ [:exception], %{duration: duration}, Map.put(metadata, :error, e))
        reraise e, __STACKTRACE__
    end
  end

  # Default handlers

  defp handle_request_event([@honeycomb, :request, :start], _measurements, metadata, _config) do
    Logger.debug("Request started: #{inspect(metadata)}")
  end

  defp handle_request_event([@honeycomb, :request, :stop], %{duration: duration}, metadata, _config) do
    duration_ms = System.convert_time_unit(duration, :native, :millisecond)
    Logger.info("Request completed in #{duration_ms}ms: #{inspect(metadata)}")
  end

  defp handle_request_event([@honeycomb, :request, :exception], _measurements, metadata, _config) do
    Logger.error("Request failed: #{inspect(metadata)}")
  end

  defp handle_generation_event([@honeycomb, :generation, :token], _measurements, metadata, _config) do
    Logger.debug("Token generated: #{inspect(metadata)}")
  end

  defp handle_generation_event([@honeycomb, :generation, :prefill], measurements, metadata, _config) do
    Logger.debug("Prefill complete: #{measurements.num_tokens} tokens, #{Float.round(measurements.tokens_per_second, 2)} tok/s")
  end

  defp handle_generation_event([@honeycomb, :generation, :decode], measurements, _metadata, _config) do
    duration_ms = System.convert_time_unit(measurements.duration, :native, :millisecond)
    Logger.debug("Decode step: batch_size=#{measurements.batch_size}, duration=#{duration_ms}ms")
  end

  defp handle_system_event([@honeycomb, :scheduler, :iteration], measurements, _metadata, _config) do
    Logger.debug("Scheduler: prefill=#{measurements[:num_prefill] || 0}, decode=#{measurements[:num_decode] || 0}")
  end

  defp handle_system_event([@honeycomb, :kv_cache, :allocate], %{num_blocks: n}, _metadata, _config) do
    Logger.debug("KV Cache allocated #{n} blocks")
  end

  defp handle_system_event([@honeycomb, :kv_cache, :evict], %{num_blocks: n}, _metadata, _config) do
    Logger.debug("KV Cache evicted #{n} blocks")
  end
end
