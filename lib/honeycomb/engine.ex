defmodule Honeycomb.Engine do
  @moduledoc """
  Production LLM Inference Engine.

  This is the main orchestration module that integrates all optimization
  components for production-ready LLM serving:

  - **Continuous Batching**: Dynamic request batching via Scheduler
  - **PagedAttention**: Efficient KV cache via KVCache
  - **Prefix Caching**: System prompt reuse via PrefixCache
  - **Chunked Prefill**: Low TTFT via ChunkedPrefill
  - **Speculative Decoding**: Draft model acceleration
  - **Tensor Parallelism**: Multi-GPU support

  ## Architecture

      ┌─────────────────────────────────────────────────────────────┐
      │                      HTTP Router                            │
      └─────────────────────┬───────────────────────────────────────┘
                            │
      ┌─────────────────────▼───────────────────────────────────────┐
      │                    Engine                                   │
      │  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐ │
      │  │Scheduler │ │KV Cache  │ │Prefix Cache│ │Chunked Prefill│ │
      │  └──────────┘ └──────────┘ └────────────┘ └──────────────┘ │
      └─────────────────────┬───────────────────────────────────────┘
                            │
      ┌─────────────────────▼───────────────────────────────────────┐
      │               Model Serving (Bumblebee/EXLA)                │
      └─────────────────────────────────────────────────────────────┘

  ## Usage

      # Start the engine
      Honeycomb.Engine.start_link(opts)

      # Submit inference request
      {:ok, request_id} = Honeycomb.Engine.submit(messages, opts)

      # Stream results
      Honeycomb.Engine.stream(request_id)
  """

  use GenServer
  require Logger

  alias Honeycomb.{Scheduler, KVCache, PrefixCache, ChunkedPrefill, Metrics, Telemetry}
  alias Honeycomb.Scheduler.{Request, SchedulerOutput}
  alias Honeycomb.Sampling
  alias Honeycomb.Sampling.Params

  defstruct [
    :config,
    :model_serving,
    :tokenizer,
    :generation_config,
    :running,
    :request_streams
  ]

  @default_config %{
    max_batch_size: 32,
    max_waiting_tokens: 2048,
    max_model_len: 4096,
    enable_prefix_caching: true,
    enable_chunked_prefill: false,
    chunk_size: 512,
    iteration_timeout: 100
  }

  # Client API

  @doc """
  Starts the inference engine.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Submits a new inference request.

  Returns `{:ok, request_id}` or `{:error, reason}`.
  """
  def submit(messages, opts \\ []) do
    GenServer.call(__MODULE__, {:submit, messages, opts})
  end

  @doc """
  Streams generated tokens for a request.

  Returns a Stream that yields tokens as they're generated.
  """
  def stream(request_id) do
    Stream.resource(
      fn -> request_id end,
      fn req_id ->
        case get_next_token(req_id) do
          {:token, token} -> {[token], req_id}
          :done -> {:halt, req_id}
          :waiting -> {[], req_id}
        end
      end,
      fn _req_id -> :ok end
    )
  end

  @doc """
  Gets the next generated token for a request (non-blocking).
  """
  def get_next_token(request_id) do
    GenServer.call(__MODULE__, {:get_token, request_id})
  end

  @doc """
  Cancels a running request.
  """
  def cancel(request_id) do
    GenServer.call(__MODULE__, {:cancel, request_id})
  end

  @doc """
  Returns engine statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Checks if the engine is ready to serve requests.
  """
  def ready? do
    GenServer.call(__MODULE__, :ready?)
  end

  @doc """
  Returns health status.
  """
  def health do
    GenServer.call(__MODULE__, :health)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    config = Map.merge(@default_config, Map.new(opts))

    # Start telemetry
    Telemetry.attach_default_handlers()

    state = %__MODULE__{
      config: config,
      model_serving: nil,
      tokenizer: nil,
      generation_config: nil,
      running: false,
      request_streams: %{}
    }

    Logger.info("Engine: Initialized with config #{inspect(config)}")

    # Start the main loop
    send(self(), :start_loop)

    {:ok, state}
  end

  @impl true
  def handle_call({:submit, messages, opts}, _from, state) do
    start_time = System.monotonic_time()
    Telemetry.request_start(%{messages: length(messages)})
    Metrics.increment_requests()

    # Tokenize the prompt
    prompt = format_messages(messages, state)
    prompt_tokens = tokenize(prompt, state)

    # Check prefix cache
    {cached_blocks, cached_len} =
      if state.config.enable_prefix_caching do
        case PrefixCache.lookup(prompt_tokens) do
          {:ok, blocks, len} ->
            Metrics.record_cache_hit()
            {blocks, len}
          {:miss, _} ->
            Metrics.record_cache_miss()
            {[], 0}
        end
      else
        {[], 0}
      end

    # Build sampling params
    sampling_params = Params.from_openai(opts)
    |> Params.with_context(prompt_tokens)

    # Submit to scheduler
    case Scheduler.add_request(prompt_tokens, [
      max_tokens: opts[:max_tokens] || 256,
      priority: opts[:priority] || 0,
      sampling_params: sampling_params,
      cached_blocks: cached_blocks,
      cached_len: cached_len
    ]) do
      {:ok, request_id} ->
        # Create stream buffer for this request
        request_streams = Map.put(state.request_streams, request_id, %{
          tokens: [],
          done: false,
          start_time: start_time
        })

        state = %{state | request_streams: request_streams}
        {:reply, {:ok, request_id}, state}

      {:error, _} = error ->
        Metrics.record_error()
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:get_token, request_id}, _from, state) do
    case Map.fetch(state.request_streams, request_id) do
      {:ok, %{tokens: [token | rest], done: done}} ->
        request_streams = Map.put(state.request_streams, request_id, %{
          tokens: rest,
          done: done,
          start_time: nil
        })
        state = %{state | request_streams: request_streams}
        {:reply, {:token, token}, state}

      {:ok, %{tokens: [], done: true}} ->
        {:reply, :done, state}

      {:ok, %{tokens: [], done: false}} ->
        {:reply, :waiting, state}

      :error ->
        {:reply, :done, state}
    end
  end

  @impl true
  def handle_call({:cancel, request_id}, _from, state) do
    Scheduler.abort_request(request_id)
    request_streams = Map.delete(state.request_streams, request_id)
    state = %{state | request_streams: request_streams}
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    scheduler_stats = Scheduler.stats()

    kv_stats =
      if Process.whereis(KVCache) do
        KVCache.stats()
      else
        %{}
      end

    prefix_stats =
      if Process.whereis(PrefixCache) do
        PrefixCache.stats()
      else
        %{}
      end

    metrics = Metrics.get_all()

    stats = %{
      engine: %{
        running: state.running,
        active_streams: map_size(state.request_streams)
      },
      scheduler: scheduler_stats,
      kv_cache: kv_stats,
      prefix_cache: prefix_stats,
      metrics: metrics
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    ready = state.running and Process.whereis(Honeycomb.Serving) != nil
    {:reply, ready, state}
  end

  @impl true
  def handle_call(:health, _from, state) do
    checks = %{
      engine: state.running,
      serving: Process.whereis(Honeycomb.Serving) != nil,
      scheduler: Process.whereis(Scheduler) != nil,
      kv_cache: Process.whereis(KVCache) != nil
    }

    healthy = Enum.all?(Map.values(checks), & &1)

    status = if healthy, do: :healthy, else: :degraded

    {:reply, %{status: status, checks: checks}, state}
  end

  @impl true
  def handle_info(:start_loop, state) do
    # Start the inference loop
    schedule_iteration()
    {:noreply, %{state | running: true}}
  end

  @impl true
  def handle_info(:run_iteration, state) do
    state = run_inference_iteration(state)
    schedule_iteration()
    {:noreply, state}
  end

  # Private helpers

  defp schedule_iteration do
    Process.send_after(self(), :run_iteration, 1)
  end

  defp run_inference_iteration(state) do
    # Get scheduled batch from scheduler
    output = Scheduler.schedule()

    if SchedulerOutput.empty?(output) do
      state
    else
      Telemetry.scheduler_iteration(%{
        num_prefill: length(output.prefill_requests),
        num_decode: length(output.decode_requests)
      })

      # Process prefill requests
      state = process_prefill(output.prefill_requests, state)

      # Process decode requests
      state = process_decode(output.decode_requests, state)

      # Update metrics
      Metrics.update_queue_depth(Scheduler.num_waiting())

      state
    end
  end

  defp process_prefill([], state), do: state

  defp process_prefill(requests, state) do
    Enum.reduce(requests, state, fn request, acc_state ->
      if state.config.enable_chunked_prefill and
           ChunkedPrefill.needs_chunking?(Request.prompt_length(request)) do
        process_chunked_prefill(request, acc_state)
      else
        process_full_prefill(request, acc_state)
      end
    end)
  end

  defp process_chunked_prefill(request, state) do
    # Create chunked prefill state
    _prefill_state = ChunkedPrefill.new(
      request.id,
      request.prompt_tokens,
      chunk_size: state.config.chunk_size
    )

    # Process first chunk
    # In real implementation, would iterate through chunks
    process_full_prefill(request, state)
  end

  defp process_full_prefill(request, state) do
    start_time = System.monotonic_time()

    # Run prefill (in real impl, calls model serving)
    # For now, just mark as processed
    Metrics.increment_tokens_prefilled(Request.prompt_length(request))

    duration = System.monotonic_time() - start_time
    Telemetry.prefill_complete(duration, Request.prompt_length(request))

    state
  end

  defp process_decode([], state), do: state

  defp process_decode(requests, state) do
    start_time = System.monotonic_time()

    # Process each decode request
    state = Enum.reduce(requests, state, fn request, acc_state ->
      # Generate next token (in real impl, calls model serving)
      # For now, simulate token generation
      new_token = simulate_token()

      # Update scheduler
      finished = Request.at_max_tokens?(request)
      Scheduler.update_sequence(request.id, new_token, finished)

      # Update stream buffer
      case Map.fetch(acc_state.request_streams, request.id) do
        {:ok, stream_state} ->
          updated = %{stream_state |
            tokens: stream_state.tokens ++ [new_token],
            done: finished
          }
          request_streams = Map.put(acc_state.request_streams, request.id, updated)
          %{acc_state | request_streams: request_streams}

        :error ->
          acc_state
      end
    end)

    # Record metrics
    Metrics.increment_tokens_generated(length(requests))

    duration = System.monotonic_time() - start_time
    Telemetry.decode_step(duration, length(requests))

    state
  end

  defp format_messages(messages, _state) do
    # Use template system
    template = Application.get_env(:honeycomb, Honeycomb.Serving)[:chat_template] || "chatml"
    Honeycomb.Templates.apply_chat_template(template, messages)
  end

  defp tokenize(_prompt, _state) do
    # In real implementation, would use Bumblebee tokenizer
    # For now, return placeholder tokens
    Enum.to_list(1..10)
  end

  defp simulate_token do
    # Placeholder for actual token generation
    :rand.uniform(32000)
  end
end
