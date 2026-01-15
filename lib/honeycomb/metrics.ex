defmodule Honeycomb.Metrics do
  @moduledoc """
  Real-time metrics collection and aggregation for Honeycomb.

  Collects and exposes metrics for monitoring dashboards (Prometheus, etc.):

  - Request latency histograms
  - Token throughput counters
  - Queue depth gauges
  - Memory utilization
  - Cache hit rates

  ## Usage

      # Start metrics collector
      Honeycomb.Metrics.start_link()

      # Record metrics
      Honeycomb.Metrics.record_request_latency(150.5)
      Honeycomb.Metrics.increment_tokens_generated(10)

      # Get current metrics
      Honeycomb.Metrics.get_all()
  """

  use GenServer
  require Logger

  @default_histogram_buckets [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

  defstruct [
    :start_time,
    :requests_total,
    :requests_active,
    :tokens_generated,
    :tokens_prefilled,
    :latency_histogram,
    :ttft_histogram,
    :tps_samples,
    :queue_depth,
    :cache_hits,
    :cache_misses,
    :kv_blocks_used,
    :kv_blocks_total,
    :errors_total
  ]

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Records a completed request with its latency in milliseconds.
  """
  def record_request_latency(latency_ms) do
    GenServer.cast(__MODULE__, {:record_latency, latency_ms})
  end

  @doc """
  Records time to first token in milliseconds.
  """
  def record_ttft(ttft_ms) do
    GenServer.cast(__MODULE__, {:record_ttft, ttft_ms})
  end

  @doc """
  Increments the number of tokens generated.
  """
  def increment_tokens_generated(count \\ 1) do
    GenServer.cast(__MODULE__, {:increment_tokens, count})
  end

  @doc """
  Increments the number of tokens prefilled.
  """
  def increment_tokens_prefilled(count) do
    GenServer.cast(__MODULE__, {:increment_prefill, count})
  end

  @doc """
  Records a cache hit.
  """
  def record_cache_hit do
    GenServer.cast(__MODULE__, :cache_hit)
  end

  @doc """
  Records a cache miss.
  """
  def record_cache_miss do
    GenServer.cast(__MODULE__, :cache_miss)
  end

  @doc """
  Updates KV cache utilization.
  """
  def update_kv_cache(used, total) do
    GenServer.cast(__MODULE__, {:kv_cache_update, used, total})
  end

  @doc """
  Updates queue depth.
  """
  def update_queue_depth(depth) do
    GenServer.cast(__MODULE__, {:queue_depth, depth})
  end

  @doc """
  Increments request counter.
  """
  def increment_requests do
    GenServer.cast(__MODULE__, :request_start)
  end

  @doc """
  Decrements active request counter.
  """
  def decrement_active_requests do
    GenServer.cast(__MODULE__, :request_complete)
  end

  @doc """
  Records an error.
  """
  def record_error do
    GenServer.cast(__MODULE__, :error)
  end

  @doc """
  Records tokens per second sample.
  """
  def record_tps(tps) do
    GenServer.cast(__MODULE__, {:record_tps, tps})
  end

  @doc """
  Gets all current metrics.
  """
  def get_all do
    GenServer.call(__MODULE__, :get_all)
  end

  @doc """
  Gets metrics in Prometheus format.
  """
  def prometheus_format do
    GenServer.call(__MODULE__, :prometheus_format)
  end

  @doc """
  Resets all metrics.
  """
  def reset do
    GenServer.call(__MODULE__, :reset)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    state = initial_state()
    {:ok, state}
  end

  @impl true
  def handle_cast({:record_latency, latency_ms}, state) do
    histogram = add_to_histogram(state.latency_histogram, latency_ms)
    {:noreply, %{state | latency_histogram: histogram}}
  end

  @impl true
  def handle_cast({:record_ttft, ttft_ms}, state) do
    histogram = add_to_histogram(state.ttft_histogram, ttft_ms)
    {:noreply, %{state | ttft_histogram: histogram}}
  end

  @impl true
  def handle_cast({:increment_tokens, count}, state) do
    {:noreply, %{state | tokens_generated: state.tokens_generated + count}}
  end

  @impl true
  def handle_cast({:increment_prefill, count}, state) do
    {:noreply, %{state | tokens_prefilled: state.tokens_prefilled + count}}
  end

  @impl true
  def handle_cast(:cache_hit, state) do
    {:noreply, %{state | cache_hits: state.cache_hits + 1}}
  end

  @impl true
  def handle_cast(:cache_miss, state) do
    {:noreply, %{state | cache_misses: state.cache_misses + 1}}
  end

  @impl true
  def handle_cast({:kv_cache_update, used, total}, state) do
    {:noreply, %{state | kv_blocks_used: used, kv_blocks_total: total}}
  end

  @impl true
  def handle_cast({:queue_depth, depth}, state) do
    {:noreply, %{state | queue_depth: depth}}
  end

  @impl true
  def handle_cast(:request_start, state) do
    {:noreply, %{state |
      requests_total: state.requests_total + 1,
      requests_active: state.requests_active + 1
    }}
  end

  @impl true
  def handle_cast(:request_complete, state) do
    {:noreply, %{state | requests_active: max(0, state.requests_active - 1)}}
  end

  @impl true
  def handle_cast(:error, state) do
    {:noreply, %{state | errors_total: state.errors_total + 1}}
  end

  @impl true
  def handle_cast({:record_tps, tps}, state) do
    # Keep last 100 samples for moving average
    samples = Enum.take([tps | state.tps_samples], 100)
    {:noreply, %{state | tps_samples: samples}}
  end

  @impl true
  def handle_call(:get_all, _from, state) do
    uptime = System.monotonic_time(:second) - state.start_time

    metrics = %{
      uptime_seconds: uptime,
      requests_total: state.requests_total,
      requests_active: state.requests_active,
      tokens_generated: state.tokens_generated,
      tokens_prefilled: state.tokens_prefilled,
      errors_total: state.errors_total,
      queue_depth: state.queue_depth,
      cache_hit_rate: cache_hit_rate(state),
      kv_cache_utilization: kv_utilization(state),
      latency_p50: percentile(state.latency_histogram, 0.5),
      latency_p95: percentile(state.latency_histogram, 0.95),
      latency_p99: percentile(state.latency_histogram, 0.99),
      ttft_p50: percentile(state.ttft_histogram, 0.5),
      ttft_p95: percentile(state.ttft_histogram, 0.95),
      avg_tps: average_tps(state),
      throughput_tps: state.tokens_generated / max(uptime, 1)
    }

    {:reply, metrics, state}
  end

  @impl true
  def handle_call(:prometheus_format, _from, state) do
    uptime = System.monotonic_time(:second) - state.start_time

    lines = [
      "# HELP honeycomb_requests_total Total number of requests",
      "# TYPE honeycomb_requests_total counter",
      "honeycomb_requests_total #{state.requests_total}",
      "",
      "# HELP honeycomb_requests_active Currently active requests",
      "# TYPE honeycomb_requests_active gauge",
      "honeycomb_requests_active #{state.requests_active}",
      "",
      "# HELP honeycomb_tokens_generated_total Total tokens generated",
      "# TYPE honeycomb_tokens_generated_total counter",
      "honeycomb_tokens_generated_total #{state.tokens_generated}",
      "",
      "# HELP honeycomb_queue_depth Current queue depth",
      "# TYPE honeycomb_queue_depth gauge",
      "honeycomb_queue_depth #{state.queue_depth}",
      "",
      "# HELP honeycomb_cache_hit_rate Cache hit rate",
      "# TYPE honeycomb_cache_hit_rate gauge",
      "honeycomb_cache_hit_rate #{cache_hit_rate(state)}",
      "",
      "# HELP honeycomb_kv_cache_utilization KV cache utilization",
      "# TYPE honeycomb_kv_cache_utilization gauge",
      "honeycomb_kv_cache_utilization #{kv_utilization(state)}",
      "",
      "# HELP honeycomb_uptime_seconds Server uptime",
      "# TYPE honeycomb_uptime_seconds gauge",
      "honeycomb_uptime_seconds #{uptime}"
    ]

    {:reply, Enum.join(lines, "\n"), state}
  end

  @impl true
  def handle_call(:reset, _from, _state) do
    {:reply, :ok, initial_state()}
  end

  # Private helpers

  defp initial_state do
    %__MODULE__{
      start_time: System.monotonic_time(:second),
      requests_total: 0,
      requests_active: 0,
      tokens_generated: 0,
      tokens_prefilled: 0,
      latency_histogram: new_histogram(),
      ttft_histogram: new_histogram(),
      tps_samples: [],
      queue_depth: 0,
      cache_hits: 0,
      cache_misses: 0,
      kv_blocks_used: 0,
      kv_blocks_total: 0,
      errors_total: 0
    }
  end

  defp new_histogram do
    buckets = Enum.map(@default_histogram_buckets, fn b -> {b, 0} end)
    %{buckets: buckets, sum: 0.0, count: 0, values: []}
  end

  defp add_to_histogram(histogram, value) do
    buckets = Enum.map(histogram.buckets, fn {threshold, count} ->
      if value <= threshold do
        {threshold, count + 1}
      else
        {threshold, count}
      end
    end)

    # Keep last 1000 values for percentile calculation
    values = Enum.take([value | histogram.values], 1000)

    %{histogram |
      buckets: buckets,
      sum: histogram.sum + value,
      count: histogram.count + 1,
      values: values
    }
  end

  defp percentile(%{values: []}, _p), do: 0.0

  defp percentile(%{values: values}, p) do
    sorted = Enum.sort(values)
    idx = round(p * (length(sorted) - 1))
    Enum.at(sorted, idx, 0.0)
  end

  defp cache_hit_rate(%{cache_hits: hits, cache_misses: misses}) do
    total = hits + misses
    if total > 0, do: hits / total, else: 0.0
  end

  defp kv_utilization(%{kv_blocks_used: used, kv_blocks_total: total}) do
    if total > 0, do: used / total, else: 0.0
  end

  defp average_tps(%{tps_samples: []}), do: 0.0

  defp average_tps(%{tps_samples: samples}) do
    Enum.sum(samples) / length(samples)
  end
end
