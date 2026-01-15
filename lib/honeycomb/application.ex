defmodule Honeycomb.Application do
  @moduledoc """
  Honeycomb Application.

  Starts all required services for LLM inference:
  - Model serving (Nx.Serving)
  - HTTP Router (Bandit)
  - Template cache
  - Metrics collector
  - Optional: Scheduler, KV Cache, Prefix Cache, Engine
  """

  use Application

  require Logger

  def start(_type, _args) do
    children = base_children()

    # Add serving if configured
    children = maybe_add_serving(children)

    # Add production components if configured
    children = maybe_add_production_components(children)

    # Add router last
    children = maybe_add_router(children)

    Logger.info("Honeycomb: Starting with #{length(children)} components")

    opts = [strategy: :one_for_one, name: Honeycomb.Supervisor]
    Supervisor.start_link(children, opts)
  end

  defp base_children do
    # Always start these components
    [
      Honeycomb.Metrics,
      Honeycomb.Templates
    ]
  end

  defp maybe_add_serving(children) do
    if get_env(:start_serving, false) do
      serving_config = Application.get_env(:honeycomb, Honeycomb.Serving, [])
      batch_size = Keyword.get(serving_config, :batch_size, 1)
      batch_timeout = Keyword.get(serving_config, :batch_timeout, 50)

      serving =
        {Nx.Serving,
         serving: Honeycomb.Serving.serving(),
         name: Honeycomb.Serving,
         batch_size: batch_size,
         batch_timeout: batch_timeout}

      [serving | children]
    else
      children
    end
  end

  defp maybe_add_production_components(children) do
    if get_env(:start_engine, false) do
      engine_config = Application.get_env(:honeycomb, Honeycomb.Engine, [])
      serving_config = Application.get_env(:honeycomb, Honeycomb.Serving, [])

      # KV Cache configuration
      kv_cache_config = [
        block_size: Keyword.get(engine_config, :block_size, 16),
        num_blocks: Keyword.get(engine_config, :max_kv_cache_blocks, 1000),
        num_layers: Keyword.get(serving_config, :num_layers, 32),
        num_heads: Keyword.get(serving_config, :num_heads, 32),
        head_dim: Keyword.get(serving_config, :head_dim, 128)
      ]

      # Scheduler configuration
      scheduler_config = [
        max_num_seqs: Keyword.get(engine_config, :max_num_seqs, 256),
        max_num_batched_tokens: Keyword.get(engine_config, :max_num_batched_tokens, 2048),
        max_model_len: Keyword.get(serving_config, :sequence_length, 4096)
      ]

      # Prefix cache configuration
      prefix_cache_config = [
        block_size: Keyword.get(engine_config, :block_size, 16),
        max_cached_blocks: Keyword.get(engine_config, :max_prefix_cache_blocks, 1000)
      ]

      production_children = [
        {Honeycomb.KVCache, kv_cache_config},
        {Honeycomb.Scheduler, scheduler_config},
        {Honeycomb.PrefixCache, prefix_cache_config},
        {Honeycomb.MemoryPool, [max_memory_gb: Keyword.get(engine_config, :max_memory_gb, 8)]},
        {Honeycomb.Engine, engine_config}
      ]

      # Filter based on configuration
      production_children =
        if Keyword.get(engine_config, :enable_prefix_caching, true) do
          production_children
        else
          Enum.reject(production_children, fn
            {Honeycomb.PrefixCache, _} -> true
            _ -> false
          end)
        end

      children ++ production_children
    else
      children
    end
  end

  defp maybe_add_router(children) do
    if get_env(:start_router, false) do
      port = Application.get_env(:honeycomb, :port, 4000)
      router = {Bandit, plug: Honeycomb.Router, port: port}
      children ++ [router]
    else
      children
    end
  end

  defp get_env(key, default) do
    case Application.fetch_env(:honeycomb, key) do
      {:ok, value} -> value
      :error -> default
    end
  end
end
