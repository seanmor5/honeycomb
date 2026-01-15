defmodule Honeycomb.Scheduler.SchedulerConfig do
  @moduledoc """
  Configuration for the scheduler.
  """

  defstruct [
    :max_num_seqs,
    :max_num_batched_tokens,
    :max_model_len,
    :max_paddings,
    :preemption_mode,
    :delay_factor,
    :enable_chunked_prefill,
    :max_prefill_tokens
  ]

  @default_max_num_seqs 256
  @default_max_num_batched_tokens 2048
  @default_max_model_len 4096
  @default_max_paddings 256
  @default_preemption_mode :recompute
  @default_delay_factor 0.0
  @default_max_prefill_tokens 512

  @doc """
  Creates a new scheduler configuration.

  ## Options

    * `:max_num_seqs` - Maximum number of concurrent sequences (default: 256)
    * `:max_num_batched_tokens` - Maximum tokens per iteration (default: 2048)
    * `:max_model_len` - Maximum sequence length (default: 4096)
    * `:max_paddings` - Maximum padding tokens (default: 256)
    * `:preemption_mode` - :recompute or :swap (default: :recompute)
    * `:delay_factor` - Scheduling delay factor for SJF (default: 0.0)
    * `:enable_chunked_prefill` - Enable chunked prefill (default: false)
    * `:max_prefill_tokens` - Max tokens per prefill chunk (default: 512)
  """
  def new(opts \\ []) do
    %__MODULE__{
      max_num_seqs: Keyword.get(opts, :max_num_seqs, @default_max_num_seqs),
      max_num_batched_tokens: Keyword.get(opts, :max_num_batched_tokens, @default_max_num_batched_tokens),
      max_model_len: Keyword.get(opts, :max_model_len, @default_max_model_len),
      max_paddings: Keyword.get(opts, :max_paddings, @default_max_paddings),
      preemption_mode: Keyword.get(opts, :preemption_mode, @default_preemption_mode),
      delay_factor: Keyword.get(opts, :delay_factor, @default_delay_factor),
      enable_chunked_prefill: Keyword.get(opts, :enable_chunked_prefill, false),
      max_prefill_tokens: Keyword.get(opts, :max_prefill_tokens, @default_max_prefill_tokens)
    }
  end

  @doc """
  Validates the configuration.
  """
  def validate(%__MODULE__{} = config) do
    cond do
      config.max_num_seqs <= 0 ->
        {:error, "max_num_seqs must be positive"}

      config.max_num_batched_tokens <= 0 ->
        {:error, "max_num_batched_tokens must be positive"}

      config.max_model_len <= 0 ->
        {:error, "max_model_len must be positive"}

      config.preemption_mode not in [:recompute, :swap] ->
        {:error, "preemption_mode must be :recompute or :swap"}

      true ->
        :ok
    end
  end
end
