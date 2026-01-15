defmodule Honeycomb.ChunkedPrefill do
  @moduledoc """
  Chunked Prefill implementation for reduced time-to-first-token.

  Long prompts can cause high latency before the first token is generated.
  Chunked prefill breaks up the prefill phase into smaller chunks that can
  be interleaved with decode steps from other requests.

  ## How It Works

  1. Split long prompts into fixed-size chunks
  2. Process chunks incrementally, updating KV cache
  3. Interleave with decode steps from running requests
  4. This keeps decode latency consistent regardless of prompt length

  ## Configuration

    * `:max_chunk_size` - Maximum tokens per prefill chunk (default: 512)
    * `:enable_chunking` - Enable/disable chunked prefill (default: true)
  """

  require Logger

  defstruct [
    :sequence_id,
    :tokens,
    :processed_tokens,
    :chunk_size,
    :total_chunks,
    :current_chunk,
    :status
  ]

  @default_chunk_size 512

  @doc """
  Creates a new chunked prefill state for a sequence.

  ## Options

    * `:chunk_size` - Tokens per chunk (default: 512)
  """
  def new(sequence_id, tokens, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    total = length(tokens)
    num_chunks = div(total + chunk_size - 1, chunk_size)

    %__MODULE__{
      sequence_id: sequence_id,
      tokens: tokens,
      processed_tokens: 0,
      chunk_size: chunk_size,
      total_chunks: num_chunks,
      current_chunk: 0,
      status: :pending
    }
  end

  @doc """
  Checks if chunked prefill is needed for the given token count.
  """
  def needs_chunking?(num_tokens, chunk_size \\ @default_chunk_size) do
    num_tokens > chunk_size
  end

  @doc """
  Gets the next chunk of tokens to process.

  Returns `{:ok, tokens, updated_state}` or `:done` if complete.
  """
  def next_chunk(%__MODULE__{status: :complete}), do: :done

  def next_chunk(%__MODULE__{} = state) do
    start_idx = state.processed_tokens
    end_idx = min(start_idx + state.chunk_size, length(state.tokens))
    chunk_tokens = Enum.slice(state.tokens, start_idx, end_idx - start_idx)

    new_processed = end_idx

    new_status =
      if new_processed >= length(state.tokens) do
        :complete
      else
        :in_progress
      end

    state = %{state |
      processed_tokens: new_processed,
      current_chunk: state.current_chunk + 1,
      status: new_status
    }

    {:ok, chunk_tokens, state}
  end

  @doc """
  Returns progress as a percentage.
  """
  def progress(%__MODULE__{tokens: tokens, processed_tokens: processed}) do
    total = length(tokens)
    if total > 0, do: processed / total * 100, else: 100.0
  end

  @doc """
  Returns remaining tokens to process.
  """
  def remaining_tokens(%__MODULE__{tokens: tokens, processed_tokens: processed}) do
    length(tokens) - processed
  end

  @doc """
  Checks if prefill is complete.
  """
  def complete?(%__MODULE__{status: :complete}), do: true
  def complete?(%__MODULE__{}), do: false

  @doc """
  Checks if this is the first chunk.
  """
  def first_chunk?(%__MODULE__{current_chunk: 0}), do: true
  def first_chunk?(%__MODULE__{}), do: false

  @doc """
  Checks if this is the last chunk.
  """
  def last_chunk?(%__MODULE__{} = state) do
    state.current_chunk >= state.total_chunks - 1
  end

  @doc """
  Creates chunks from a list of tokens.
  """
  def chunk_tokens(tokens, chunk_size \\ @default_chunk_size) do
    Enum.chunk_every(tokens, chunk_size)
  end

  @doc """
  Schedules prefill chunks with interleaved decode.

  Takes a list of prefill states and decode requests, returns
  an ordered schedule that prioritizes decode latency.
  """
  def schedule_interleaved(prefill_states, decode_requests, max_tokens_per_step) do
    schedule_interleaved(prefill_states, decode_requests, max_tokens_per_step, [])
  end

  defp schedule_interleaved([], [], _max_tokens, schedule) do
    Enum.reverse(schedule)
  end

  defp schedule_interleaved(prefills, decodes, max_tokens, schedule) do
    # Always prioritize decode (one token per request)
    decode_tokens = length(decodes)
    remaining_tokens = max_tokens - decode_tokens

    # Add decode batch if any
    schedule =
      if decode_tokens > 0 do
        [{:decode, decodes} | schedule]
      else
        schedule
      end

    # Add prefill chunk if we have capacity
    {schedule, prefills} =
      if remaining_tokens > 0 and length(prefills) > 0 do
        [first | rest] = prefills

        case next_chunk(first) do
          {:ok, chunk_tokens, updated_state} ->
            chunk = Enum.take(chunk_tokens, remaining_tokens)
            entry = {:prefill, first.sequence_id, chunk}

            if complete?(updated_state) do
              {[entry | schedule], rest}
            else
              {[entry | schedule], [updated_state | rest]}
            end

          :done ->
            {schedule, rest}
        end
      else
        {schedule, prefills}
      end

    # Continue scheduling
    if prefills == [] and decodes == [] do
      Enum.reverse(schedule)
    else
      # In real implementation, would wait for next iteration
      Enum.reverse(schedule)
    end
  end

  @doc """
  Estimates time savings from chunked prefill.

  Returns estimated reduction in decode latency variance.
  """
  def estimate_latency_improvement(prompt_lengths, chunk_size, decode_time_ms) do
    # Without chunking: max_prompt_length * prefill_time blocks decode
    # With chunking: only chunk_size * prefill_time blocks at a time

    max_prompt = Enum.max(prompt_lengths, fn -> 0 end)

    without_chunking = max_prompt * 0.1  # ~0.1ms per token prefill
    with_chunking = chunk_size * 0.1

    %{
      max_decode_delay_without: without_chunking,
      max_decode_delay_with: with_chunking,
      improvement_factor: if(with_chunking > 0, do: without_chunking / with_chunking, else: 1.0)
    }
  end
end
