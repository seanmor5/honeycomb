defmodule Honeycomb.Scheduler.SchedulerOutput do
  @moduledoc """
  Output from a scheduler iteration.

  Contains the sequences to process in the current iteration,
  separated into prefill (new sequences) and decode (continuing sequences).
  """

  defstruct [
    :prefill_requests,
    :decode_requests,
    :finished_request_ids,
    :num_prefill_tokens,
    :num_decode_tokens,
    :is_empty
  ]

  alias Honeycomb.Scheduler.Request

  @doc """
  Creates a new scheduler output.
  """
  def new(prefill_requests, decode_requests, finished_ids) do
    num_prefill = Enum.reduce(prefill_requests, 0, fn req, acc ->
      acc + Request.prompt_length(req)
    end)

    num_decode = length(decode_requests)

    %__MODULE__{
      prefill_requests: prefill_requests,
      decode_requests: decode_requests,
      finished_request_ids: finished_ids,
      num_prefill_tokens: num_prefill,
      num_decode_tokens: num_decode,
      is_empty: Enum.empty?(prefill_requests) and Enum.empty?(decode_requests)
    }
  end

  @doc """
  Returns all requests to process (prefill + decode).
  """
  def all_requests(%__MODULE__{} = output) do
    output.prefill_requests ++ output.decode_requests
  end

  @doc """
  Returns total number of tokens to process.
  """
  def total_tokens(%__MODULE__{} = output) do
    output.num_prefill_tokens + output.num_decode_tokens
  end

  @doc """
  Returns number of sequences to process.
  """
  def num_sequences(%__MODULE__{} = output) do
    length(output.prefill_requests) + length(output.decode_requests)
  end

  @doc """
  Checks if there are any requests to process.
  """
  def empty?(%__MODULE__{is_empty: is_empty}), do: is_empty

  @doc """
  Returns request IDs for all requests.
  """
  def request_ids(%__MODULE__{} = output) do
    prefill_ids = Enum.map(output.prefill_requests, & &1.id)
    decode_ids = Enum.map(output.decode_requests, & &1.id)
    prefill_ids ++ decode_ids
  end

  @doc """
  Groups requests by their position in the batch for padding.
  """
  def batch_info(%__MODULE__{} = output) do
    requests = all_requests(output)

    max_prompt_len =
      requests
      |> Enum.map(&Request.prompt_length/1)
      |> Enum.max(fn -> 0 end)

    max_total_len =
      requests
      |> Enum.map(&Request.num_tokens/1)
      |> Enum.max(fn -> 0 end)

    %{
      batch_size: length(requests),
      max_prompt_len: max_prompt_len,
      max_total_len: max_total_len,
      request_ids: Enum.map(requests, & &1.id)
    }
  end
end
