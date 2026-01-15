defmodule Honeycomb.Scheduler.Request do
  @moduledoc """
  Represents a single inference request in the scheduler.

  Tracks the request state, tokens, and timing information.
  """

  @type status :: :waiting | :running | :preempted | :finished | :aborted

  defstruct [
    :id,
    :prompt_tokens,
    :output_tokens,
    :max_tokens,
    :priority,
    :arrival_time,
    :start_time,
    :finish_time,
    :status,
    :num_preemptions,
    :opts
  ]

  @doc """
  Creates a new request.
  """
  def new(id, prompt_tokens, max_tokens, priority, arrival_time, opts \\ []) do
    %__MODULE__{
      id: id,
      prompt_tokens: prompt_tokens,
      output_tokens: [],
      max_tokens: max_tokens,
      priority: priority,
      arrival_time: arrival_time,
      start_time: nil,
      finish_time: nil,
      status: :waiting,
      num_preemptions: 0,
      opts: opts
    }
  end

  @doc """
  Transitions request to running state.
  """
  def start_running(%__MODULE__{status: :waiting} = req) do
    %{req | status: :running, start_time: System.monotonic_time(:microsecond)}
  end

  def start_running(%__MODULE__{status: :preempted} = req) do
    %{req | status: :running}
  end

  def start_running(%__MODULE__{} = req), do: req

  @doc """
  Adds a generated token to the output.
  """
  def add_token(%__MODULE__{} = req, token_id) do
    %{req | output_tokens: req.output_tokens ++ [token_id]}
  end

  @doc """
  Marks the request as finished.
  """
  def finish(%__MODULE__{} = req) do
    %{req |
      status: :finished,
      finish_time: System.monotonic_time(:microsecond)
    }
  end

  @doc """
  Marks the request as preempted.
  """
  def preempt(%__MODULE__{} = req) do
    %{req |
      status: :preempted,
      num_preemptions: req.num_preemptions + 1
    }
  end

  @doc """
  Marks the request as aborted.
  """
  def abort(%__MODULE__{} = req) do
    %{req |
      status: :aborted,
      finish_time: System.monotonic_time(:microsecond)
    }
  end

  @doc """
  Checks if request is finished.
  """
  def finished?(%__MODULE__{status: :finished}), do: true
  def finished?(%__MODULE__{}), do: false

  @doc """
  Checks if request is aborted.
  """
  def aborted?(%__MODULE__{status: :aborted}), do: true
  def aborted?(%__MODULE__{}), do: false

  @doc """
  Checks if request has reached max tokens.
  """
  def at_max_tokens?(%__MODULE__{} = req) do
    length(req.output_tokens) >= req.max_tokens
  end

  @doc """
  Returns total number of tokens (prompt + output).
  """
  def num_tokens(%__MODULE__{} = req) do
    length(req.prompt_tokens) + length(req.output_tokens)
  end

  @doc """
  Returns number of generated tokens.
  """
  def num_generated(%__MODULE__{output_tokens: tokens}), do: length(tokens)

  @doc """
  Returns prompt length.
  """
  def prompt_length(%__MODULE__{prompt_tokens: tokens}), do: length(tokens)

  @doc """
  Returns all tokens (prompt + output).
  """
  def all_tokens(%__MODULE__{} = req) do
    req.prompt_tokens ++ req.output_tokens
  end

  @doc """
  Calculates latency in microseconds.
  """
  def latency(%__MODULE__{arrival_time: arrival, finish_time: finish})
      when not is_nil(finish) do
    finish - arrival
  end

  def latency(%__MODULE__{}), do: nil

  @doc """
  Calculates time to first token in microseconds.
  """
  def time_to_first_token(%__MODULE__{arrival_time: arrival, start_time: start})
      when not is_nil(start) do
    start - arrival
  end

  def time_to_first_token(%__MODULE__{}), do: nil

  @doc """
  Returns tokens per second throughput.
  """
  def tokens_per_second(%__MODULE__{} = req) do
    case {latency(req), num_generated(req)} do
      {lat, gen} when not is_nil(lat) and lat > 0 and gen > 0 ->
        gen / (lat / 1_000_000)

      _ ->
        nil
    end
  end
end
