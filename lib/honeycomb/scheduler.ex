defmodule Honeycomb.Scheduler do
  @moduledoc """
  Continuous Batching Scheduler for LLM inference.

  Implements vLLM-style iteration-level scheduling that enables:

  1. **Dynamic batching**: Requests can join/leave the batch at any iteration
  2. **Preemption**: Low-priority or long-running requests can be preempted
  3. **Memory-aware scheduling**: Respects KV cache capacity limits
  4. **Priority queues**: Supports request priorities

  ## Scheduling Algorithm

  Each iteration (forward pass), the scheduler:
  1. Checks for completed sequences and removes them
  2. Attempts to add waiting sequences if memory permits
  3. Handles preemption if memory is exhausted
  4. Returns the batch of sequences for the current iteration

  ## Request States

  - `:waiting` - In queue, not yet started
  - `:running` - Currently generating tokens
  - `:preempted` - Was running but preempted for memory
  - `:finished` - Generation complete
  """

  use GenServer
  require Logger

  alias Honeycomb.Scheduler.{Request, SchedulerConfig, SchedulerOutput}

  defstruct [
    :config,
    :waiting_queue,
    :running_queue,
    :preempted_queue,
    :request_map,
    :next_request_id
  ]

  # Client API

  @doc """
  Starts the scheduler.

  ## Options

    * `:max_num_seqs` - Maximum concurrent sequences (default: 256)
    * `:max_num_batched_tokens` - Max tokens per batch (default: 2048)
    * `:max_model_len` - Maximum sequence length (default: 4096)
    * `:preemption_mode` - :recompute or :swap (default: :recompute)
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adds a new request to the scheduler.

  Returns `{:ok, request_id}`.
  """
  def add_request(prompt_tokens, opts \\ []) do
    GenServer.call(__MODULE__, {:add_request, prompt_tokens, opts})
  end

  @doc """
  Aborts a request.
  """
  def abort_request(request_id) do
    GenServer.call(__MODULE__, {:abort_request, request_id})
  end

  @doc """
  Runs one scheduling iteration.

  Returns a `SchedulerOutput` with sequences to run.
  """
  def schedule do
    GenServer.call(__MODULE__, :schedule)
  end

  @doc """
  Updates a sequence after a forward pass (adds generated token).
  """
  def update_sequence(request_id, new_token_id, finished? \\ false) do
    GenServer.call(__MODULE__, {:update_sequence, request_id, new_token_id, finished?})
  end

  @doc """
  Returns scheduler statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Returns the number of waiting requests.
  """
  def num_waiting do
    GenServer.call(__MODULE__, :num_waiting)
  end

  @doc """
  Returns the number of running requests.
  """
  def num_running do
    GenServer.call(__MODULE__, :num_running)
  end

  @doc """
  Checks if there are pending requests.
  """
  def has_pending? do
    GenServer.call(__MODULE__, :has_pending?)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    config = SchedulerConfig.new(opts)

    state = %__MODULE__{
      config: config,
      waiting_queue: :queue.new(),
      running_queue: :queue.new(),
      preempted_queue: :queue.new(),
      request_map: %{},
      next_request_id: 0
    }

    Logger.info("Scheduler: Started with max_seqs=#{config.max_num_seqs}, max_tokens=#{config.max_num_batched_tokens}")

    {:ok, state}
  end

  @impl true
  def handle_call({:add_request, prompt_tokens, opts}, _from, state) do
    request_id = state.next_request_id
    arrival_time = System.monotonic_time(:microsecond)

    request = Request.new(
      request_id,
      prompt_tokens,
      Keyword.get(opts, :max_tokens, state.config.max_model_len),
      Keyword.get(opts, :priority, 0),
      arrival_time,
      opts
    )

    state = %{state |
      waiting_queue: :queue.in(request_id, state.waiting_queue),
      request_map: Map.put(state.request_map, request_id, request),
      next_request_id: request_id + 1
    }

    {:reply, {:ok, request_id}, state}
  end

  @impl true
  def handle_call({:abort_request, request_id}, _from, state) do
    case Map.fetch(state.request_map, request_id) do
      {:ok, request} ->
        request = Request.abort(request)
        state = %{state | request_map: Map.put(state.request_map, request_id, request)}
        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:schedule, _from, state) do
    {output, state} = do_schedule(state)
    {:reply, output, state}
  end

  @impl true
  def handle_call({:update_sequence, request_id, new_token_id, finished?}, _from, state) do
    case Map.fetch(state.request_map, request_id) do
      {:ok, request} ->
        request = Request.add_token(request, new_token_id)

        request =
          if finished? do
            Request.finish(request)
          else
            request
          end

        state = %{state | request_map: Map.put(state.request_map, request_id, request)}
        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      num_waiting: :queue.len(state.waiting_queue),
      num_running: :queue.len(state.running_queue),
      num_preempted: :queue.len(state.preempted_queue),
      total_requests: map_size(state.request_map)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:num_waiting, _from, state) do
    {:reply, :queue.len(state.waiting_queue), state}
  end

  @impl true
  def handle_call(:num_running, _from, state) do
    {:reply, :queue.len(state.running_queue), state}
  end

  @impl true
  def handle_call(:has_pending?, _from, state) do
    has_pending = :queue.len(state.waiting_queue) > 0 or :queue.len(state.running_queue) > 0
    {:reply, has_pending, state}
  end

  # Private scheduling logic

  defp do_schedule(state) do
    # Step 1: Remove finished sequences from running queue
    {running_queue, finished_ids, state} = process_finished(state)
    state = %{state | running_queue: running_queue}

    # Step 2: Try to schedule preempted sequences first (higher priority)
    {preempted_to_run, preempted_queue, state} = schedule_preempted(state)
    state = %{state | preempted_queue: preempted_queue}

    # Step 3: Schedule waiting sequences
    {waiting_to_run, waiting_queue, state} = schedule_waiting(state)
    state = %{state | waiting_queue: waiting_queue}

    # Step 4: Collect sequences to run
    to_prefill = preempted_to_run ++ waiting_to_run
    running_ids = :queue.to_list(state.running_queue)

    # Add newly scheduled sequences to running queue
    running_queue =
      Enum.reduce(to_prefill, state.running_queue, fn id, q ->
        :queue.in(id, q)
      end)

    state = %{state | running_queue: running_queue}

    # Step 5: Check if we need to preempt
    {to_decode, state} = maybe_preempt(running_ids, state)

    # Build output
    prefill_requests = Enum.map(to_prefill, &Map.fetch!(state.request_map, &1))
    decode_requests = Enum.map(to_decode, &Map.fetch!(state.request_map, &1))

    output = SchedulerOutput.new(
      prefill_requests,
      decode_requests,
      finished_ids
    )

    {output, state}
  end

  defp process_finished(state) do
    {new_queue, finished_ids} =
      :queue.to_list(state.running_queue)
      |> Enum.reduce({:queue.new(), []}, fn id, {q, finished} ->
        case Map.fetch(state.request_map, id) do
          {:ok, request} ->
            if Request.finished?(request) or Request.aborted?(request) do
              {q, [id | finished]}
            else
              {:queue.in(id, q), finished}
            end

          :error ->
            {q, finished}
        end
      end)

    {new_queue, Enum.reverse(finished_ids), state}
  end

  defp schedule_preempted(state) do
    schedule_from_queue(
      state.preempted_queue,
      state,
      state.config.max_num_seqs - :queue.len(state.running_queue)
    )
  end

  defp schedule_waiting(state) do
    available_slots = state.config.max_num_seqs - :queue.len(state.running_queue)
    schedule_from_queue(state.waiting_queue, state, available_slots)
  end

  defp schedule_from_queue(queue, state, max_to_schedule) when max_to_schedule <= 0 do
    {[], queue, state}
  end

  defp schedule_from_queue(queue, state, max_to_schedule) do
    {to_run, remaining, _count} =
      :queue.to_list(queue)
      |> Enum.reduce({[], [], 0}, fn id, {to_run, remaining, count} ->
        if count < max_to_schedule do
          case Map.fetch(state.request_map, id) do
            {:ok, request} ->
              if can_schedule?(request, state) do
                request = Request.start_running(request)
                state = %{state | request_map: Map.put(state.request_map, id, request)}
                {[id | to_run], remaining, count + 1}
              else
                {to_run, [id | remaining], count}
              end

            :error ->
              {to_run, remaining, count}
          end
        else
          {to_run, [id | remaining], count}
        end
      end)

    {Enum.reverse(to_run), :queue.from_list(Enum.reverse(remaining)), state}
  end

  defp can_schedule?(request, state) do
    # Check if we have capacity for this request
    current_running = :queue.len(state.running_queue)
    current_tokens = count_running_tokens(state)

    current_running < state.config.max_num_seqs and
      current_tokens + Request.num_tokens(request) <= state.config.max_num_batched_tokens
  end

  defp count_running_tokens(state) do
    :queue.to_list(state.running_queue)
    |> Enum.reduce(0, fn id, acc ->
      case Map.fetch(state.request_map, id) do
        {:ok, request} -> acc + Request.num_tokens(request)
        :error -> acc
      end
    end)
  end

  defp maybe_preempt(running_ids, state) do
    # Simple preemption: if over memory budget, preempt lowest priority
    current_tokens = count_running_tokens(state)

    if current_tokens > state.config.max_num_batched_tokens do
      # Sort by priority (lowest first) and preempt
      sorted =
        running_ids
        |> Enum.map(fn id -> {id, Map.fetch!(state.request_map, id)} end)
        |> Enum.sort_by(fn {_id, req} -> {req.priority, -req.arrival_time} end)

      {to_preempt, to_keep, _} =
        Enum.reduce(sorted, {[], [], 0}, fn {id, req}, {preempt, keep, tokens} ->
          if tokens + Request.num_tokens(req) <= state.config.max_num_batched_tokens do
            {preempt, [id | keep], tokens + Request.num_tokens(req)}
          else
            {[id | preempt], keep, tokens}
          end
        end)

      # Update preempted requests
      state =
        Enum.reduce(to_preempt, state, fn id, s ->
          request = Map.fetch!(s.request_map, id) |> Request.preempt()
          preempted_queue = :queue.in(id, s.preempted_queue)
          %{s |
            request_map: Map.put(s.request_map, id, request),
            preempted_queue: preempted_queue
          }
        end)

      {Enum.reverse(to_keep), state}
    else
      {running_ids, state}
    end
  end
end
