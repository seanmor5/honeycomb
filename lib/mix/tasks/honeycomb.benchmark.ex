defmodule Mix.Tasks.Honeycomb.Benchmark do
  @moduledoc """
  Benchmarks the given Honeycomb configuration.

  ## Usage

      mix honeycomb.benchmark --model=microsoft/Phi-3-mini-4k-instruct --chat-template=phi3

  ## Options

  #{Honeycomb.CLI.usage()}
  """

  use Mix.Task

  @shortdoc "Benchmarks the given Honeycomb configuration"

  @iterations 5
  @prompt "Complete the following: The quick brown"

  @impl true
  def run(args) do
    case Honeycomb.CLI.parse_args(args) do
      {:ok, config} ->
        run_benchmark(config)

      {:help, usage} ->
        Mix.shell().info(usage)

      {:error, reason} ->
        Mix.shell().error("Error: #{reason}")
        Mix.shell().info(Honeycomb.CLI.usage())
        exit({:shutdown, 1})
    end
  end

  defp run_benchmark(config) do
    # Apply configuration
    :ok = Honeycomb.CLI.apply_config(config)

    Application.put_env(:honeycomb, :start_serving, true)

    start_time = :erlang.monotonic_time()
    Mix.Task.run("app.start")
    end_time = :erlang.monotonic_time()

    startup_time_ms = :erlang.convert_time_unit(end_time - start_time, :native, :millisecond)

    Mix.shell().info("App started. Total startup time: #{startup_time_ms}ms")
    Mix.shell().info("Benchmarking chat completions for #{@iterations} iterations")
    Mix.shell().info("Model: #{config.model}")
    Mix.shell().info("Completion prompt: #{@prompt}")
    Mix.shell().info("")

    per_iteration_results =
      Enum.map(0..@iterations, fn i ->
        res = benchmark()

        unless i == 0 do
          Mix.shell().info("Iteration #{i} Results")
          inspect_results(res)
        end

        res
      end)

    zeros = %{total_tokens: 0, average_time_per_token: 0, time_to_first_token: 0, duration: 0}

    average_results =
      per_iteration_results
      |> tl()
      |> Enum.reduce(zeros, fn res, zeros ->
        Map.merge(res, zeros, fn _, v1, v2 -> v1 + v2 end)
      end)
      |> Map.new(fn {k, v} -> {k, v / @iterations} end)

    Mix.shell().info("")
    Mix.shell().info("Aggregate Results")
    inspect_results(average_results)

    # Output metrics summary
    if Process.whereis(Honeycomb.Metrics) do
      metrics = Honeycomb.Metrics.get_all()
      Mix.shell().info("")
      Mix.shell().info("Metrics Summary")
      Mix.shell().info("  Total tokens generated: #{metrics.tokens_generated}")
      Mix.shell().info("  Avg throughput: #{Float.round(metrics.throughput_tps, 2)} tok/s")
    end
  end

  defp benchmark() do
    messages = [%{role: "user", content: @prompt}]
    stream = Honeycomb.chat_completion(messages: messages, stream: true)

    start_time = :erlang.monotonic_time()

    {_, times} =
      Enum.reduce(stream, {start_time, []}, fn _text, {last_token_time, times} ->
        token_time = :erlang.monotonic_time()
        {token_time, [token_time - last_token_time | times]}
      end)

    end_time = :erlang.monotonic_time()

    [time_to_first_token | rest] =
      times
      |> Enum.reverse()
      |> Enum.map(&:erlang.convert_time_unit(&1, :native, :nanosecond))

    duration = :erlang.convert_time_unit(end_time - start_time, :native, :nanosecond)

    average_time_per_token_ms = Enum.sum(rest) / max(length(rest), 1) / 1_000_000
    time_to_first_token_ms = time_to_first_token / 1_000_000
    duration_ms = duration / 1_000_000

    %{
      total_tokens: length(rest) + 1,
      average_time_per_token: average_time_per_token_ms,
      time_to_first_token: time_to_first_token_ms,
      duration: duration_ms
    }
  end

  defp inspect_results(res) do
    Mix.shell().info(
      "  time_per_token=#{float_format(res.average_time_per_token)}ms  " <>
      "time_to_first_token=#{float_format(res.time_to_first_token)}ms  " <>
      "duration=#{float_format(res.duration)}ms  " <>
      "tokens=#{res.total_tokens}"
    )
  end

  defp float_format(float) do
    :io_lib.format(~c"~.2f", [float]) |> to_string()
  end
end
