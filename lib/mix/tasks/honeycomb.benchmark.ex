defmodule Mix.Tasks.Honeycomb.Benchmark do
  use Mix.Task

  @shortdoc "Benchmarks the given Honeycomb configuration"
  @iterations 5

  @prompt "Complete the following: The quick brown"

  @impl true
  def run(args) do
    Application.put_env(:honeycomb, :start_serving, true)
    :ok = parse_serving_args(args, [])

    start_time = :erlang.monotonic_time()
    Mix.Task.run("app.start")
    end_time = :erlang.monotonic_time()

    startup_time_ms = :erlang.convert_time_unit(end_time - start_time, :native, :millisecond)

    IO.puts("App started. Total startup time: #{startup_time_ms}")
    IO.puts("Benchmarking chat completions for #{@iterations} iterations")
    IO.puts("Completion prompt: #{@prompt}")

    per_iteration_results =
      Enum.map(0..@iterations, fn i ->
        res = benchmark()

        unless i == 0 do
          IO.puts("Iteration #{i} Results")
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

    IO.puts("\nAggregate Results")
    inspect_results(average_results)
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

    average_time_per_token_ms = Enum.sum(rest) / length(rest) / 1_000_000
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
    IO.write("time_per_token=#{float_format(res.average_time_per_token)}ms\t")
    IO.write("time_to_first_token=#{float_format(res.time_to_first_token)}ms\t")
    IO.write("duration=#{float_format(res.duration)}ms\n")
  end

  defp float_format(float) do
    :io_lib.format(~c"~.2f", [float])
  end

  # TODO: Do not duplicate this
  defp parse_serving_args([], env), do: Application.put_env(:honeycomb, Honeycomb.Serving, env)

  defp parse_serving_args(["--model=" <> model_id | args], env) do
    env = Keyword.put(env, :model, model_id)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--model", model_id | args], env) do
    env = Keyword.put(env, :model, model_id)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--chat-template=" <> template | args], env) do
    env = Keyword.put(env, :chat_template, template)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--chat-template", template | args], env) do
    env = Keyword.put(env, :chat_template, template)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--hf-auth-token=" <> token | args], env) do
    env = Keyword.put(env, :auth_token, token)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--hf-auth-token", token | args], env) do
    env = Keyword.put(env, :auth_token, token)
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--max-sequence-length=" <> seqlen | args], env) do
    env = Keyword.put(env, :sequence_length, String.to_integer(seqlen))
    parse_serving_args(args, env)
  end

  defp parse_serving_args(["--max-sequence-length", seqlen | args], env) do
    env = Keyword.put(env, :sequence_length, String.to_integer(seqlen))
    parse_serving_args(args, env)
  end

  defp parse_serving_args([arg | _], _env) do
    raise "unknown serving argument #{arg}"
  end
end
