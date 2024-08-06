defmodule Mix.Tasks.Honeycomb.Profile do
  use Mix.Task

  @shortdoc "Profiles the given Honeycomb configuration"

  @prompt "Complete the following: The quick brown"

  @impl true
  def run(args) do
    Application.put_env(:honeycomb, :start_serving, true)
    :ok = parse_serving_args(args, [])

    Mix.Task.run("app.start")

    messages = [%{role: "user", content: @prompt}]
    opts = [messages: messages, stream: false]

    Mix.Tasks.Profile.Fprof.profile(
      fn ->
        Honeycomb.chat_completion(opts)
      end,
      details: true,
      callers: true
    )
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

  defp parse_serving_args([arg | _], _env) do
    raise "unknown serving argument #{arg}"
  end
end
