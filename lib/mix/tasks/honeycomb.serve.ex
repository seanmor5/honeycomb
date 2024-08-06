defmodule Mix.Tasks.Honeycomb.Serve do
  use Mix.Task

  @shortdoc "Starts the Honeycomb server"

  @impl true
  def run(args) do
    Application.put_env(:honeycomb, :start_serving, true)
    Application.put_env(:honeycomb, :start_router, true)

    :ok = parse_serving_args(args, [])
    Mix.Tasks.Run.run(["--no-halt"])
  end

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
