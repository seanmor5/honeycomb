defmodule Honeycomb.Controller.OpenAI do
  @moduledoc false

  alias Honeycomb.Serving

  defmodule Response do
    defstruct [:choices, :created, :id, :model, :object, :usage]

    def new(model, %{text: text, token_summary: usage}) do
      struct(__MODULE__,
        choices: choices(text),
        created: System.os_time(:second),
        id: id(),
        model: model,
        object: "chat.completion",
        usage: usage(usage)
      )
    end

    defp id() do
      encoded = Base.url_encode64(:crypto.strong_rand_bytes(21), padding: false)
      "honeycomb-#{String.slice(encoded, 0..21)}"
    end

    defp choices(generation) do
      %{
        finish_reason: "stop",
        index: 0,
        message: %{
          content: generation,
          role: "assistant"
        },
        logprobs: nil
      }
    end

    defp usage(%{input: inp, output: out}) do
      %{
        completion_tokens: out,
        prompt_tokens: inp,
        total_tokens: inp + out
      }
    end
  end

  import Plug.Conn

  # Implements logic for OpenAI compatible endpoints

  def chat_completion(conn) do
    case conn.body_params do
      %{"messages" => messages} ->
        %{results: [generation]} =
          messages
          |> Enum.map(&Map.new(&1, fn {k, v} -> {String.to_atom(k), v} end))
          |> Serving.generate()

        response = Response.new(Serving.model(), generation)
        json!(conn, 200, Map.from_struct(response))

      _ ->
        json!(conn, 400, %{code: "bad_request"})
    end
  end

  defp json!(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> resp(status, Jason.encode!(data))
    |> send_resp()
  end
end
