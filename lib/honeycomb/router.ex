defmodule Honeycomb.Router do
  use Plug.Router

  plug :match

  plug Plug.Parsers,
    parsers: [:json],
    json_decoder: {Jason, :decode!, [[keys: :atoms]]}

  plug Plug.Logger

  plug :dispatch

  post "/v1/chat/completions" do
    opts = Enum.into(conn.body_params, [])

    case Honeycomb.chat_completion(opts) do
      {:ok, response} -> json!(conn, 200, response)
      {:error, msg} -> json!(conn, 400, %{code: "bad_request", message: msg})
    end
  end

  defp json!(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> resp(status, Jason.encode!(data))
    |> send_resp()
  end
end
