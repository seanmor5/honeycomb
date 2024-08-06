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

    stream? = Keyword.get(opts, :stream, false)

    if stream? do
      do_stream_http_sse(conn, opts)
    else
      do_http_response(conn, opts)
    end
  end

  defp do_http_response(conn, opts) do
    case Honeycomb.chat_completion(opts) do
      {:ok, response} -> json!(conn, 200, response)
      {:error, msg} -> json!(conn, 400, %{code: "bad_request", message: msg})
    end
  end

  defp do_stream_http_sse(conn, opts) do
    case Honeycomb.chat_completion(opts) do
      {:error, msg} ->
        json!(conn, 400, %{code: "bad_request", message: msg})

      stream ->
        # TODO: This is not really SSE
        Enum.reduce_while(stream, send_chunked(conn, 200), fn chunk, conn ->
          data = Jason.encode!(chunk)

          case chunk(conn, data) do
            {:ok, conn} ->
              {:cont, conn}

            _ ->
              {:halt, conn}
          end
        end)
    end
  end

  defp json!(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> resp(status, Jason.encode!(data))
    |> send_resp()
  end
end
