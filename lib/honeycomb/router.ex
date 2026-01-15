defmodule Honeycomb.Router do
  @moduledoc """
  HTTP Router for OpenAI-compatible API.

  Provides endpoints for:
  - Chat completions (streaming and non-streaming)
  - Health checks
  - Metrics
  - Model information
  """

  use Plug.Router

  alias Honeycomb.{Engine, Metrics}

  plug :match

  plug Plug.Parsers,
    parsers: [:json],
    json_decoder: {Jason, :decode!, [[keys: :atoms]]}

  plug Plug.Logger

  plug :dispatch

  # OpenAI-compatible endpoints

  post "/v1/chat/completions" do
    opts = Enum.into(conn.body_params, [])

    stream? = Keyword.get(opts, :stream, false)

    if stream? do
      do_stream_http_sse(conn, opts)
    else
      do_http_response(conn, opts)
    end
  end

  get "/v1/models" do
    model = Honeycomb.Serving.model()

    response = %{
      object: "list",
      data: [
        %{
          id: model,
          object: "model",
          created: System.os_time(:second),
          owned_by: "honeycomb"
        }
      ]
    }

    json!(conn, 200, response)
  end

  # Health check endpoints

  get "/health" do
    case get_health_status() do
      %{status: :healthy} = health ->
        json!(conn, 200, health)

      %{status: :degraded} = health ->
        json!(conn, 503, health)

      health ->
        json!(conn, 503, health)
    end
  end

  get "/health/live" do
    # Liveness probe - is the process running?
    json!(conn, 200, %{status: "alive"})
  end

  get "/health/ready" do
    # Readiness probe - is the server ready to accept requests?
    ready = is_ready?()

    if ready do
      json!(conn, 200, %{status: "ready"})
    else
      json!(conn, 503, %{status: "not_ready"})
    end
  end

  # Metrics endpoint

  get "/metrics" do
    metrics = get_metrics()

    conn
    |> put_resp_content_type("text/plain")
    |> resp(200, metrics)
    |> send_resp()
  end

  get "/stats" do
    stats = get_stats()
    json!(conn, 200, stats)
  end

  # Catch-all for unmatched routes

  match _ do
    json!(conn, 404, %{error: "not_found", message: "Endpoint not found"})
  end

  # Private helpers

  defp do_http_response(conn, opts) do
    start_time = System.monotonic_time(:millisecond)

    case Honeycomb.chat_completion(opts) do
      {:ok, response} ->
        duration = System.monotonic_time(:millisecond) - start_time
        Metrics.record_request_latency(duration)
        json!(conn, 200, response)

      {:error, msg} ->
        Metrics.record_error()
        json!(conn, 400, %{code: "bad_request", message: msg})
    end
  end

  defp do_stream_http_sse(conn, opts) do
    start_time = System.monotonic_time(:millisecond)

    case Honeycomb.chat_completion(opts) do
      {:error, msg} ->
        Metrics.record_error()
        json!(conn, 400, %{code: "bad_request", message: msg})

      stream ->
        conn
        |> put_resp_content_type("text/event-stream")
        |> put_resp_header("cache-control", "no-cache")
        |> put_resp_header("connection", "keep-alive")
        |> send_chunked(200)
        |> stream_chunks(stream, start_time)
    end
  end

  defp stream_chunks(conn, stream, start_time) do
    first_token_time = System.monotonic_time(:millisecond)
    ttft_recorded = false

    {conn, _ttft_recorded} =
      Enum.reduce_while(stream, {conn, ttft_recorded}, fn chunk, {conn, recorded} ->
        # Record TTFT on first chunk
        recorded =
          if not recorded do
            ttft = System.monotonic_time(:millisecond) - first_token_time
            Metrics.record_ttft(ttft)
            true
          else
            recorded
          end

        data = Jason.encode!(chunk)

        case chunk(conn, "data: #{data}\n\n") do
          {:ok, conn} ->
            {:cont, {conn, recorded}}

          _ ->
            {:halt, {conn, recorded}}
        end
      end)

    # Send done marker
    case chunk(conn, "data: [DONE]\n\n") do
      {:ok, conn} -> conn
      _ -> conn
    end

    duration = System.monotonic_time(:millisecond) - start_time
    Metrics.record_request_latency(duration)

    conn
  end

  defp json!(conn, status, data) do
    conn
    |> put_resp_content_type("application/json")
    |> resp(status, Jason.encode!(data))
    |> send_resp()
  end

  defp get_health_status do
    if Process.whereis(Engine) do
      Engine.health()
    else
      %{
        status: :healthy,
        checks: %{
          serving: Process.whereis(Honeycomb.Serving) != nil
        }
      }
    end
  end

  defp is_ready? do
    cond do
      Process.whereis(Engine) ->
        Engine.ready?()

      Process.whereis(Honeycomb.Serving) ->
        true

      true ->
        false
    end
  end

  defp get_metrics do
    if Process.whereis(Metrics) do
      Metrics.prometheus_format()
    else
      "# No metrics available\n"
    end
  end

  defp get_stats do
    if Process.whereis(Engine) do
      Engine.stats()
    else
      metrics = if Process.whereis(Metrics), do: Metrics.get_all(), else: %{}
      %{metrics: metrics}
    end
  end
end
