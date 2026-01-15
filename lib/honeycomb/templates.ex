defmodule Honeycomb.Templates do
  @moduledoc """
  Chat template management with compile-time caching.

  Templates are compiled at startup for maximum performance during inference.
  Supports multiple template formats for different model families.
  """

  use GenServer
  require Logger

  @templates_dir Path.join(Path.dirname(__ENV__.file), "templates")

  # Client API

  @doc """
  Starts the template cache.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Applies a chat template to messages.

  Uses cached compiled templates for fast rendering.
  """
  def apply_chat_template(template, messages) do
    case Process.whereis(__MODULE__) do
      nil ->
        # Fallback to direct evaluation if GenServer not started
        apply_chat_template_direct(template, messages)

      _pid ->
        case get_compiled(template) do
          {:ok, compiled} ->
            apply_compiled(compiled, messages)

          :error ->
            # Template not found in cache, try direct
            apply_chat_template_direct(template, messages)
        end
    end
  end

  @doc """
  Gets a compiled template by name.
  """
  def get_compiled(template) do
    GenServer.call(__MODULE__, {:get_compiled, template})
  end

  @doc """
  Registers a custom template.
  """
  def register_template(name, template_string) do
    GenServer.call(__MODULE__, {:register, name, template_string})
  end

  @doc """
  Lists available templates.
  """
  def list_templates do
    GenServer.call(__MODULE__, :list)
  end

  @doc """
  Reloads all templates from disk.
  """
  def reload do
    GenServer.call(__MODULE__, :reload)
  end

  @doc """
  Direct template application without cache (fallback).
  """
  def apply_chat_template_direct(template, messages) do
    path = template_path(template)

    if File.exists?(path) do
      EEx.eval_file(path, messages: messages)
    else
      raise "Template not found: #{template}"
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    templates = load_all_templates()
    Logger.info("Templates: Loaded #{map_size(templates)} templates")

    {:ok, %{templates: templates}}
  end

  @impl true
  def handle_call({:get_compiled, template}, _from, state) do
    result = Map.fetch(state.templates, to_string(template))
    {:reply, result, state}
  end

  @impl true
  def handle_call({:register, name, template_string}, _from, state) do
    compiled = EEx.compile_string(template_string)
    templates = Map.put(state.templates, to_string(name), compiled)
    {:reply, :ok, %{state | templates: templates}}
  end

  @impl true
  def handle_call(:list, _from, state) do
    {:reply, Map.keys(state.templates), state}
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    templates = load_all_templates()
    {:reply, :ok, %{templates: templates}}
  end

  # Private helpers

  defp load_all_templates do
    if File.dir?(@templates_dir) do
      @templates_dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".eex"))
      |> Enum.map(fn file ->
        name = String.trim_trailing(file, ".eex")
        path = Path.join(@templates_dir, file)

        case compile_template(path) do
          {:ok, compiled} -> {name, compiled}
          {:error, _} -> nil
        end
      end)
      |> Enum.reject(&is_nil/1)
      |> Map.new()
    else
      %{}
    end
  end

  defp compile_template(path) do
    try do
      compiled = EEx.compile_file(path)
      {:ok, compiled}
    rescue
      e ->
        Logger.warning("Failed to compile template #{path}: #{inspect(e)}")
        {:error, e}
    end
  end

  defp template_path(template) do
    Path.join(@templates_dir, "#{template}.eex")
  end

  defp apply_compiled(compiled, messages) do
    try do
      {result, _} = Code.eval_quoted(compiled, messages: messages)
      result
    rescue
      e ->
        Logger.error("Template rendering failed: #{inspect(e)}")
        raise e
    end
  end
end
