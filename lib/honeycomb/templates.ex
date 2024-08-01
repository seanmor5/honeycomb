defmodule Honeycomb.Templates do
  def apply_chat_template(template, messages) do
    path = Path.join([Path.dirname(__ENV__.file), "templates", "#{template}.eex"])
    EEx.eval_file(path, messages: messages)
  end
end
