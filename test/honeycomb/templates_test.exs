defmodule Honeycomb.TemplatesTest do
  use ExUnit.Case

  setup do
    messages = [
      %{role: "user", content: "Hello!"}
    ]

    [messages: messages]
  end

  test "phi3", %{messages: messages} do
    result = Honeycomb.Templates.apply_chat_template("phi3", messages)

    assert result == """
           <|user|>
           Hello!<|end|>
           <|assistant|>
           """
  end
end
