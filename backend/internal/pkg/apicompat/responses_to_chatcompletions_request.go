package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesToChatCompletionsRequest converts a Responses API request into a
// Chat Completions request. This is used for OpenAI-compatible upstreams that
// only expose /v1/chat/completions but do not implement /v1/responses.
func ResponsesToChatCompletionsRequest(req *ResponsesRequest) (*ChatCompletionsRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("responses request is nil")
	}

	messages, err := convertResponsesInputToChatMessages(req.Instructions, req.Input)
	if err != nil {
		return nil, err
	}

	out := &ChatCompletionsRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		ServiceTier: req.ServiceTier,
	}
	if req.Stream {
		out.StreamOptions = &ChatStreamOptions{IncludeUsage: true}
	}

	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		v := *req.MaxOutputTokens
		out.MaxTokens = &v
	}

	if req.Reasoning != nil && strings.TrimSpace(req.Reasoning.Effort) != "" {
		out.ReasoningEffort = strings.TrimSpace(req.Reasoning.Effort)
	}

	if len(req.Tools) > 0 {
		out.Tools = convertResponsesToolsToChat(req.Tools)
	}

	if len(req.ToolChoice) > 0 {
		toolChoice, err := convertResponsesToolChoiceToChat(req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("convert tool_choice: %w", err)
		}
		out.ToolChoice = toolChoice
	}

	return out, nil
}

func convertResponsesInputToChatMessages(instructions string, inputRaw json.RawMessage) ([]ChatMessage, error) {
	messages := make([]ChatMessage, 0, 8)

	if trimmed := strings.TrimSpace(instructions); trimmed != "" {
		content, _ := json.Marshal(trimmed)
		messages = append(messages, ChatMessage{
			Role:    "system",
			Content: content,
		})
	}

	if len(inputRaw) == 0 || string(inputRaw) == "null" {
		return messages, nil
	}

	var inputStr string
	if err := json.Unmarshal(inputRaw, &inputStr); err == nil {
		content, _ := json.Marshal(inputStr)
		messages = append(messages, ChatMessage{
			Role:    "user",
			Content: content,
		})
		return messages, nil
	}

	var items []map[string]any
	if err := json.Unmarshal(inputRaw, &items); err != nil {
		return nil, fmt.Errorf("parse responses input: %w", err)
	}

	pendingUserParts := make([]ChatContentPart, 0, 4)
	flushPendingUserParts := func() error {
		if len(pendingUserParts) == 0 {
			return nil
		}
		content, err := json.Marshal(pendingUserParts)
		if err != nil {
			return fmt.Errorf("marshal pending user parts: %w", err)
		}
		messages = append(messages, ChatMessage{
			Role:    "user",
			Content: content,
		})
		pendingUserParts = pendingUserParts[:0]
		return nil
	}

	for _, item := range items {
		if len(item) == 0 {
			continue
		}

		role := responsesInputString(item["role"])
		typ := responsesInputString(item["type"])

		switch {
		case role == "system" || role == "developer":
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			content, err := convertResponsesMessageContentToChat(item["content"], true)
			if err != nil {
				return nil, err
			}
			messages = append(messages, ChatMessage{
				Role:    "system",
				Content: content,
			})

		case role == "user":
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			content, err := convertResponsesMessageContentToChat(item["content"], false)
			if err != nil {
				return nil, err
			}
			messages = append(messages, ChatMessage{
				Role:    "user",
				Content: content,
			})

		case role == "assistant":
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			content, reasoning, err := convertResponsesAssistantContentToChat(item["content"])
			if err != nil {
				return nil, err
			}
			msg := ChatMessage{Role: "assistant"}
			if len(content) > 0 {
				msg.Content = content
			}
			if reasoning != "" {
				msg.ReasoningContent = reasoning
			}
			messages = append(messages, msg)

		case isResponsesToolCallInputType(typ):
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			msg := responsesToolCallItemToChatMessage(item)
			messages = append(messages, msg)

		case isResponsesToolOutputInputType(typ):
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			msg := responsesToolOutputItemToChatMessage(item)
			messages = append(messages, msg)

		case isResponsesTopLevelContentPartType(typ):
			part := responsesTopLevelContentPartToChat(item)
			if part.Type != "" {
				pendingUserParts = append(pendingUserParts, part)
			}

		case typ == "item_reference" || typ == "reasoning":
			continue

		default:
			if err := flushPendingUserParts(); err != nil {
				return nil, err
			}
			if item["content"] == nil {
				continue
			}
			content, err := convertResponsesMessageContentToChat(item["content"], false)
			if err != nil {
				return nil, err
			}
			messages = append(messages, ChatMessage{
				Role:    "user",
				Content: content,
			})
		}
	}

	if err := flushPendingUserParts(); err != nil {
		return nil, err
	}

	return messages, nil
}

func convertResponsesMessageContentToChat(raw any, textOnly bool) (json.RawMessage, error) {
	if raw == nil {
		return json.Marshal("")
	}

	if s, ok := raw.(string); ok {
		return json.Marshal(s)
	}

	contentBytes, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("marshal responses content: %w", err)
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(contentBytes, &parts); err != nil {
		return contentBytes, nil
	}

	if textOnly {
		var textParts []string
		for _, part := range parts {
			switch part.Type {
			case "input_text", "output_text", "text":
				if part.Text != "" {
					textParts = append(textParts, part.Text)
				}
			}
		}
		return json.Marshal(strings.Join(textParts, "\n\n"))
	}

	chatParts := make([]ChatContentPart, 0, len(parts))
	var textParts []string
	hasImage := false
	for _, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "text":
			if part.Text == "" {
				continue
			}
			textParts = append(textParts, part.Text)
			chatParts = append(chatParts, ChatContentPart{
				Type: "text",
				Text: part.Text,
			})
		case "input_image":
			if part.ImageURL == "" {
				continue
			}
			hasImage = true
			chatParts = append(chatParts, ChatContentPart{
				Type: "image_url",
				ImageURL: &ChatImageURL{
					URL: part.ImageURL,
				},
			})
		}
	}

	if !hasImage {
		return json.Marshal(strings.Join(textParts, "\n\n"))
	}
	if len(chatParts) == 0 {
		return json.Marshal("")
	}
	return json.Marshal(chatParts)
}

func convertResponsesAssistantContentToChat(raw any) (json.RawMessage, string, error) {
	if raw == nil {
		return nil, "", nil
	}

	if s, ok := raw.(string); ok {
		if strings.TrimSpace(s) == "" {
			return nil, "", nil
		}
		content, _ := json.Marshal(s)
		return content, "", nil
	}

	contentBytes, err := json.Marshal(raw)
	if err != nil {
		return nil, "", fmt.Errorf("marshal assistant content: %w", err)
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(contentBytes, &parts); err != nil {
		return contentBytes, "", nil
	}

	var textParts []string
	for _, part := range parts {
		switch part.Type {
		case "output_text", "input_text", "text":
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}
	}

	if len(textParts) == 0 {
		return nil, "", nil
	}

	content, _ := json.Marshal(strings.Join(textParts, "\n\n"))
	return content, "", nil
}

func convertResponsesToolsToChat(tools []ResponsesTool) []ChatTool {
	out := make([]ChatTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			continue
		}
		fn := &ChatFunction{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
			Strict:      tool.Strict,
		}
		out = append(out, ChatTool{
			Type:     "function",
			Function: fn,
		})
	}
	return out
}

func convertResponsesToolChoiceToChat(raw json.RawMessage) (json.RawMessage, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto", "required", "none":
			return json.Marshal(s)
		default:
			return raw, nil
		}
	}

	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, err
	}

	if responsesInputString(obj["type"]) == "function" {
		if responsesInputString(obj["name"]) != "" {
			return raw, nil
		}
		if fn, ok := obj["function"].(map[string]any); ok {
			if name := responsesInputString(fn["name"]); name != "" {
				obj["name"] = name
				delete(obj, "function")
				return json.Marshal(obj)
			}
		}
	}

	return raw, nil
}

func responsesToolCallItemToChatMessage(item map[string]any) ChatMessage {
	callID := firstNonEmptyResponsesString(item["call_id"], item["id"])
	name := firstNonEmptyResponsesString(item["name"], item["tool_name"])
	if name == "" {
		if fn, ok := item["function"].(map[string]any); ok {
			name = responsesInputString(fn["name"])
		}
	}
	if name == "" {
		name = "tool"
	}

	arguments := responsesInputString(item["arguments"])
	if arguments == "" {
		arguments = responsesInputArguments(item["input"])
	}
	if arguments == "" {
		arguments = "{}"
	}
	if callID == "" {
		callID = "call_" + name
	}

	return ChatMessage{
		Role: "assistant",
		ToolCalls: []ChatToolCall{{
			ID:   callID,
			Type: "function",
			Function: ChatFunctionCall{
				Name:      name,
				Arguments: arguments,
			},
		}},
	}
}

func responsesToolOutputItemToChatMessage(item map[string]any) ChatMessage {
	callID := firstNonEmptyResponsesString(item["call_id"], item["id"])
	output := responsesInputString(item["output"])
	if output == "" {
		output = responsesInputArguments(item["output"])
	}
	if output == "" {
		output = "(empty)"
	}
	content, _ := json.Marshal(output)
	return ChatMessage{
		Role:       "tool",
		ToolCallID: callID,
		Content:    content,
	}
}

func isResponsesToolCallInputType(typ string) bool {
	switch strings.TrimSpace(typ) {
	case "function_call", "tool_call", "local_shell_call", "tool_search_call", "custom_tool_call", "mcp_tool_call":
		return true
	default:
		return false
	}
}

func isResponsesToolOutputInputType(typ string) bool {
	switch strings.TrimSpace(typ) {
	case "function_call_output", "tool_search_output", "custom_tool_call_output", "mcp_tool_call_output":
		return true
	default:
		return false
	}
}

func isResponsesTopLevelContentPartType(typ string) bool {
	switch strings.TrimSpace(typ) {
	case "input_text", "text", "input_image":
		return true
	default:
		return false
	}
}

func responsesTopLevelContentPartToChat(item map[string]any) ChatContentPart {
	typ := responsesInputString(item["type"])
	switch typ {
	case "input_text", "text":
		text := responsesInputString(item["text"])
		if text == "" {
			return ChatContentPart{}
		}
		return ChatContentPart{
			Type: "text",
			Text: text,
		}
	case "input_image":
		imageURL := responsesInputString(item["image_url"])
		if imageURL == "" {
			return ChatContentPart{}
		}
		return ChatContentPart{
			Type: "image_url",
			ImageURL: &ChatImageURL{
				URL: imageURL,
			},
		}
	default:
		return ChatContentPart{}
	}
}

func responsesInputArguments(raw any) string {
	switch v := raw.(type) {
	case nil:
		return ""
	case string:
		trimmed := strings.TrimSpace(v)
		if trimmed == "" {
			return ""
		}
		if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
			return trimmed
		}
		b, _ := json.Marshal(map[string]string{"input": v})
		return string(b)
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func firstNonEmptyResponsesString(values ...any) string {
	for _, value := range values {
		if s := responsesInputString(value); s != "" {
			return s
		}
	}
	return ""
}

func responsesInputString(v any) string {
	switch value := v.(type) {
	case string:
		return strings.TrimSpace(value)
	default:
		return ""
	}
}
