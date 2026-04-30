package apicompat

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ChatCompletionsToResponsesResponse converts a Chat Completions response into
// a Responses API response. This is used for OpenAI-compatible upstreams that
// only expose /v1/chat/completions but must serve /v1/responses clients.
func ChatCompletionsToResponsesResponse(resp *ChatCompletionsResponse) *ResponsesResponse {
	if resp == nil {
		return &ResponsesResponse{
			ID:     generateResponsesID(),
			Object: "response",
			Status: "completed",
			Output: []ResponsesOutput{{
				Type:    "message",
				Role:    "assistant",
				Status:  "completed",
				Content: []ResponsesContentPart{{Type: "output_text", Text: ""}},
			}},
		}
	}

	id := strings.TrimSpace(resp.ID)
	if id == "" {
		id = generateResponsesID()
	}

	out := &ResponsesResponse{
		ID:     id,
		Object: "response",
		Model:  resp.Model,
		Status: "completed",
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		out.Output = append(out.Output, chatMessageToResponsesOutputs(choice.Message)...)
		out.Status, out.IncompleteDetails = chatFinishReasonToResponsesStatus(choice.FinishReason)
	}

	if len(out.Output) == 0 {
		out.Output = []ResponsesOutput{{
			Type:    "message",
			Role:    "assistant",
			Status:  "completed",
			Content: []ResponsesContentPart{{Type: "output_text", Text: ""}},
		}}
	}

	if resp.Usage != nil {
		usage := &ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
		if resp.Usage.PromptTokensDetails != nil && resp.Usage.PromptTokensDetails.CachedTokens > 0 {
			usage.InputTokensDetails = &ResponsesInputTokensDetails{
				CachedTokens: resp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
		out.Usage = usage
	}

	return out
}

func chatMessageToResponsesOutputs(msg ChatMessage) []ResponsesOutput {
	var outputs []ResponsesOutput
	reasoningText := strings.TrimSpace(msg.ReasoningContent)

	if reasoningText != "" {
		outputs = append(outputs, ResponsesOutput{
			Type: "reasoning",
			ID:   generateItemID(),
			Summary: []ResponsesSummary{{
				Type: "summary_text",
				Text: reasoningText,
			}},
		})
	}

	contentText := extractChatResponseText(msg.Content)
	if contentText != "" {
		outputs = append(outputs, ResponsesOutput{
			Type:   "message",
			ID:     generateItemID(),
			Role:   "assistant",
			Status: "completed",
			Content: []ResponsesContentPart{{
				Type: "output_text",
				Text: contentText,
			}},
		})
	} else if reasoningText != "" {
		outputs = append(outputs, ResponsesOutput{
			Type:   "message",
			ID:     generateItemID(),
			Role:   "assistant",
			Status: "completed",
			Content: []ResponsesContentPart{{
				Type: "output_text",
				Text: reasoningText,
			}},
		})
	}

	for _, toolCall := range msg.ToolCalls {
		callID := strings.TrimSpace(toolCall.ID)
		if callID == "" {
			callID = fmt.Sprintf("call_%d", len(outputs))
		}
		outputs = append(outputs, ResponsesOutput{
			Type:      "function_call",
			ID:        generateItemID(),
			CallID:    callID,
			Name:      strings.TrimSpace(toolCall.Function.Name),
			Arguments: toolCall.Function.Arguments,
			Status:    "completed",
		})
	}

	return outputs
}

func extractChatResponseText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}

	content, err := parseChatMessageContent(raw)
	if err == nil {
		if content.Text != nil {
			return *content.Text
		}
		return flattenChatContentParts(content.Parts)
	}

	return ""
}

func chatFinishReasonToResponsesStatus(finishReason string) (string, *ResponsesIncompleteDetails) {
	switch strings.TrimSpace(finishReason) {
	case "length":
		return "incomplete", &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	case "content_filter":
		return "incomplete", &ResponsesIncompleteDetails{Reason: "content_filter"}
	default:
		return "completed", nil
	}
}

// ChatCompletionsToResponsesState tracks state while converting Chat
// Completions SSE chunks into Responses SSE events.
type ChatCompletionsToResponsesState struct {
	ResponseID     string
	Model          string
	Created        int64
	SequenceNumber int

	CreatedSent    bool
	InProgressSent bool
	CompletedSent  bool

	NextOutputIndex int

	Reasoning *chatOpenResponsesItem
	Message   *chatOpenResponsesItem
	Tools     map[int]*chatOpenResponsesTool

	PendingFinishReason string
	Usage               *ResponsesUsage
	SawToolCall         bool
}

type chatOpenResponsesItem struct {
	OutputIndex int
	ItemID      string
	Text        string
	Kind        string
	Added       bool
	Visible     bool
}

type chatOpenResponsesTool struct {
	OutputIndex int
	ItemID      string
	CallID      string
	Name        string
	Added       bool
	Arguments   string
}

// NewChatCompletionsToResponsesState returns an initialized stream converter state.
func NewChatCompletionsToResponsesState() *ChatCompletionsToResponsesState {
	return &ChatCompletionsToResponsesState{
		ResponseID: generateResponsesID(),
		Created:    time.Now().Unix(),
		Tools:      make(map[int]*chatOpenResponsesTool),
	}
}

// ChatCompletionsChunkToResponsesEvents converts a single Chat Completions
// chunk into zero or more Responses events.
func ChatCompletionsChunkToResponsesEvents(
	chunk *ChatCompletionsChunk,
	state *ChatCompletionsToResponsesState,
) []ResponsesStreamEvent {
	if chunk == nil || state == nil {
		return nil
	}

	if id := strings.TrimSpace(chunk.ID); id != "" {
		state.ResponseID = id
	}
	if state.Model == "" && strings.TrimSpace(chunk.Model) != "" {
		state.Model = strings.TrimSpace(chunk.Model)
	}
	if chunk.Usage != nil {
		state.Usage = chatUsageToResponsesUsage(chunk.Usage)
	}

	var events []ResponsesStreamEvent
	if !state.CreatedSent {
		events = append(events, state.makeCreatedEvent())
	}
	if !state.InProgressSent {
		events = append(events, state.makeInProgressEvent())
	}

	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && strings.TrimSpace(*choice.FinishReason) != "" {
			state.PendingFinishReason = strings.TrimSpace(*choice.FinishReason)
		}

		if delta := strings.TrimSpace(choice.Delta.Role); delta != "" && delta != "assistant" {
			// Responses output is always assistant-side; ignore other role hints.
		}

		if choice.Delta.ReasoningContent != nil {
			if delta := *choice.Delta.ReasoningContent; delta != "" {
				events = append(events, state.ensureReasoningOpen()...)
				state.Reasoning.Text += delta
				events = append(events, state.makeEvent("response.reasoning_summary_text.delta", &ResponsesStreamEvent{
					OutputIndex:  state.Reasoning.OutputIndex,
					SummaryIndex: 0,
					ItemID:       state.Reasoning.ItemID,
					Delta:        delta,
				}))
			}
		}

		if choice.Delta.Content != nil {
			events = append(events, state.ensureMessageOpen()...)
			state.Message.Text += *choice.Delta.Content
			if *choice.Delta.Content != "" {
				state.Message.Visible = true
			}
			events = append(events, state.makeEvent("response.output_text.delta", &ResponsesStreamEvent{
				OutputIndex:  state.Message.OutputIndex,
				ContentIndex: 0,
				ItemID:       state.Message.ItemID,
				Delta:        *choice.Delta.Content,
			}))
		}

		for toolDeltaIdx, toolCall := range choice.Delta.ToolCalls {
			openTool := state.ensureToolOpen(toolCall, toolDeltaIdx)
			if openTool == nil {
				continue
			}
			state.SawToolCall = true

			events = append(events, state.ensureToolAddedEvent(toolCall, toolDeltaIdx, openTool)...)
			if strings.TrimSpace(toolCall.Function.Arguments) != "" {
				openTool.Arguments += toolCall.Function.Arguments
				events = append(events, state.makeEvent("response.function_call_arguments.delta", &ResponsesStreamEvent{
					OutputIndex: openTool.OutputIndex,
					ItemID:      openTool.ItemID,
					CallID:      openTool.CallID,
					Name:        openTool.Name,
					Delta:       toolCall.Function.Arguments,
				}))
			}
			if name := strings.TrimSpace(toolCall.Function.Name); name != "" {
				openTool.Name = name
			}
			if callID := strings.TrimSpace(toolCall.ID); callID != "" {
				openTool.CallID = callID
			}
			state.Tools[state.chatToolIndex(toolCall, toolDeltaIdx)] = openTool
		}
	}

	return events
}

// FinalizeChatCompletionsResponsesStream emits trailing done/completed events
// after the upstream Chat Completions stream ends.
func FinalizeChatCompletionsResponsesStream(state *ChatCompletionsToResponsesState) []ResponsesStreamEvent {
	if state == nil || state.CompletedSent {
		return nil
	}
	state.applyVisibleTextFallbackFromReasoning()

	var events []ResponsesStreamEvent
	if !state.CreatedSent {
		events = append(events, state.makeCreatedEvent())
	}

	for _, item := range state.sortedOpenItems() {
		switch item.Kind {
		case "reasoning":
			events = append(events, state.makeEvent("response.reasoning_summary_text.done", &ResponsesStreamEvent{
				OutputIndex:  item.OutputIndex,
				SummaryIndex: 0,
				ItemID:       item.ItemID,
			}))
			events = append(events, state.makeEvent("response.output_item.done", &ResponsesStreamEvent{
				OutputIndex: item.OutputIndex,
				Item: &ResponsesOutput{
					Type:   "reasoning",
					ID:     item.ItemID,
					Status: "completed",
					Summary: []ResponsesSummary{{
						Type: "summary_text",
						Text: item.Text,
					}},
				},
			}))
		case "message":
			if !item.Added {
				events = append(events, state.makeMessageOpenEvents(&chatOpenResponsesItem{
					OutputIndex: item.OutputIndex,
					ItemID:      item.ItemID,
				})...)
			}
			if item.Text != "" && !item.Visible {
				events = append(events, state.makeEvent("response.output_text.delta", &ResponsesStreamEvent{
					OutputIndex:  item.OutputIndex,
					ContentIndex: 0,
					ItemID:       item.ItemID,
					Delta:        item.Text,
				}))
			}
			events = append(events, state.makeEvent("response.output_text.done", &ResponsesStreamEvent{
				OutputIndex:  item.OutputIndex,
				ContentIndex: 0,
				ItemID:       item.ItemID,
				Text:         item.Text,
			}))
			events = append(events, state.makeEvent("response.content_part.done", &ResponsesStreamEvent{
				OutputIndex:  item.OutputIndex,
				ContentIndex: 0,
				ItemID:       item.ItemID,
				Part: &ResponsesContentPart{
					Type: "output_text",
					Text: item.Text,
				},
			}))
			events = append(events, state.makeEvent("response.output_item.done", &ResponsesStreamEvent{
				OutputIndex: item.OutputIndex,
				Item: &ResponsesOutput{
					Type:   "message",
					ID:     item.ItemID,
					Role:   "assistant",
					Status: "completed",
					Content: []ResponsesContentPart{{
						Type: "output_text",
						Text: item.Text,
					}},
				},
			}))
		case "function_call":
			events = append(events, state.makeEvent("response.function_call_arguments.done", &ResponsesStreamEvent{
				OutputIndex: item.OutputIndex,
				ItemID:      item.ItemID,
				CallID:      item.CallID,
				Name:        item.Name,
			}))
			events = append(events, state.makeEvent("response.output_item.done", &ResponsesStreamEvent{
				OutputIndex: item.OutputIndex,
				Item: &ResponsesOutput{
					Type:      "function_call",
					ID:        item.ItemID,
					CallID:    item.CallID,
					Name:      item.Name,
					Arguments: item.Arguments,
					Status:    "completed",
				},
			}))
		}
	}

	status, incompleteDetails := chatFinishReasonToResponsesStatus(state.PendingFinishReason)
	events = append(events, state.makeCompletedEvent(status, incompleteDetails))
	state.CompletedSent = true
	state.Reasoning = nil
	state.Message = nil
	state.Tools = map[int]*chatOpenResponsesTool{}
	return events
}

type chatResponseStreamItem struct {
	Kind        string
	OutputIndex int
	ItemID      string
	CallID      string
	Name        string
	Text        string
	Arguments   string
	Added       bool
	Visible     bool
}

func (s *ChatCompletionsToResponsesState) sortedOpenItems() []chatResponseStreamItem {
	items := make([]chatResponseStreamItem, 0, len(s.Tools)+2)
	if s.Reasoning != nil {
		items = append(items, chatResponseStreamItem{
			Kind:        "reasoning",
			OutputIndex: s.Reasoning.OutputIndex,
			ItemID:      s.Reasoning.ItemID,
			Text:        s.Reasoning.Text,
			Added:       s.Reasoning.Added,
			Visible:     s.Reasoning.Visible,
		})
	}
	if s.Message != nil {
		items = append(items, chatResponseStreamItem{
			Kind:        "message",
			OutputIndex: s.Message.OutputIndex,
			ItemID:      s.Message.ItemID,
			Text:        s.Message.Text,
			Added:       s.Message.Added,
			Visible:     s.Message.Visible,
		})
	}
	for _, tool := range s.Tools {
		if tool == nil {
			continue
		}
		items = append(items, chatResponseStreamItem{
			Kind:        "function_call",
			OutputIndex: tool.OutputIndex,
			ItemID:      tool.ItemID,
			CallID:      tool.CallID,
			Name:        tool.Name,
			Arguments:   tool.Arguments,
		})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].OutputIndex < items[j].OutputIndex
	})
	return items
}

func (s *ChatCompletionsToResponsesState) ensureReasoningOpen() []ResponsesStreamEvent {
	if s.Reasoning != nil {
		return nil
	}
	s.Reasoning = &chatOpenResponsesItem{
		OutputIndex: s.NextOutputIndex,
		ItemID:      generateItemID(),
		Kind:        "reasoning",
		Added:       true,
	}
	s.NextOutputIndex++
	return []ResponsesStreamEvent{s.makeEvent("response.output_item.added", &ResponsesStreamEvent{
		OutputIndex: s.Reasoning.OutputIndex,
		Item: &ResponsesOutput{
			Type:   "reasoning",
			ID:     s.Reasoning.ItemID,
			Status: "in_progress",
		},
	})}
}

func (s *ChatCompletionsToResponsesState) ensureMessageOpen() []ResponsesStreamEvent {
	if s.Message != nil {
		return nil
	}
	s.Message = &chatOpenResponsesItem{
		OutputIndex: s.NextOutputIndex,
		ItemID:      generateItemID(),
		Kind:        "message",
		Added:       true,
	}
	s.NextOutputIndex++
	return s.makeMessageOpenEvents(s.Message)
}

func (s *ChatCompletionsToResponsesState) makeMessageOpenEvents(item *chatOpenResponsesItem) []ResponsesStreamEvent {
	if item == nil {
		return nil
	}
	return []ResponsesStreamEvent{
		s.makeEvent("response.output_item.added", &ResponsesStreamEvent{
			OutputIndex: item.OutputIndex,
			Item: &ResponsesOutput{
				Type:   "message",
				ID:     item.ItemID,
				Role:   "assistant",
				Status: "in_progress",
				Content: []ResponsesContentPart{{
					Type: "output_text",
					Text: "",
				}},
			},
		}),
		s.makeEvent("response.content_part.added", &ResponsesStreamEvent{
			OutputIndex:  item.OutputIndex,
			ContentIndex: 0,
			ItemID:       item.ItemID,
			Part: &ResponsesContentPart{
				Type: "output_text",
				Text: "",
			},
		}),
	}
}

func (s *ChatCompletionsToResponsesState) ensureToolOpen(
	toolCall ChatToolCall,
	fallbackIdx int,
) *chatOpenResponsesTool {
	toolIdx := s.chatToolIndex(toolCall, fallbackIdx)
	if existing, ok := s.Tools[toolIdx]; ok && existing != nil {
		if existing.CallID == "" {
			existing.CallID = firstNonEmptyChatString(toolCall.ID, fmt.Sprintf("call_%d", toolIdx))
		}
		if existing.Name == "" {
			existing.Name = firstNonEmptyChatString(toolCall.Function.Name, "tool")
		}
		return existing
	}

	callID := firstNonEmptyChatString(toolCall.ID, fmt.Sprintf("call_%d", toolIdx))
	name := firstNonEmptyChatString(toolCall.Function.Name, "tool")

	openTool := &chatOpenResponsesTool{
		OutputIndex: s.NextOutputIndex,
		ItemID:      generateItemID(),
		CallID:      callID,
		Name:        name,
	}
	s.NextOutputIndex++
	s.Tools[toolIdx] = openTool
	return openTool
}

func (s *ChatCompletionsToResponsesState) ensureToolAddedEvent(
	toolCall ChatToolCall,
	fallbackIdx int,
	openTool *chatOpenResponsesTool,
) []ResponsesStreamEvent {
	if openTool == nil {
		return nil
	}
	toolIdx := s.chatToolIndex(toolCall, fallbackIdx)
	if toolIdx < 0 {
		return nil
	}
	if openTool.CallID == "" {
		openTool.CallID = fmt.Sprintf("call_%d", toolIdx)
	}
	if openTool.Name == "" {
		openTool.Name = "tool"
	}

	if s.Tools[toolIdx] == nil || openTool.Added {
		return nil
	}
	if openTool.ItemID == "" {
		return nil
	}
	openTool.Added = true
	return []ResponsesStreamEvent{s.makeEvent("response.output_item.added", &ResponsesStreamEvent{
		OutputIndex: openTool.OutputIndex,
		Item: &ResponsesOutput{
			Type:   "function_call",
			ID:     openTool.ItemID,
			CallID: openTool.CallID,
			Name:   openTool.Name,
			Status: "in_progress",
		},
	})}
}

func (s *ChatCompletionsToResponsesState) makeCreatedEvent() ResponsesStreamEvent {
	s.CreatedSent = true
	return s.makeEvent("response.created", &ResponsesStreamEvent{
		Response: &ResponsesResponse{
			ID:     s.ResponseID,
			Object: "response",
			Model:  s.Model,
			Status: "in_progress",
			Output: []ResponsesOutput{},
		},
	})
}

func (s *ChatCompletionsToResponsesState) makeInProgressEvent() ResponsesStreamEvent {
	s.InProgressSent = true
	return s.makeEvent("response.in_progress", &ResponsesStreamEvent{
		Response: &ResponsesResponse{
			ID:     s.ResponseID,
			Object: "response",
			Model:  s.Model,
			Status: "in_progress",
			Output: []ResponsesOutput{},
		},
	})
}

func (s *ChatCompletionsToResponsesState) makeCompletedEvent(
	status string,
	incompleteDetails *ResponsesIncompleteDetails,
) ResponsesStreamEvent {
	return s.makeEvent("response.completed", &ResponsesStreamEvent{
		Response: &ResponsesResponse{
			ID:                s.ResponseID,
			Object:            "response",
			Model:             s.Model,
			Status:            status,
			Output:            s.buildCompletedOutputs(),
			Usage:             s.Usage,
			IncompleteDetails: incompleteDetails,
		},
	})
}

func (s *ChatCompletionsToResponsesState) buildCompletedOutputs() []ResponsesOutput {
	items := s.sortedOpenItems()
	if len(items) == 0 {
		return []ResponsesOutput{}
	}

	outputs := make([]ResponsesOutput, 0, len(items))
	for _, item := range items {
		switch item.Kind {
		case "reasoning":
			outputs = append(outputs, ResponsesOutput{
				Type: "reasoning",
				ID:   item.ItemID,
				Summary: []ResponsesSummary{{
					Type: "summary_text",
					Text: item.Text,
				}},
			})
		case "message":
			outputs = append(outputs, ResponsesOutput{
				Type:   "message",
				ID:     item.ItemID,
				Role:   "assistant",
				Status: "completed",
				Content: []ResponsesContentPart{{
					Type: "output_text",
					Text: item.Text,
				}},
			})
		case "function_call":
			outputs = append(outputs, ResponsesOutput{
				Type:      "function_call",
				ID:        item.ItemID,
				CallID:    item.CallID,
				Name:      item.Name,
				Arguments: item.Arguments,
				Status:    "completed",
			})
		}
	}
	return outputs
}

func (s *ChatCompletionsToResponsesState) applyVisibleTextFallbackFromReasoning() {
	if s == nil || s.Reasoning == nil || strings.TrimSpace(s.Reasoning.Text) == "" {
		return
	}
	if s.Message == nil {
		s.Message = &chatOpenResponsesItem{
			OutputIndex: s.NextOutputIndex,
			ItemID:      generateItemID(),
			Text:        s.Reasoning.Text,
			Kind:        "message",
			Added:       false,
			Visible:     false,
		}
		s.NextOutputIndex++
		return
	}
	if strings.TrimSpace(s.Message.Text) == "" {
		s.Message.Text = s.Reasoning.Text
	}
}

func (s *ChatCompletionsToResponsesState) makeEvent(
	eventType string,
	template *ResponsesStreamEvent,
) ResponsesStreamEvent {
	evt := ResponsesStreamEvent{Type: eventType, SequenceNumber: s.SequenceNumber}
	s.SequenceNumber++
	if template == nil {
		return evt
	}
	evt.Response = template.Response
	evt.Item = template.Item
	evt.Part = template.Part
	evt.OutputIndex = template.OutputIndex
	evt.ContentIndex = template.ContentIndex
	evt.Delta = template.Delta
	evt.Text = template.Text
	evt.ItemID = template.ItemID
	evt.CallID = template.CallID
	evt.Name = template.Name
	evt.Arguments = template.Arguments
	evt.SummaryIndex = template.SummaryIndex
	evt.Code = template.Code
	evt.Param = template.Param
	return evt
}

func (s *ChatCompletionsToResponsesState) chatToolIndex(toolCall ChatToolCall, fallback int) int {
	if toolCall.Index != nil && *toolCall.Index >= 0 {
		return *toolCall.Index
	}
	if fallback >= 0 {
		return fallback
	}
	return 0
}

func chatUsageToResponsesUsage(usage *ChatUsage) *ResponsesUsage {
	if usage == nil {
		return nil
	}
	out := &ResponsesUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
		TotalTokens:  usage.TotalTokens,
	}
	if usage.PromptTokensDetails != nil && usage.PromptTokensDetails.CachedTokens > 0 {
		out.InputTokensDetails = &ResponsesInputTokensDetails{
			CachedTokens: usage.PromptTokensDetails.CachedTokens,
		}
	}
	return out
}

func firstNonEmptyChatString(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
