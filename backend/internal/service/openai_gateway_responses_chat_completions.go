package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

const openAIResponsesCompatLogPreviewMaxBytes = 600

func isOpenAICompatMiniMaxProvider(account *Account) bool {
	if account == nil {
		return false
	}
	baseURL := strings.ToLower(strings.TrimSpace(account.GetOpenAIBaseURL()))
	apiKey := strings.ToLower(strings.TrimSpace(account.GetOpenAIApiKey()))
	return strings.Contains(baseURL, "minimax") ||
		strings.Contains(baseURL, "abab") ||
		strings.HasPrefix(apiKey, "sk-minimax")
}

func estimateOpenAICompatPromptTokensFromChatBody(body []byte) int {
	total := 0
	for _, msg := range gjson.GetBytes(body, "messages").Array() {
		total += 4
		content := msg.Get("content")
		if content.Type == gjson.String {
			total += estimateOpenAICompatTextTokens(content.String())
			continue
		}
		if content.IsArray() {
			for _, part := range content.Array() {
				total += estimateOpenAICompatTextTokens(part.Get("text").String())
			}
		}
	}
	if total > 0 {
		total += 2
	}
	return total
}

func extractChatCompletionResponseText(resp *apicompat.ChatCompletionsResponse) string {
	if resp == nil {
		return ""
	}
	var builder strings.Builder
	for _, choice := range resp.Choices {
		if len(choice.Message.Content) > 0 {
			var text string
			if err := json.Unmarshal(choice.Message.Content, &text); err == nil {
				builder.WriteString(text)
			}
		}
		builder.WriteString(choice.Message.ReasoningContent)
	}
	return builder.String()
}

func estimateOpenAICompatTextTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	cjk := 0
	other := 0
	for _, r := range text {
		if unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r) {
			cjk++
		} else if !unicode.IsSpace(r) {
			other++
		}
	}
	tokens := cjk + (other+3)/4
	if tokens <= 0 {
		return 1
	}
	return tokens
}

func (s *OpenAIGatewayService) forwardOpenAIResponsesAsChatCompletions(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	startTime time.Time,
) (*OpenAIForwardResult, error) {
	var responsesReq apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		writeResponsesError(c, http.StatusBadRequest, "invalid_request_error", "Failed to parse responses request")
		return nil, fmt.Errorf("parse responses request: %w", err)
	}

	originalModel := strings.TrimSpace(responsesReq.Model)
	clientStream := responsesReq.Stream
	billingModel := resolveOpenAIForwardModel(account, originalModel, "")
	upstreamModel := billingModel
	if isOpenAIResponsesCompactPath(c) {
		if compactModel := resolveOpenAICompactForwardModel(account, billingModel); compactModel != "" {
			upstreamModel = compactModel
		}
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=request_in account_id=%d original_model=%s billing_model=%s upstream_model=%s stream=%v input_first_type=%s input_first_role=%s body=%s",
		account.ID,
		originalModel,
		billingModel,
		upstreamModel,
		clientStream,
		openAIResponsesCompatExtractPath(body, "input.0.type", "input.type"),
		openAIResponsesCompatExtractPath(body, "input.0.role"),
		truncateForLog(body, openAIResponsesCompatLogPreviewMaxBytes),
	)

	if responsesReq.Reasoning != nil {
		if normalized := normalizeOpenAIReasoningEffort(strings.TrimSpace(responsesReq.Reasoning.Effort)); normalized != "" {
			responsesReq.Reasoning.Effort = normalized
		}
	}

	chatReq, err := apicompat.ResponsesToChatCompletionsRequest(&responsesReq)
	if err != nil {
		writeResponsesError(c, http.StatusBadRequest, "invalid_request_error", "Failed to convert responses request")
		return nil, fmt.Errorf("convert responses to chat completions: %w", err)
	}
	chatReq.Model = upstreamModel
	chatReq.Stream = clientStream
	if normalized := normalizeOpenAIReasoningEffort(strings.TrimSpace(chatReq.ReasoningEffort)); normalized != "" {
		chatReq.ReasoningEffort = normalized
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=converted_chat_request account_id=%d upstream_model=%s messages=%d stream=%v first_role=%s first_content_type=%s",
		account.ID,
		chatReq.Model,
		len(chatReq.Messages),
		chatReq.Stream,
		openAIResponsesCompatFirstChatRole(chatReq.Messages),
		openAIResponsesCompatFirstChatContentType(chatReq.Messages),
	)

	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		writeResponsesError(c, http.StatusInternalServerError, "server_error", "Failed to encode upstream request")
		return nil, fmt.Errorf("marshal chat completions request: %w", err)
	}
	if account != nil && account.IsOpenAIApiKey() {
		if minimizedBody, minimized := minimizeOpenAICompatChatRequestForProvider(account, chatBody); minimized {
			chatBody = minimizedBody
		}
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=chat_request_body account_id=%d upstream_model=%s body=%s",
		account.ID,
		chatReq.Model,
		truncateForLog(chatBody, openAIResponsesCompatLogPreviewMaxBytes),
	)

	updatedBody, policyErr := s.applyOpenAIFastPolicyToBody(ctx, account, upstreamModel, chatBody)
	if policyErr != nil {
		var blocked *OpenAIFastBlockedError
		if errors.As(policyErr, &blocked) {
			writeResponsesError(c, http.StatusForbidden, "permission_error", blocked.Message)
		}
		return nil, policyErr
	}
	chatBody = updatedBody
	estimatedPromptTokens := 0
	if isOpenAICompatMiniMaxProvider(account) {
		estimatedPromptTokens = estimateOpenAICompatPromptTokensFromChatBody(chatBody)
		logger.LegacyPrintf("service.openai_gateway", "[Compat billing debug] stage=minimax_prompt_estimate input_tokens=%d", estimatedPromptTokens)
	}

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to get upstream token")
		return nil, fmt.Errorf("get access token: %w", err)
	}

	baseURL := account.GetOpenAIBaseURL()
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		writeResponsesError(c, http.StatusBadRequest, "invalid_request_error", "Invalid upstream base_url")
		return nil, fmt.Errorf("validate base URL: %w", err)
	}
	targetURL := buildOpenAIChatCompletionsURL(validatedURL)

	upstreamReq, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(chatBody))
	if err != nil {
		writeResponsesError(c, http.StatusInternalServerError, "server_error", "Failed to create upstream request")
		return nil, fmt.Errorf("create request: %w", err)
	}
	upstreamReq.Header.Set("Content-Type", "application/json")
	upstreamReq.Header.Set("Authorization", "Bearer "+token)

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Upstream request failed")
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=upstream_response account_id=%d status=%d content_type=%s request_id=%s",
		account.ID,
		resp.StatusCode,
		resp.Header.Get("Content-Type"),
		resp.Header.Get("x-request-id"),
	)

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(respBody))

		upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
		upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && (isPoolModeRetryableStatus(resp.StatusCode) || isOpenAITransientProcessingError(resp.StatusCode, upstreamMsg, respBody)),
			}
		}

		writeResponsesError(c, mapUpstreamStatusCode(resp.StatusCode), "server_error", upstreamMsg)
		return nil, fmt.Errorf("upstream error %d: %s", resp.StatusCode, upstreamMsg)
	}

	reasoningEffort := (*string)(nil)
	if trimmed := strings.TrimSpace(chatReq.ReasoningEffort); trimmed != "" {
		reasoningEffort = &trimmed
	}
	serviceTier := (*string)(nil)
	if trimmed := strings.TrimSpace(chatReq.ServiceTier); trimmed != "" {
		serviceTier = &trimmed
	}

	if clientStream {
		return s.handleResponsesViaChatCompletionsStream(resp, c, account, originalModel, billingModel, upstreamModel, serviceTier, reasoningEffort, startTime, estimatedPromptTokens)
	}
	return s.handleResponsesViaChatCompletionsNonStream(resp, c, account, originalModel, billingModel, upstreamModel, serviceTier, reasoningEffort, startTime, estimatedPromptTokens)
}

func minimizeOpenAICompatChatRequestForProvider(account *Account, body []byte) ([]byte, bool) {
	if account == nil || len(body) == 0 {
		return body, false
	}
	if !isOpenAICompatMiniMaxProvider(account) {
		return body, false
	}

	var reqBody map[string]any
	if err := json.Unmarshal(body, &reqBody); err != nil {
		return body, false
	}

	changed := false
	if model, ok := reqBody["model"].(string); ok {
		switch strings.TrimSpace(strings.ToLower(model)) {
		case "minimax-m2.7":
			reqBody["model"] = "codex-MiniMax-M2.7"
			changed = true
		case "minimax-m2.7-highspeed":
			reqBody["model"] = "codex-MiniMax-M2.7-highspeed"
			changed = true
		}
	}
	dropKeys := []string{
		"stream_options",
		"reasoning_effort",
		"service_tier",
		"temperature",
		"top_p",
		"tool_choice",
		"tools",
		"stop",
		"instructions",
	}
	for _, key := range dropKeys {
		if _, ok := reqBody[key]; ok {
			delete(reqBody, key)
			changed = true
		}
	}

	if messages, ok := reqBody["messages"].([]any); ok && len(messages) > 0 {
		systemTexts := make([]string, 0, 2)
		kept := make([]any, 0, len(messages))
		for _, raw := range messages {
			msg, ok := raw.(map[string]any)
			if !ok {
				kept = append(kept, raw)
				continue
			}
			role, _ := msg["role"].(string)
			role = strings.TrimSpace(strings.ToLower(role))
			if role == "system" || role == "developer" {
				if content, ok := msg["content"].(string); ok && strings.TrimSpace(content) != "" {
					systemTexts = append(systemTexts, strings.TrimSpace(content))
					changed = true
					continue
				}
			}
			kept = append(kept, msg)
		}
		if len(systemTexts) > 0 {
			prefix := strings.Join(systemTexts, "\n\n")
			if len(kept) == 0 {
				kept = append(kept, map[string]any{
					"role":    "user",
					"content": prefix,
				})
			} else if first, ok := kept[0].(map[string]any); ok {
				if role, _ := first["role"].(string); strings.EqualFold(strings.TrimSpace(role), "user") {
					if content, ok := first["content"].(string); ok {
						first["content"] = prefix + "\n\n" + content
						kept[0] = first
					}
				} else {
					kept = append([]any{map[string]any{
						"role":    "user",
						"content": prefix,
					}}, kept...)
				}
			}
			reqBody["messages"] = kept
		}
	}
	if !changed {
		return body, false
	}

	minimizedBody, err := json.Marshal(reqBody)
	if err != nil {
		return body, false
	}
	logger.LegacyPrintf("service.openai_gateway", "[OpenAI compat debug] stage=minimax_request_minimized account_id=%d body=%s",
		account.ID, truncateForLog(minimizedBody, openAIResponsesCompatLogPreviewMaxBytes))
	return minimizedBody, true
}

func (s *OpenAIGatewayService) handleResponsesViaChatCompletionsNonStream(
	resp *http.Response,
	c *gin.Context,
	account *Account,
	originalModel string,
	billingModel string,
	upstreamModel string,
	serviceTier *string,
	reasoningEffort *string,
	startTime time.Time,
	estimatedPromptTokens int,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to read upstream response")
		return nil, fmt.Errorf("read response: %w", err)
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=nonstream_upstream_body account_id=%d request_id=%s body=%s",
		account.ID,
		requestID,
		truncateForLog(body, openAIResponsesCompatLogPreviewMaxBytes),
	)

	var chatResp apicompat.ChatCompletionsResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to parse upstream response")
		return nil, fmt.Errorf("parse chat completions response: %w", err)
	}

	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&chatResp)
	responsesResp.Model = originalModel
	usage := chatUsageToOpenAIUsage(chatResp.Usage)
	if isOpenAICompatMiniMaxProvider(account) && usage.InputTokens == 0 && usage.OutputTokens == 0 {
		usage.InputTokens = estimatedPromptTokens
		usage.OutputTokens = estimateOpenAICompatTextTokens(extractChatCompletionResponseText(&chatResp))
		logger.LegacyPrintf("service.openai_gateway", "[Compat billing debug] stage=minimax_usage_estimated_nonstream input_tokens=%d output_tokens=%d",
			usage.InputTokens, usage.OutputTokens)
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.JSON(http.StatusOK, responsesResp)

	return &OpenAIForwardResult{
		RequestID:       requestID,
		Usage:           usage,
		Model:           originalModel,
		BillingModel:    originalModel,
		UpstreamModel:   upstreamModel,
		ServiceTier:     serviceTier,
		ReasoningEffort: reasoningEffort,
		Stream:          false,
		Duration:        time.Since(startTime),
	}, nil
}

func (s *OpenAIGatewayService) handleResponsesViaChatCompletionsStream(
	resp *http.Response,
	c *gin.Context,
	account *Account,
	originalModel string,
	billingModel string,
	upstreamModel string,
	serviceTier *string,
	reasoningEffort *string,
	startTime time.Time,
	estimatedPromptTokens int,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	contentType := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Type")))
	if contentType != "" && !strings.Contains(contentType, "text/event-stream") {
		logger.LegacyPrintf(
			"service.openai_gateway",
			"[OpenAI compat debug] stage=stream_fallback_to_buffered account_id=%d request_id=%s content_type=%s",
			account.ID,
			requestID,
			contentType,
		)
		return s.handleResponsesViaChatCompletionsBufferedAsStream(resp, c, account, originalModel, billingModel, upstreamModel, serviceTier, reasoningEffort, startTime, estimatedPromptTokens)
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	state := apicompat.NewChatCompletionsToResponsesState()
	state.Model = originalModel

	var usage OpenAIUsage
	var firstTokenMs *int
	firstEventWritten := false
	firstUpstreamSSEPayloadLogged := false
	firstUpstreamContentLogged := false
	upstreamChunkCount := 0
	upstreamContentChars := 0
	upstreamReasoningChars := 0
	downstreamEventCount := 0
	downstreamVisibleChars := 0
	var upstreamContentBuilder strings.Builder

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)

	resultWithUsage := func() *OpenAIForwardResult {
		return &OpenAIForwardResult{
			RequestID:       requestID,
			Usage:           usage,
			Model:           originalModel,
			BillingModel:    originalModel,
			UpstreamModel:   upstreamModel,
			ServiceTier:     serviceTier,
			ReasoningEffort: reasoningEffort,
			Stream:          true,
			Duration:        time.Since(startTime),
			FirstTokenMs:    firstTokenMs,
		}
	}

	writeEvents := func(events []apicompat.ResponsesStreamEvent) bool {
		for _, evt := range events {
			downstreamEventCount++
			if evt.Type == "response.output_text.delta" {
				downstreamVisibleChars += len(evt.Delta)
			}
			sse, err := apicompat.ResponsesEventToSSE(evt)
			if err != nil {
				logger.L().Warn("openai responses via chat stream: marshal event failed",
					zap.Error(err),
					zap.String("request_id", requestID),
				)
				continue
			}
			if !firstEventWritten {
				firstEventWritten = true
				ms := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &ms
			}
			if _, err := fmt.Fprint(c.Writer, sse); err != nil {
				logger.L().Info("openai responses via chat stream: client disconnected",
					zap.String("request_id", requestID),
				)
				return true
			}
		}
		if len(events) > 0 {
			c.Writer.Flush()
		}
		return false
	}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		if line == "data: [DONE]" {
			logger.LegacyPrintf(
				"service.openai_gateway",
				"[OpenAI compat debug] stage=stream_done account_id=%d request_id=%s",
				account.ID,
				requestID,
			)
			break
		}

		payload := line[6:]
		if !firstUpstreamSSEPayloadLogged {
			firstUpstreamSSEPayloadLogged = true
			logger.LegacyPrintf(
				"service.openai_gateway",
				"[OpenAI compat debug] stage=stream_first_upstream_payload account_id=%d request_id=%s payload=%s",
				account.ID,
				requestID,
				truncateForLog([]byte(payload), openAIResponsesCompatLogPreviewMaxBytes),
			)
		}
		var chunk apicompat.ChatCompletionsChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			logger.L().Warn("openai responses via chat stream: parse chunk failed",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
			continue
		}
		upstreamChunkCount++
		contentDelta, reasoningDelta := openAIResponsesCompatChunkDeltaSummary(&chunk)
		upstreamContentChars += len(contentDelta)
		upstreamReasoningChars += len(reasoningDelta)
		upstreamContentBuilder.WriteString(contentDelta)
		if !firstUpstreamContentLogged && strings.TrimSpace(contentDelta) != "" {
			firstUpstreamContentLogged = true
			logger.LegacyPrintf(
				"service.openai_gateway",
				"[OpenAI compat debug] stage=stream_first_upstream_content account_id=%d request_id=%s content=%s",
				account.ID,
				requestID,
				truncateForLog([]byte(contentDelta), openAIResponsesCompatLogPreviewMaxBytes),
			)
		}

		if chunk.Usage != nil {
			usage = chatUsageToOpenAIUsage(chunk.Usage)
		}

		events := apicompat.ChatCompletionsChunkToResponsesEvents(&chunk, state)
		if writeEvents(events) {
			return resultWithUsage(), nil
		}
	}

	if err := scanner.Err(); err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		logger.L().Warn("openai responses via chat stream: read error",
			zap.Error(err),
			zap.String("request_id", requestID),
		)
	}

	if state.Usage != nil {
		usage = responsesUsageToOpenAIUsage(state.Usage)
	}
	if isOpenAICompatMiniMaxProvider(account) && usage.InputTokens == 0 && usage.OutputTokens == 0 && upstreamContentBuilder.Len() > 0 {
		usage.InputTokens = estimatedPromptTokens
		usage.OutputTokens = estimateOpenAICompatTextTokens(upstreamContentBuilder.String())
		logger.LegacyPrintf("service.openai_gateway", "[Compat billing debug] stage=minimax_usage_estimated_stream input_tokens=%d output_tokens=%d",
			usage.InputTokens, usage.OutputTokens)
	}
	finalEvents := apicompat.FinalizeChatCompletionsResponsesStream(state)
	if writeEvents(finalEvents) {
		return resultWithUsage(), nil
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=stream_summary account_id=%d request_id=%s upstream_chunks=%d upstream_content_chars=%d upstream_reasoning_chars=%d downstream_events=%d downstream_visible_chars=%d",
		account.ID,
		requestID,
		upstreamChunkCount,
		upstreamContentChars,
		upstreamReasoningChars,
		downstreamEventCount,
		downstreamVisibleChars,
	)

	return resultWithUsage(), nil
}

func (s *OpenAIGatewayService) handleResponsesViaChatCompletionsBufferedAsStream(
	resp *http.Response,
	c *gin.Context,
	account *Account,
	originalModel string,
	billingModel string,
	upstreamModel string,
	serviceTier *string,
	reasoningEffort *string,
	startTime time.Time,
	estimatedPromptTokens int,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to read upstream response")
		return nil, fmt.Errorf("read buffered upstream response: %w", err)
	}
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=buffered_stream_upstream_body account_id=%d request_id=%s body=%s",
		account.ID,
		requestID,
		truncateForLog(body, openAIResponsesCompatLogPreviewMaxBytes),
	)

	var chatResp apicompat.ChatCompletionsResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		writeResponsesError(c, http.StatusBadGateway, "server_error", "Failed to parse upstream response")
		return nil, fmt.Errorf("parse buffered chat completions response: %w", err)
	}

	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&chatResp)
	responsesResp.Model = originalModel
	usage := chatUsageToOpenAIUsage(chatResp.Usage)
	if isOpenAICompatMiniMaxProvider(account) && usage.InputTokens == 0 && usage.OutputTokens == 0 {
		usage.InputTokens = estimatedPromptTokens
		usage.OutputTokens = estimateOpenAICompatTextTokens(extractChatCompletionResponseText(&chatResp))
		logger.LegacyPrintf("service.openai_gateway", "[Compat billing debug] stage=minimax_usage_estimated_buffered input_tokens=%d output_tokens=%d",
			usage.InputTokens, usage.OutputTokens)
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	events := buildResponsesSSEFromBufferedResponse(responsesResp)
	if len(events) > 0 {
		if firstEventJSON, err := json.Marshal(events[0]); err == nil {
			logger.LegacyPrintf(
				"service.openai_gateway",
				"[OpenAI compat debug] stage=buffered_stream_first_event account_id=%d request_id=%s event=%s",
				account.ID,
				requestID,
				truncateForLog(firstEventJSON, openAIResponsesCompatLogPreviewMaxBytes),
			)
		}
	}
	var firstTokenMs *int
	for idx, evt := range events {
		sse, err := apicompat.ResponsesEventToSSE(evt)
		if err != nil {
			return nil, fmt.Errorf("marshal buffered responses event: %w", err)
		}
		if idx == 1 {
			ms := int(time.Since(startTime).Milliseconds())
			firstTokenMs = &ms
		}
		if _, err := fmt.Fprint(c.Writer, sse); err != nil {
			return nil, err
		}
	}
	c.Writer.Flush()
	logger.LegacyPrintf(
		"service.openai_gateway",
		"[OpenAI compat debug] stage=buffered_stream_events_written account_id=%d request_id=%s events=%d",
		account.ID,
		requestID,
		len(events),
	)

	return &OpenAIForwardResult{
		RequestID:       requestID,
		Usage:           usage,
		Model:           originalModel,
		BillingModel:    originalModel,
		UpstreamModel:   upstreamModel,
		ServiceTier:     serviceTier,
		ReasoningEffort: reasoningEffort,
		Stream:          true,
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
	}, nil
}

func buildResponsesSSEFromBufferedResponse(resp *apicompat.ResponsesResponse) []apicompat.ResponsesStreamEvent {
	if resp == nil {
		return nil
	}

	seq := 0
	nextSeq := func() int {
		current := seq
		seq++
		return current
	}

	events := []apicompat.ResponsesStreamEvent{{
		Type:           "response.created",
		SequenceNumber: nextSeq(),
		Response: &apicompat.ResponsesResponse{
			ID:     resp.ID,
			Object: "response",
			Model:  resp.Model,
			Status: "in_progress",
			Output: []apicompat.ResponsesOutput{},
		},
	}}

	for outputIndex, item := range resp.Output {
		current := item
		current.Status = "in_progress"
		current.Content = nil
		current.Summary = nil
		current.Arguments = ""
		if item.Type == "message" {
			current.Content = []apicompat.ResponsesContentPart{{
				Type: "output_text",
				Text: "",
			}}
		}
		events = append(events, apicompat.ResponsesStreamEvent{
			Type:           "response.output_item.added",
			SequenceNumber: nextSeq(),
			OutputIndex:    outputIndex,
			Item:           &current,
		})

		switch item.Type {
		case "message":
			events = append(events, apicompat.ResponsesStreamEvent{
				Type:           "response.content_part.added",
				SequenceNumber: nextSeq(),
				OutputIndex:    outputIndex,
				ContentIndex:   0,
				ItemID:         item.ID,
				Part: &apicompat.ResponsesContentPart{
					Type: "output_text",
					Text: "",
				},
			})
			var text string
			for _, part := range item.Content {
				if part.Type == "output_text" {
					text += part.Text
				}
			}
			if text != "" {
				events = append(events, apicompat.ResponsesStreamEvent{
					Type:           "response.output_text.delta",
					SequenceNumber: nextSeq(),
					OutputIndex:    outputIndex,
					ContentIndex:   0,
					ItemID:         item.ID,
					Delta:          text,
				})
			}
			events = append(events, apicompat.ResponsesStreamEvent{
				Type:           "response.output_text.done",
				SequenceNumber: nextSeq(),
				OutputIndex:    outputIndex,
				ContentIndex:   0,
				ItemID:         item.ID,
				Text:           text,
			})
			events = append(events, apicompat.ResponsesStreamEvent{
				Type:           "response.content_part.done",
				SequenceNumber: nextSeq(),
				OutputIndex:    outputIndex,
				ContentIndex:   0,
				ItemID:         item.ID,
				Part: &apicompat.ResponsesContentPart{
					Type: "output_text",
					Text: text,
				},
			})
		case "reasoning":
			var text string
			for _, part := range item.Summary {
				if part.Type == "summary_text" {
					text += part.Text
				}
			}
			if text != "" {
				events = append(events, apicompat.ResponsesStreamEvent{
					Type:           "response.reasoning_summary_text.delta",
					SequenceNumber: nextSeq(),
					OutputIndex:    outputIndex,
					SummaryIndex:   0,
					ItemID:         item.ID,
					Delta:          text,
				})
			}
			events = append(events, apicompat.ResponsesStreamEvent{
				Type:           "response.reasoning_summary_text.done",
				SequenceNumber: nextSeq(),
				OutputIndex:    outputIndex,
				SummaryIndex:   0,
				ItemID:         item.ID,
			})
		case "function_call":
			if item.Arguments != "" {
				events = append(events, apicompat.ResponsesStreamEvent{
					Type:           "response.function_call_arguments.delta",
					SequenceNumber: nextSeq(),
					OutputIndex:    outputIndex,
					ItemID:         item.ID,
					CallID:         item.CallID,
					Name:           item.Name,
					Delta:          item.Arguments,
				})
			}
			events = append(events, apicompat.ResponsesStreamEvent{
				Type:           "response.function_call_arguments.done",
				SequenceNumber: nextSeq(),
				OutputIndex:    outputIndex,
				ItemID:         item.ID,
				CallID:         item.CallID,
				Name:           item.Name,
			})
		}

		doneItem := item
		doneItem.Status = "completed"
		events = append(events, apicompat.ResponsesStreamEvent{
			Type:           "response.output_item.done",
			SequenceNumber: nextSeq(),
			OutputIndex:    outputIndex,
			Item:           &doneItem,
		})
	}

	events = append(events, apicompat.ResponsesStreamEvent{
		Type:           "response.completed",
		SequenceNumber: nextSeq(),
		Response:       resp,
	})
	return events
}

func openAIResponsesCompatExtractPath(body []byte, paths ...string) string {
	for _, path := range paths {
		if strings.TrimSpace(path) == "" {
			continue
		}
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			return value
		}
	}
	return "-"
}

func openAIResponsesCompatFirstChatRole(messages []apicompat.ChatMessage) string {
	if len(messages) == 0 {
		return "-"
	}
	role := strings.TrimSpace(messages[0].Role)
	if role == "" {
		return "-"
	}
	return role
}

func openAIResponsesCompatFirstChatContentType(messages []apicompat.ChatMessage) string {
	if len(messages) == 0 || len(messages[0].Content) == 0 {
		return "-"
	}
	if strings.TrimSpace(gjson.GetBytes(messages[0].Content, "0.type").String()) != "" {
		return gjson.GetBytes(messages[0].Content, "0.type").String()
	}
	var text string
	if err := json.Unmarshal(messages[0].Content, &text); err == nil {
		return "text"
	}
	return "raw"
}

func openAIResponsesCompatChunkDeltaSummary(chunk *apicompat.ChatCompletionsChunk) (content string, reasoning string) {
	if chunk == nil {
		return "", ""
	}
	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	for _, choice := range chunk.Choices {
		if choice.Delta.Content != nil {
			contentBuilder.WriteString(*choice.Delta.Content)
		}
		if choice.Delta.ReasoningContent != nil {
			reasoningBuilder.WriteString(*choice.Delta.ReasoningContent)
		}
	}
	return contentBuilder.String(), reasoningBuilder.String()
}

func chatUsageToOpenAIUsage(usage *apicompat.ChatUsage) OpenAIUsage {
	if usage == nil {
		return OpenAIUsage{}
	}
	result := OpenAIUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
	}
	if usage.PromptTokensDetails != nil {
		result.CacheReadInputTokens = usage.PromptTokensDetails.CachedTokens
	}
	return result
}

func responsesUsageToOpenAIUsage(usage *apicompat.ResponsesUsage) OpenAIUsage {
	if usage == nil {
		return OpenAIUsage{}
	}
	result := OpenAIUsage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
	if usage.InputTokensDetails != nil {
		result.CacheReadInputTokens = usage.InputTokensDetails.CachedTokens
	}
	return result
}
