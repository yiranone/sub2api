package service

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAIVideosGenerationsEndpoint = "/v1/videos/generations"

	miniMaxVideoGenerationURL      = "https://api.minimaxi.com/v1/video_generation"
	miniMaxVideoQueryPath          = "/v1/query/video_generation"
	miniMaxVideoFileRetrievePath   = "/v1/files/retrieve"
	miniMaxVideoDefaultPollTimeout = 5 * time.Minute
	miniMaxVideoDefaultPollEvery   = 5 * time.Second
)

type OpenAIVideosRequest struct {
	Model           string
	Prompt          string
	PromptOptimizer *bool
	Duration        int
	Resolution      string
	AspectRatio     string
	FirstFrameURL   string
	Body            []byte
}

func isOpenAIVideoModel(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(normalized, "gpt-video-") || strings.HasPrefix(normalized, "claude-video-")
}

func isMiniMaxVideoGenerationModel(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(normalized, "minimax-hailuo-") ||
		strings.HasPrefix(normalized, "hailuo-") ||
		strings.HasPrefix(normalized, "video-") ||
		strings.Contains(normalized, "t2v")
}

func (s *OpenAIGatewayService) ParseOpenAIVideosRequest(c *gin.Context, body []byte) (*OpenAIVideosRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	path := strings.TrimSpace(c.Request.URL.Path)
	if path != "" && !strings.Contains(path, "/videos/generations") && !strings.Contains(path, "/video/generations") {
		return nil, fmt.Errorf("unsupported videos endpoint")
	}
	if len(body) == 0 {
		return nil, fmt.Errorf("request body is empty")
	}
	if !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("failed to parse request body")
	}
	req := &OpenAIVideosRequest{
		Model:       strings.TrimSpace(gjson.GetBytes(body, "model").String()),
		Prompt:      strings.TrimSpace(gjson.GetBytes(body, "prompt").String()),
		Resolution:  strings.TrimSpace(gjson.GetBytes(body, "resolution").String()),
		AspectRatio: strings.TrimSpace(gjson.GetBytes(body, "aspect_ratio").String()),
		FirstFrameURL: strings.TrimSpace(firstNonEmptyString(
			gjson.GetBytes(body, "first_frame_image").String(),
			gjson.GetBytes(body, "first_frame_image_url").String(),
			gjson.GetBytes(body, "image_url").String(),
		)),
		Body: body,
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if !isOpenAIVideoModel(req.Model) {
		return nil, fmt.Errorf("videos endpoint requires a video model, got %q", req.Model)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if optimizer := gjson.GetBytes(body, "prompt_optimizer"); optimizer.Exists() {
		if optimizer.Type != gjson.True && optimizer.Type != gjson.False {
			return nil, fmt.Errorf("invalid prompt_optimizer field type")
		}
		v := optimizer.Bool()
		req.PromptOptimizer = &v
	}
	if duration := gjson.GetBytes(body, "duration"); duration.Exists() {
		if duration.Type != gjson.Number {
			return nil, fmt.Errorf("invalid duration field type")
		}
		req.Duration = int(duration.Int())
	}
	return req, nil
}

func buildMiniMaxVideoGenerationURL(base string) string {
	return buildMiniMaxVideoURL(base, "/v1/video_generation")
}

func buildMiniMaxVideoQueryURL(base string, taskID string) string {
	values := url.Values{}
	values.Set("task_id", strings.TrimSpace(taskID))
	return buildMiniMaxVideoURL(base, miniMaxVideoQueryPath) + "?" + values.Encode()
}

func buildMiniMaxVideoFileRetrieveURL(base string, fileID string) string {
	values := url.Values{}
	values.Set("file_id", strings.TrimSpace(fileID))
	return buildMiniMaxVideoURL(base, miniMaxVideoFileRetrievePath) + "?" + values.Encode()
}

func buildMiniMaxVideoURL(base string, endpoint string) string {
	normalized := normalizeMiniMaxEndpointBase(base, miniMaxDefaultBaseURL)
	relative := strings.TrimPrefix(strings.TrimSpace(endpoint), "/v1")
	if strings.HasSuffix(normalized, endpoint) || strings.HasSuffix(normalized, relative) {
		return normalized
	}
	if strings.HasSuffix(normalized, "/v1") {
		return normalized + relative
	}
	return normalized + endpoint
}

func buildMiniMaxVideoGenerationBody(parsed *OpenAIVideosRequest, model string) ([]byte, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed videos request is required")
	}
	body := []byte(`{"model":"","prompt":""}`)
	body, _ = sjson.SetBytes(body, "model", strings.TrimSpace(model))
	body, _ = sjson.SetBytes(body, "prompt", strings.TrimSpace(parsed.Prompt))
	if parsed.PromptOptimizer != nil {
		body, _ = sjson.SetBytes(body, "prompt_optimizer", *parsed.PromptOptimizer)
	}
	if parsed.Duration > 0 {
		body, _ = sjson.SetBytes(body, "duration", parsed.Duration)
	}
	if parsed.Resolution != "" {
		body, _ = sjson.SetBytes(body, "resolution", parsed.Resolution)
	}
	if parsed.AspectRatio != "" {
		body, _ = sjson.SetBytes(body, "aspect_ratio", parsed.AspectRatio)
	}
	if parsed.FirstFrameURL != "" {
		body, _ = sjson.SetBytes(body, "first_frame_image", parsed.FirstFrameURL)
	}
	return body, nil
}

func extractMiniMaxVideoTaskID(body []byte) string {
	for _, path := range []string{
		"task_id",
		"data.task_id",
		"taskId",
		"data.taskId",
		"task.id",
		"data.task.id",
		"result.task_id",
		"result.taskId",
		"id",
		"data.id",
	} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			return value
		}
	}
	return ""
}

func miniMaxBaseRespError(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	for _, prefix := range []string{"base_resp", "data.base_resp"} {
		statusCode := gjson.GetBytes(body, prefix+".status_code")
		if !statusCode.Exists() {
			continue
		}
		if statusCode.Int() == 0 {
			return ""
		}
		statusMsg := strings.TrimSpace(gjson.GetBytes(body, prefix+".status_msg").String())
		if statusMsg == "" {
			statusMsg = strings.TrimSpace(gjson.GetBytes(body, prefix+".message").String())
		}
		if statusMsg == "" {
			statusMsg = "unknown MiniMax business error"
		}
		return fmt.Sprintf("MiniMax API error: status_code=%d status_msg=%s", statusCode.Int(), statusMsg)
	}
	for _, path := range []string{"error.message", "message", "msg"} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			return "MiniMax API error: " + value
		}
	}
	return ""
}

func miniMaxVideoResponsePreview(body []byte) string {
	preview := strings.TrimSpace(string(body))
	if preview == "" {
		return "<empty>"
	}
	preview = strings.ReplaceAll(preview, "\n", " ")
	preview = strings.ReplaceAll(preview, "\r", " ")
	if len(preview) > 1000 {
		preview = preview[:1000] + "...(truncated)"
	}
	return preview
}

func extractMiniMaxVideoFileID(body []byte) (string, string, bool) {
	status := strings.TrimSpace(firstNonEmptyString(
		gjson.GetBytes(body, "status").String(),
		gjson.GetBytes(body, "data.status").String(),
		gjson.GetBytes(body, "task_status").String(),
		gjson.GetBytes(body, "data.task_status").String(),
	))
	lowerStatus := strings.ToLower(status)
	failed := strings.Contains(lowerStatus, "fail") || strings.Contains(lowerStatus, "error")
	for _, path := range []string{"file_id", "data.file_id", "video_file_id", "data.video_file_id"} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			return value, status, failed
		}
	}
	return "", status, failed
}

func extractMiniMaxVideoDownloadURL(body []byte) string {
	for _, path := range []string{
		"file.download_url",
		"data.file.download_url",
		"download_url",
		"data.download_url",
		"url",
		"data.url",
		"file_url",
		"data.file_url",
	} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			return value
		}
	}
	return ""
}

func buildOpenAIVideoGenerationResponse(videoURL, taskID, fileID, model string, createdAt int64) ([]byte, error) {
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	out := []byte(`{"created":0,"data":[]}`)
	out, _ = sjson.SetBytes(out, "created", createdAt)
	if strings.TrimSpace(model) != "" {
		out, _ = sjson.SetBytes(out, "model", strings.TrimSpace(model))
	}
	item := []byte(`{}`)
	item, _ = sjson.SetBytes(item, "url", strings.TrimSpace(videoURL))
	item, _ = sjson.SetBytes(item, "mime_type", "video/mp4")
	if strings.TrimSpace(taskID) != "" {
		item, _ = sjson.SetBytes(item, "task_id", strings.TrimSpace(taskID))
	}
	if strings.TrimSpace(fileID) != "" {
		item, _ = sjson.SetBytes(item, "file_id", strings.TrimSpace(fileID))
	}
	out, _ = sjson.SetRawBytes(out, "data.-1", item)
	return out, nil
}

func miniMaxVideoPollConfig() (time.Duration, time.Duration) {
	return miniMaxVideoDefaultPollTimeout, miniMaxVideoDefaultPollEvery
}

func (s *OpenAIGatewayService) ForwardVideos(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIVideosRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed videos request is required")
	}
	if account == nil || account.Type != AccountTypeAPIKey {
		return nil, fmt.Errorf("videos endpoint currently supports API key accounts only")
	}
	startTime := time.Now()
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	if !isOpenAIVideoModel(requestModel) {
		return nil, fmt.Errorf("videos endpoint requires a video model, got %q", requestModel)
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if !isMiniMaxMediaProvider(account) || !isMiniMaxVideoGenerationModel(upstreamModel) {
		return nil, fmt.Errorf("unsupported video upstream model %q", upstreamModel)
	}

	forwardBody, err := buildMiniMaxVideoGenerationBody(parsed, upstreamModel)
	if err != nil {
		return nil, err
	}
	setOpsUpstreamRequestBody(c, forwardBody)
	logger.LegacyPrintf("service.openai_gateway", "[OpenAI] MiniMax video request body=%s", miniMaxVideoResponsePreview(forwardBody))

	upstreamCtx, releaseUpstreamCtx := detachStreamUpstreamContext(ctx, false)
	defer releaseUpstreamCtx()

	token := miniMaxMediaAPIKey(account)
	if token == "" {
		return nil, fmt.Errorf("api_key not found in credentials")
	}
	baseURL := miniMaxMediaBaseURL(account)
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return nil, err
	}
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	respBody, respHeader, err := s.doMiniMaxVideoJSON(upstreamCtx, c, account, proxyURL, token, buildMiniMaxVideoGenerationURL(validatedURL), http.MethodPost, forwardBody)
	if err != nil {
		return nil, err
	}
	if baseRespErr := miniMaxBaseRespError(respBody); baseRespErr != "" {
		return nil, errors.New(baseRespErr)
	}
	taskID := extractMiniMaxVideoTaskID(respBody)
	if taskID == "" {
		return nil, fmt.Errorf("MiniMax video_generation did not return task_id; body_preview=%s", miniMaxVideoResponsePreview(respBody))
	}

	timeout, interval := miniMaxVideoPollConfig()
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var fileID string
	pollOnce := func() error {
		queryBody, _, queryErr := s.doMiniMaxVideoJSON(upstreamCtx, c, account, proxyURL, token, buildMiniMaxVideoQueryURL(validatedURL, taskID), http.MethodGet, nil)
		if queryErr != nil {
			return queryErr
		}
		nextFileID, status, failed := extractMiniMaxVideoFileID(queryBody)
		if failed {
			return fmt.Errorf("MiniMax video generation failed: %s", strings.TrimSpace(status))
		}
		fileID = nextFileID
		return nil
	}
	if err := pollOnce(); err != nil {
		return nil, err
	}
	for fileID == "" {
		select {
		case <-upstreamCtx.Done():
			return nil, upstreamCtx.Err()
		case <-deadline.C:
			return nil, fmt.Errorf("MiniMax video generation timed out waiting for task %s", taskID)
		case <-ticker.C:
			if err := pollOnce(); err != nil {
				return nil, err
			}
		}
	}

	fileBody, _, err := s.doMiniMaxVideoJSON(upstreamCtx, c, account, proxyURL, token, buildMiniMaxVideoFileRetrieveURL(validatedURL, fileID), http.MethodGet, nil)
	if err != nil {
		return nil, err
	}
	videoURL := extractMiniMaxVideoDownloadURL(fileBody)
	if videoURL == "" {
		return nil, fmt.Errorf("MiniMax files/retrieve did not return video download url")
	}

	responseBody, err := buildOpenAIVideoGenerationResponse(videoURL, taskID, fileID, upstreamModel, time.Now().Unix())
	if err != nil {
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), respHeader, s.responseHeaderFilter)
	c.Data(http.StatusOK, "application/json; charset=utf-8", responseBody)
	return &OpenAIForwardResult{
		RequestID:       respHeader.Get("x-request-id"),
		Model:           requestModel,
		UpstreamModel:   upstreamModel,
		ResponseHeaders: respHeader.Clone(),
		Duration:        time.Since(startTime),
	}, nil
}

func (s *OpenAIGatewayService) doMiniMaxVideoJSON(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	proxyURL string,
	token string,
	targetURL string,
	method string,
	body []byte,
) ([]byte, http.Header, error) {
	var reader io.Reader
	if len(body) > 0 {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, targetURL, reader)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if len(body) > 0 {
		req.Header.Set("Content-Type", "application/json")
	}
	customUA := miniMaxMediaUserAgent(account)
	if customUA != "" {
		req.Header.Set("User-Agent", customUA)
	}
	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		return nil, nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return nil, resp.Header.Clone(), err
	}
	if resp.StatusCode >= 400 {
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		_, handleErr := s.handleErrorResponse(ctx, resp, c, account, body)
		return nil, resp.Header.Clone(), handleErr
	}
	logger.LegacyPrintf("service.openai_gateway", "[OpenAI] MiniMax video upstream status=%d url=%s", resp.StatusCode, safeUpstreamURL(targetURL))
	return respBody, resp.Header.Clone(), nil
}

func openAIVideoTestPayload(modelID, prompt string) []byte {
	body := []byte(`{"model":"","prompt":""}`)
	body, _ = sjson.SetBytes(body, "model", strings.TrimSpace(modelID))
	body, _ = sjson.SetBytes(body, "prompt", strings.TrimSpace(prompt))
	return body
}

type openAIVideoTestResult struct {
	URL      string
	TaskID   string
	FileID   string
	MimeType string
}

func parseOpenAIVideoGenerationResponse(body []byte) (openAIVideoTestResult, error) {
	if !gjson.ValidBytes(body) {
		return openAIVideoTestResult{}, fmt.Errorf("failed to parse response")
	}
	item := gjson.GetBytes(body, "data.0")
	if !item.Exists() {
		return openAIVideoTestResult{}, fmt.Errorf("no videos returned from API")
	}
	result := openAIVideoTestResult{
		URL:      strings.TrimSpace(item.Get("url").String()),
		TaskID:   strings.TrimSpace(item.Get("task_id").String()),
		FileID:   strings.TrimSpace(item.Get("file_id").String()),
		MimeType: strings.TrimSpace(item.Get("mime_type").String()),
	}
	if result.URL == "" {
		return openAIVideoTestResult{}, fmt.Errorf("no video URL returned from API")
	}
	if result.MimeType == "" {
		result.MimeType = "video/mp4"
	}
	return result, nil
}
