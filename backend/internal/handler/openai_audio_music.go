package handler

import (
	"net/http"
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// AudioSpeech handles OpenAI-style speech synthesis requests.
// POST /v1/audio/speech
func (h *OpenAIGatewayHandler) AudioSpeech(c *gin.Context) {
	h.handleAudioMusic(c, "handler.openai_gateway.audio_speech", "speech")
}

// Music handles OpenAI-style music generation requests.
// POST /v1/music/generations
func (h *OpenAIGatewayHandler) Music(c *gin.Context) {
	h.handleAudioMusic(c, "handler.openai_gateway.music", "music")
}

// Lyrics handles OpenAI-style lyrics generation requests.
// POST /v1/lyrics/generations
func (h *OpenAIGatewayHandler) Lyrics(c *gin.Context) {
	h.handleAudioMusic(c, "handler.openai_gateway.lyrics", "lyrics")
}

func (h *OpenAIGatewayHandler) handleAudioMusic(c *gin.Context, component string, kind string) {
	streamStarted := false
	defer h.recoverResponsesPanic(c, &streamStarted)

	requestStart := time.Now()
	apiKey, ok := middleware2.GetAPIKeyFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusUnauthorized, "authentication_error", "Invalid API key")
		return
	}
	subject, ok := middleware2.GetAuthSubjectFromContext(c)
	if !ok {
		h.errorResponse(c, http.StatusInternalServerError, "api_error", "User context not found")
		return
	}
	reqLog := requestLogger(c, component,
		zap.Int64("user_id", subject.UserID),
		zap.Int64("api_key_id", apiKey.ID),
		zap.Any("group_id", apiKey.GroupID),
	)
	if !h.ensureResponsesDependencies(c, reqLog) {
		return
	}
	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	if len(body) == 0 {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}

	var model string
	var result *service.OpenAIForwardResult
	var forwardErr error
	var upstreamModel string
	var channelMapping service.ChannelMappingResult
	auditMediaRequest := func(model string) bool {
		if decision := h.checkSecurityAudit(c, reqLog, apiKey, subject, "media", model, body); decision != nil && !decision.AllowNextStage {
			h.openAISecurityAuditError(c, decision)
			return false
		}
		return true
	}

	service.SetOpsLatencyMs(c, service.OpsAuthLatencyMsKey, time.Since(requestStart).Milliseconds())
	if kind == "speech" {
		parsed, parseErr := h.gatewayService.ParseOpenAIAudioSpeechRequest(c, body)
		if parseErr != nil {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", parseErr.Error())
			return
		}
		model = parsed.Model
		if !auditMediaRequest(model) {
			return
		}
		channelMapping, _ = h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, parsed.Model)
		_ = subject
		account, selected := h.selectSimpleOpenAIAccount(c, reqLog, apiKey, parsed.Model, &streamStarted)
		if !selected {
			return
		}
		forwardStart := time.Now()
		result, forwardErr = h.gatewayService.ForwardAudioSpeech(c.Request.Context(), c, account, parsed, channelMapping.MappedModel)
		service.SetOpsLatencyMs(c, service.OpsResponseLatencyMsKey, time.Since(forwardStart).Milliseconds())
		if result != nil {
			upstreamModel = result.UpstreamModel
		}
		h.finishSimpleOpenAIForward(c, reqLog, apiKey, account, result, forwardErr, channelMapping, parsed.Model, upstreamModel, body, component, &streamStarted)
		return
	}

	if kind == "lyrics" {
		parsed, parseErr := h.gatewayService.ParseOpenAILyricsGenerationRequest(c, body)
		if parseErr != nil {
			h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", parseErr.Error())
			return
		}
		model = parsed.Model
		if !auditMediaRequest(model) {
			return
		}
		channelMapping, _ = h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, parsed.Model)
		_ = subject
		account, selected := h.selectSimpleOpenAIAccount(c, reqLog, apiKey, parsed.Model, &streamStarted)
		if !selected {
			return
		}
		forwardStart := time.Now()
		result, forwardErr = h.gatewayService.ForwardLyrics(c.Request.Context(), c, account, parsed, channelMapping.MappedModel)
		service.SetOpsLatencyMs(c, service.OpsResponseLatencyMsKey, time.Since(forwardStart).Milliseconds())
		if result != nil {
			upstreamModel = result.UpstreamModel
		}
		h.finishSimpleOpenAIForward(c, reqLog, apiKey, account, result, forwardErr, channelMapping, model, upstreamModel, body, component, &streamStarted)
		return
	}

	parsed, parseErr := h.gatewayService.ParseOpenAIMusicGenerationRequest(c, body)
	if parseErr != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", parseErr.Error())
		return
	}
	model = parsed.Model
	if !auditMediaRequest(model) {
		return
	}
	channelMapping, _ = h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, parsed.Model)
	_ = subject
	account, selected := h.selectSimpleOpenAIAccount(c, reqLog, apiKey, parsed.Model, &streamStarted)
	if !selected {
		return
	}
	forwardStart := time.Now()
	result, forwardErr = h.gatewayService.ForwardMusic(c.Request.Context(), c, account, parsed, channelMapping.MappedModel)
	service.SetOpsLatencyMs(c, service.OpsResponseLatencyMsKey, time.Since(forwardStart).Milliseconds())
	if result != nil {
		upstreamModel = result.UpstreamModel
	}
	h.finishSimpleOpenAIForward(c, reqLog, apiKey, account, result, forwardErr, channelMapping, model, upstreamModel, body, component, &streamStarted)
}

func (h *OpenAIGatewayHandler) selectSimpleOpenAIAccount(c *gin.Context, reqLog *zap.Logger, apiKey *service.APIKey, model string, streamStarted *bool) (*service.Account, bool) {
	setOpsRequestContext(c, model, false)
	setOpsEndpointContext(c, "", int16(service.RequestTypeFromLegacy(false, false)))
	selection, _, err := h.gatewayService.SelectAccountWithScheduler(
		c.Request.Context(),
		apiKey.GroupID,
		"",
		h.gatewayService.GenerateExplicitSessionHash(c, nil),
		model,
		nil,
		service.OpenAIUpstreamTransportAny,
		false,
	)
	if err != nil || selection == nil || selection.Account == nil {
		reqLog.Warn("openai.audio_music.account_select_failed", zap.Error(err))
		h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available compatible accounts", *streamStarted)
		return nil, false
	}
	setOpsSelectedAccount(c, selection.Account.ID, selection.Account.Platform)
	return selection.Account, true
}

func (h *OpenAIGatewayHandler) finishSimpleOpenAIForward(
	c *gin.Context,
	reqLog *zap.Logger,
	apiKey *service.APIKey,
	account *service.Account,
	result *service.OpenAIForwardResult,
	err error,
	channelMapping service.ChannelMappingResult,
	requestModel string,
	upstreamModel string,
	body []byte,
	component string,
	streamStarted *bool,
) {
	if err != nil {
		h.gatewayService.ReportOpenAIAccountScheduleResult(account, account.GetMappedModel(requestModel), false, nil)
		wroteFallback := h.ensureForwardErrorResponse(c, *streamStarted)
		reqLog.Warn("openai.audio_music.forward_failed",
			zap.Int64("account_id", account.ID),
			zap.Bool("fallback_error_response_written", wroteFallback),
			zap.Error(err),
		)
		return
	}
	h.gatewayService.ReportOpenAIAccountScheduleResult(account, account.GetMappedModel(requestModel), true, nil)
	_ = apiKey
	_ = result
	_ = channelMapping
	_ = requestModel
	_ = upstreamModel
	_ = body
	_ = component
}
