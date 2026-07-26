package handler

import (
	"context"
	"errors"
	"net/http"
	"strconv"
	"time"

	pkghttputil "github.com/Wei-Shaw/sub2api/internal/pkg/httputil"
	"github.com/Wei-Shaw/sub2api/internal/pkg/ip"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	middleware2 "github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type gatewayMediaKind string

const (
	gatewayMediaImages gatewayMediaKind = "images"
	gatewayMediaVideos gatewayMediaKind = "videos"
	gatewayMediaSpeech gatewayMediaKind = "audio_speech"
	gatewayMediaMusic  gatewayMediaKind = "music"
	gatewayMediaLyrics gatewayMediaKind = "lyrics"
)

func (h *GatewayHandler) Images(c *gin.Context) {
	h.handleMedia(c, gatewayMediaImages)
}

func (h *GatewayHandler) Videos(c *gin.Context) {
	h.handleMedia(c, gatewayMediaVideos)
}

func (h *GatewayHandler) AudioSpeech(c *gin.Context) {
	h.handleMedia(c, gatewayMediaSpeech)
}

func (h *GatewayHandler) Music(c *gin.Context) {
	h.handleMedia(c, gatewayMediaMusic)
}

func (h *GatewayHandler) Lyrics(c *gin.Context) {
	h.handleMedia(c, gatewayMediaLyrics)
}

func (h *GatewayHandler) handleMedia(c *gin.Context, kind gatewayMediaKind) {
	streamStarted := false
	requestStart := time.Now()
	component := "handler.gateway." + string(kind)

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
	if h.gatewayService == nil || h.billingCacheService == nil || h.apiKeyService == nil || h.concurrencyHelper == nil {
		reqLog.Error("gateway.media.dependencies_missing")
		h.errorResponse(c, http.StatusServiceUnavailable, "api_error", "Service temporarily unavailable")
		return
	}

	body, err := pkghttputil.ReadRequestBodyWithPrealloc(c.Request)
	if err != nil {
		if maxErr, ok := extractMaxBytesError(err); ok {
			h.errorResponse(c, http.StatusRequestEntityTooLarge, "invalid_request_error", buildBodyTooLargeMessage(maxErr.Limit))
			return
		}
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}
	if len(body) == 0 {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", "Request body is empty")
		return
	}

	reqModel, reqStream, moderationProtocol, moderationBody, imageSize, parseErr := h.parseMediaRequest(c, kind, body)
	if parseErr != nil {
		h.errorResponse(c, http.StatusBadRequest, "invalid_request_error", parseErr.Error())
		return
	}
	reqLog = reqLog.With(zap.String("model", reqModel), zap.Bool("stream", reqStream))

	if kind == gatewayMediaImages && !service.GroupAllowsImageGeneration(apiKey.Group) {
		h.errorResponse(c, http.StatusForbidden, "permission_error", service.ImageGenerationPermissionMessage())
		return
	}
	if decision := h.checkSecurityAudit(c, reqLog, apiKey, subject, moderationProtocol, reqModel, moderationBody); decision != nil && !decision.AllowNextStage {
		h.anthropicSecurityAuditError(c, decision)
		return
	}

	setOpsRequestContext(c, reqModel, reqStream)
	setOpsEndpointContext(c, "", int16(service.RequestTypeFromLegacy(reqStream, false)))
	channelMapping, _ := h.gatewayService.ResolveChannelMappingAndRestrict(c.Request.Context(), apiKey.GroupID, reqModel)
	if h.errorPassthroughService != nil {
		service.BindErrorPassthroughService(c, h.errorPassthroughService)
	}
	subscription, _ := middleware2.GetSubscriptionFromContext(c)
	service.SetOpsLatencyMs(c, service.OpsAuthLatencyMsKey, time.Since(requestStart).Milliseconds())

	userReleaseFunc, err := h.concurrencyHelper.AcquireUserSlotWithWait(c, subject.UserID, subject.Concurrency, reqStream, &streamStarted)
	if err != nil {
		reqLog.Warn("gateway.media.user_slot_acquire_failed", zap.Error(err))
		h.handleConcurrencyError(c, err, "user", streamStarted)
		return
	}
	userReleaseFunc = wrapReleaseOnDone(c.Request.Context(), userReleaseFunc)
	if userReleaseFunc != nil {
		defer userReleaseFunc()
	}

	quotaPlatform := service.QuotaPlatform(c.Request.Context(), apiKey)
	if err := h.billingCacheService.CheckBillingEligibility(c.Request.Context(), apiKey.User, apiKey, apiKey.Group, subscription, quotaPlatform); err != nil {
		reqLog.Info("gateway.media.billing_eligibility_check_failed", zap.Error(err))
		status, code, message, retryAfter := billingErrorDetails(err)
		if retryAfter > 0 {
			c.Header("Retry-After", strconv.Itoa(retryAfter))
		}
		h.handleStreamingAwareError(c, status, code, message, streamStarted)
		return
	}

	sessionHash := service.DeriveSessionHashFromSeed(string(body))
	fs := NewFailoverState(h.maxAccountSwitches, false)
	for {
		selection, err := h.gatewayService.SelectAccountWithLoadAwareness(c.Request.Context(), apiKey.GroupID, sessionHash, reqModel, fs.FailedAccountIDs, "", subject.UserID)
		if err != nil {
			if len(fs.FailedAccountIDs) == 0 {
				reqLog.Warn("gateway.media.select_account_no_available", zap.String("model", reqModel), zap.Error(err))
				h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available accounts: "+err.Error(), streamStarted)
				return
			}
			action := fs.HandleSelectionExhausted(c.Request.Context())
			if action == FailoverContinue {
				continue
			}
			if action == FailoverCanceled {
				return
			}
			if fs.LastFailoverErr != nil {
				h.handleFailoverExhausted(c, fs.LastFailoverErr, service.PlatformAnthropic, streamStarted)
			} else {
				h.handleFailoverExhaustedSimple(c, 502, streamStarted)
			}
			return
		}
		if selection == nil || selection.Account == nil {
			h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available accounts", streamStarted)
			return
		}

		account := selection.Account
		setOpsSelectedAccount(c, account.ID, account.Platform)
		accountReleaseFunc, ok := h.acquireGatewayMediaAccountSlot(c, selection, reqStream, &streamStarted, reqLog)
		if !ok {
			return
		}

		forwardStart := time.Now()
		writerSizeBeforeForward := c.Writer.Size()
		result, err := h.forwardMedia(c, kind, account, body, channelMapping.MappedModel)
		if accountReleaseFunc != nil {
			accountReleaseFunc()
		}
		service.SetOpsLatencyMs(c, service.OpsResponseLatencyMsKey, time.Since(forwardStart).Milliseconds())
		if err != nil {
			var failoverErr *service.UpstreamFailoverError
			if errors.As(err, &failoverErr) && c.Writer.Size() == writerSizeBeforeForward {
				action := fs.HandleFailoverError(c.Request.Context(), h.gatewayService, account.ID, account.Platform, account.GetPoolModeRetryCount(), failoverErr)
				if action == FailoverContinue {
					continue
				}
				if action == FailoverCanceled {
					return
				}
				h.handleFailoverExhausted(c, fs.LastFailoverErr, account.Platform, streamStarted)
				return
			}
			wroteFallback := h.ensureForwardErrorResponse(c, streamStarted)
			reqLog.Error("gateway.media.forward_failed",
				zap.Int64("account_id", account.ID),
				zap.String("account_platform", account.Platform),
				zap.Bool("fallback_error_response_written", wroteFallback),
				zap.Error(err),
			)
			return
		}

		userAgent := c.GetHeader("User-Agent")
		clientIP := ip.GetClientIP(c)
		requestPayloadHash := service.HashUsageRequestPayload(body)
		inboundEndpoint := GetInboundEndpoint(c)
		upstreamEndpoint := GetUpstreamEndpoint(c, account.Platform)
		upstreamModel := ""
		if result != nil {
			upstreamModel = result.UpstreamModel
			if result.ImageSize == "" {
				result.ImageSize = imageSize
			}
		}
		h.submitUsageRecordTask(c.Request.Context(), func(ctx context.Context) {
			if err := h.gatewayService.RecordUsage(ctx, &service.RecordUsageInput{
				Result:             result,
				QuotaPlatform:      quotaPlatform,
				APIKey:             apiKey,
				User:               apiKey.User,
				Account:            account,
				Subscription:       subscription,
				InboundEndpoint:    inboundEndpoint,
				UpstreamEndpoint:   upstreamEndpoint,
				UserAgent:          userAgent,
				IPAddress:          clientIP,
				RequestPayloadHash: requestPayloadHash,
				APIKeyService:      h.apiKeyService,
				ChannelUsageFields: channelMapping.ToUsageFields(reqModel, upstreamModel),
			}); err != nil {
				logger.L().With(
					zap.String("component", component),
					zap.Int64("user_id", subject.UserID),
					zap.Int64("api_key_id", apiKey.ID),
					zap.Any("group_id", apiKey.GroupID),
					zap.String("model", reqModel),
					zap.Int64("account_id", account.ID),
				).Error("gateway.media.record_usage_failed", zap.Error(err))
			}
		})
		return
	}
}

func (h *GatewayHandler) parseMediaRequest(c *gin.Context, kind gatewayMediaKind, body []byte) (string, bool, string, []byte, string, error) {
	switch kind {
	case gatewayMediaImages:
		parsed, err := h.gatewayService.ParseOpenAIImagesRequest(c, body)
		if err != nil {
			return "", false, "", nil, "", err
		}
		c.Set("gateway_media_images_request", parsed)
		return parsed.Model, parsed.Stream, service.ContentModerationProtocolOpenAIImages, parsed.ModerationBody(), parsed.SizeTier, nil
	case gatewayMediaVideos:
		parsed, err := h.gatewayService.ParseOpenAIVideosRequest(c, body)
		if err != nil {
			return "", false, "", nil, "", err
		}
		c.Set("gateway_media_videos_request", parsed)
		return parsed.Model, false, "media", body, "", nil
	case gatewayMediaSpeech:
		parsed, err := h.gatewayService.ParseOpenAIAudioSpeechRequest(c, body)
		if err != nil {
			return "", false, "", nil, "", err
		}
		c.Set("gateway_media_speech_request", parsed)
		return parsed.Model, false, "media", body, "", nil
	case gatewayMediaMusic:
		parsed, err := h.gatewayService.ParseOpenAIMusicGenerationRequest(c, body)
		if err != nil {
			return "", false, "", nil, "", err
		}
		c.Set("gateway_media_music_request", parsed)
		return parsed.Model, false, "media", body, "", nil
	case gatewayMediaLyrics:
		parsed, err := h.gatewayService.ParseOpenAILyricsGenerationRequest(c, body)
		if err != nil {
			return "", false, "", nil, "", err
		}
		c.Set("gateway_media_lyrics_request", parsed)
		return parsed.Model, false, "media", body, "", nil
	default:
		return "", false, "", nil, "", errors.New("unsupported media endpoint")
	}
}

func (h *GatewayHandler) forwardMedia(c *gin.Context, kind gatewayMediaKind, account *service.Account, body []byte, channelMappedModel string) (*service.ForwardResult, error) {
	switch kind {
	case gatewayMediaImages:
		parsed, _ := c.Get("gateway_media_images_request")
		return h.gatewayService.ForwardImages(c.Request.Context(), c, account, body, parsed.(*service.OpenAIImagesRequest), channelMappedModel)
	case gatewayMediaVideos:
		parsed, _ := c.Get("gateway_media_videos_request")
		return h.gatewayService.ForwardVideos(c.Request.Context(), c, account, body, parsed.(*service.OpenAIVideosRequest), channelMappedModel)
	case gatewayMediaSpeech:
		parsed, _ := c.Get("gateway_media_speech_request")
		return h.gatewayService.ForwardAudioSpeech(c.Request.Context(), c, account, parsed.(*service.OpenAIAudioSpeechRequest), channelMappedModel)
	case gatewayMediaMusic:
		parsed, _ := c.Get("gateway_media_music_request")
		return h.gatewayService.ForwardMusic(c.Request.Context(), c, account, parsed.(*service.OpenAIMusicGenerationRequest), channelMappedModel)
	case gatewayMediaLyrics:
		parsed, _ := c.Get("gateway_media_lyrics_request")
		return h.gatewayService.ForwardLyrics(c.Request.Context(), c, account, parsed.(*service.OpenAILyricsGenerationRequest), channelMappedModel)
	default:
		return nil, errors.New("unsupported media endpoint")
	}
}

func (h *GatewayHandler) acquireGatewayMediaAccountSlot(c *gin.Context, selection *service.AccountSelectionResult, reqStream bool, streamStarted *bool, reqLog *zap.Logger) (func(), bool) {
	account := selection.Account
	accountReleaseFunc := selection.ReleaseFunc
	if !selection.Acquired {
		if selection.WaitPlan == nil {
			reqLog.Warn("gateway.media.select_account_no_slot_no_wait_plan", zap.Int64("account_id", account.ID))
			h.handleStreamingAwareError(c, http.StatusServiceUnavailable, "api_error", "No available accounts", *streamStarted)
			return nil, false
		}
		accountWaitCounted := false
		canWait, err := h.concurrencyHelper.IncrementAccountWaitCount(c.Request.Context(), account.ID, selection.WaitPlan.MaxWaiting)
		if err != nil {
			reqLog.Warn("gateway.media.account_wait_counter_increment_failed", zap.Int64("account_id", account.ID), zap.Error(err))
		} else if !canWait {
			h.handleStreamingAwareError(c, http.StatusTooManyRequests, "rate_limit_error", "Too many pending requests, please retry later", *streamStarted)
			return nil, false
		}
		if err == nil && canWait {
			accountWaitCounted = true
		}
		releaseWait := func() {
			if accountWaitCounted {
				h.concurrencyHelper.DecrementAccountWaitCount(c.Request.Context(), account.ID)
				accountWaitCounted = false
			}
		}
		accountReleaseFunc, err = h.concurrencyHelper.AcquireAccountSlotWithWaitTimeout(
			c,
			account.ID,
			selection.WaitPlan.MaxConcurrency,
			selection.WaitPlan.Timeout,
			reqStream,
			streamStarted,
		)
		if err != nil {
			releaseWait()
			reqLog.Warn("gateway.media.account_slot_acquire_failed", zap.Int64("account_id", account.ID), zap.Error(err))
			h.handleConcurrencyError(c, err, "account", *streamStarted)
			return nil, false
		}
		releaseWait()
	}
	return wrapReleaseOnDone(c.Request.Context(), accountReleaseFunc), true
}
