package service

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
)

const (
	miniMaxDefaultBaseURL = "https://api.minimaxi.com"
)

func miniMaxMediaAPIKey(account *Account) string {
	if account == nil {
		return ""
	}
	if account.IsOpenAIApiKey() {
		return account.GetOpenAIApiKey()
	}
	if account.Type == AccountTypeAPIKey {
		return strings.TrimSpace(account.GetCredential("api_key"))
	}
	return ""
}

func miniMaxMediaBaseURL(account *Account) string {
	if account == nil {
		return miniMaxDefaultBaseURL
	}
	if account.IsOpenAI() {
		if baseURL := strings.TrimSpace(account.GetOpenAIBaseURL()); baseURL != "" {
			return baseURL
		}
	}
	if account.Type == AccountTypeAPIKey {
		if baseURL := strings.TrimSpace(account.GetCredential("base_url")); baseURL != "" {
			return baseURL
		}
	}
	return miniMaxDefaultBaseURL
}

func miniMaxMediaUserAgent(account *Account) string {
	if account == nil {
		return ""
	}
	if account.IsOpenAI() {
		return strings.TrimSpace(account.GetOpenAIUserAgent())
	}
	return strings.TrimSpace(account.GetCredential("user_agent"))
}

func isMiniMaxMediaProvider(account *Account) bool {
	if account == nil || account.Credentials == nil || account.Type != AccountTypeAPIKey {
		return false
	}
	baseURL := ""
	if account.IsOpenAI() {
		baseURL = account.GetOpenAIBaseURL()
	} else {
		baseURL = account.GetCredential("base_url")
	}
	baseURL = strings.ToLower(strings.TrimSpace(baseURL))
	apiKey := strings.ToLower(strings.TrimSpace(miniMaxMediaAPIKey(account)))
	return strings.Contains(baseURL, "minimax") ||
		strings.Contains(baseURL, "minimaxi") ||
		strings.HasPrefix(apiKey, "sk-minimax")
}

func normalizeMiniMaxEndpointBase(base string, fallback string) string {
	normalized := strings.TrimRight(strings.TrimSpace(base), "/")
	if normalized == "" {
		normalized = strings.TrimRight(strings.TrimSpace(fallback), "/")
	}
	if normalized == "" {
		normalized = miniMaxDefaultBaseURL
	}
	lower := strings.ToLower(normalized)
	for _, marker := range []string{
		"/anthropic/v1/messages/count_tokens",
		"/anthropic/v1/messages",
		"/anthropic",
		"/v1/chat/completions",
		"/chat/completions",
		"/v1/text/chatcompletion_v2",
		"/text/chatcompletion_v2",
		"/v1/text/chatcompletion_pro",
		"/text/chatcompletion_pro",
		"/v1/text/chatcompletion",
		"/text/chatcompletion",
		"/v1/responses",
		"/responses",
		"/v1/image_generation",
		"/image_generation",
		"/v1/video_generation",
		"/video_generation",
		"/v1/query/video_generation",
		"/query/video_generation",
		"/v1/files/retrieve",
		"/files/retrieve",
		"/v1/t2a_v2",
		"/t2a_v2",
		"/v1/music_generation",
		"/music_generation",
	} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			normalized = strings.TrimRight(normalized[:idx], "/")
			break
		}
	}
	if normalized == "" {
		return strings.TrimRight(strings.TrimSpace(fallback), "/")
	}
	return normalized
}

func miniMaxMusicEndpointBase(base string) string {
	return normalizeMiniMaxEndpointBase(base, miniMaxDefaultBaseURL)
}

func absoluteURLBase(raw string) string {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.Scheme == "" || parsed.Host == "" {
		return ""
	}
	return parsed.Scheme + "://" + parsed.Host
}

func (s *GatewayService) openAIMediaAdapter() *OpenAIGatewayService {
	if s == nil {
		return nil
	}
	return &OpenAIGatewayService{
		cfg:                  s.cfg,
		httpUpstream:         s.httpUpstream,
		responseHeaderFilter: s.responseHeaderFilter,
	}
}

func openAIForwardResultToGatewayResult(result *OpenAIForwardResult) *ForwardResult {
	if result == nil {
		return nil
	}
	return &ForwardResult{
		RequestID: result.RequestID,
		Usage: ClaudeUsage{
			InputTokens:              result.Usage.InputTokens,
			OutputTokens:             result.Usage.OutputTokens,
			CacheCreationInputTokens: result.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     result.Usage.CacheReadInputTokens,
			ImageOutputTokens:        result.Usage.ImageOutputTokens,
		},
		Model:           result.Model,
		UpstreamModel:   result.UpstreamModel,
		Stream:          result.Stream,
		Duration:        result.Duration,
		FirstTokenMs:    result.FirstTokenMs,
		ReasoningEffort: result.ReasoningEffort,
		ImageCount:      result.ImageCount,
		ImageSize:       result.ImageSize,
	}
}

func (s *GatewayService) ParseOpenAIImagesRequest(c *gin.Context, body []byte) (*OpenAIImagesRequest, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	return adapter.ParseOpenAIImagesRequest(c, body)
}

func (s *GatewayService) ParseOpenAIVideosRequest(c *gin.Context, body []byte) (*OpenAIVideosRequest, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	return adapter.ParseOpenAIVideosRequest(c, body)
}

func (s *GatewayService) ParseOpenAIAudioSpeechRequest(c *gin.Context, body []byte) (*OpenAIAudioSpeechRequest, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	return adapter.ParseOpenAIAudioSpeechRequest(c, body)
}

func (s *GatewayService) ParseOpenAIMusicGenerationRequest(c *gin.Context, body []byte) (*OpenAIMusicGenerationRequest, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	return adapter.ParseOpenAIMusicGenerationRequest(c, body)
}

func (s *GatewayService) ForwardImages(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIImagesRequest,
	channelMappedModel string,
) (*ForwardResult, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	result, err := adapter.ForwardImages(ctx, c, account, body, parsed, channelMappedModel)
	return openAIForwardResultToGatewayResult(result), err
}

func (s *GatewayService) ForwardVideos(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIVideosRequest,
	channelMappedModel string,
) (*ForwardResult, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	result, err := adapter.ForwardVideos(ctx, c, account, body, parsed, channelMappedModel)
	return openAIForwardResultToGatewayResult(result), err
}

func (s *GatewayService) ForwardAudioSpeech(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIAudioSpeechRequest,
	channelMappedModel string,
) (*ForwardResult, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	result, err := adapter.ForwardAudioSpeech(ctx, c, account, parsed, channelMappedModel)
	return openAIForwardResultToGatewayResult(result), err
}

func (s *GatewayService) ForwardMusic(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	parsed *OpenAIMusicGenerationRequest,
	channelMappedModel string,
) (*ForwardResult, error) {
	adapter := s.openAIMediaAdapter()
	if adapter == nil {
		return nil, fmt.Errorf("gateway service is not available")
	}
	result, err := adapter.ForwardMusic(ctx, c, account, parsed, channelMappedModel)
	return openAIForwardResultToGatewayResult(result), err
}
