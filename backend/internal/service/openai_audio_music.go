package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	openAIAudioSpeechEndpoint       = "/v1/audio/speech"
	openAIMusicGenerationsEndpoint  = "/v1/music/generations"
	openAILyricsGenerationsEndpoint = "/v1/lyrics/generations"

	miniMaxSpeechURL             = "https://api.minimaxi.com/v1/t2a_v2"
	miniMaxMusicURL              = "https://api.minimaxi.com/v1/music_generation"
	miniMaxLyricsURL             = "https://api.minimaxi.com/v1/lyrics_generation"
	miniMaxAudioMaxDownloadBytes = 30 << 20
)

type OpenAIAudioSpeechRequest struct {
	Model          string
	Input          string
	Voice          string
	ResponseFormat string
	Speed          float64
	Body           []byte
}

type OpenAIMusicGenerationRequest struct {
	Model          string
	Prompt         string
	Lyrics         string
	OutputFormat   string
	IsInstrumental *bool
	Body           []byte
}

type OpenAILyricsGenerationRequest struct {
	Model  string
	Prompt string
	Body   []byte
}

type openAIAudioResult struct {
	AudioBase64 string
	URL         string
	MimeType    string
	Format      string
	Model       string
	DurationMs  int64
}

func isOpenAISpeechModel(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return normalized == "gpt-4o-mini-tts" ||
		strings.HasPrefix(normalized, "tts-") ||
		strings.HasPrefix(normalized, "gpt-audio-") ||
		strings.HasPrefix(normalized, "claude-audio-") ||
		strings.HasPrefix(normalized, "claude-speech-")
}

func isMiniMaxSpeechModel(model string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(model)), "speech-")
}

func isOpenAIMusicModel(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(normalized, "gpt-music-") ||
		strings.HasPrefix(normalized, "claude-music-") ||
		normalized == "music" ||
		normalized == "write_full_song"
}

func isMiniMaxMusicModel(model string) bool {
	normalized := strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(normalized, "music-") ||
		normalized == "write_full_song"
}

func (s *OpenAIGatewayService) ParseOpenAIAudioSpeechRequest(c *gin.Context, body []byte) (*OpenAIAudioSpeechRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	if !strings.Contains(c.Request.URL.Path, "/audio/speech") {
		return nil, fmt.Errorf("unsupported audio endpoint")
	}
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("failed to parse request body")
	}
	req := &OpenAIAudioSpeechRequest{
		Model:          strings.TrimSpace(gjson.GetBytes(body, "model").String()),
		Input:          strings.TrimSpace(gjson.GetBytes(body, "input").String()),
		Voice:          strings.TrimSpace(gjson.GetBytes(body, "voice").String()),
		ResponseFormat: strings.TrimSpace(gjson.GetBytes(body, "response_format").String()),
		Body:           body,
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if !isOpenAISpeechModel(req.Model) {
		return nil, fmt.Errorf("audio speech endpoint requires a speech model, got %q", req.Model)
	}
	if req.Input == "" {
		return nil, fmt.Errorf("input is required")
	}
	if speed := gjson.GetBytes(body, "speed"); speed.Exists() {
		if speed.Type != gjson.Number {
			return nil, fmt.Errorf("invalid speed field type")
		}
		req.Speed = speed.Float()
	}
	return req, nil
}

func (s *OpenAIGatewayService) ParseOpenAIMusicGenerationRequest(c *gin.Context, body []byte) (*OpenAIMusicGenerationRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	if !strings.Contains(c.Request.URL.Path, "/music/generations") {
		return nil, fmt.Errorf("unsupported music endpoint")
	}
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("failed to parse request body")
	}
	req := &OpenAIMusicGenerationRequest{
		Model:        strings.TrimSpace(gjson.GetBytes(body, "model").String()),
		Prompt:       strings.TrimSpace(gjson.GetBytes(body, "prompt").String()),
		Lyrics:       strings.TrimSpace(gjson.GetBytes(body, "lyrics").String()),
		OutputFormat: strings.TrimSpace(gjson.GetBytes(body, "output_format").String()),
		Body:         body,
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if !isOpenAIMusicModel(req.Model) {
		return nil, fmt.Errorf("music endpoint requires a music model, got %q", req.Model)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	if instrumental := gjson.GetBytes(body, "is_instrumental"); instrumental.Exists() {
		if instrumental.Type != gjson.True && instrumental.Type != gjson.False {
			return nil, fmt.Errorf("invalid is_instrumental field type")
		}
		v := instrumental.Bool()
		req.IsInstrumental = &v
	}
	return req, nil
}

func (s *OpenAIGatewayService) ParseOpenAILyricsGenerationRequest(c *gin.Context, body []byte) (*OpenAILyricsGenerationRequest, error) {
	if c == nil || c.Request == nil {
		return nil, fmt.Errorf("missing request context")
	}
	if !strings.Contains(c.Request.URL.Path, "/lyrics/generations") {
		return nil, fmt.Errorf("unsupported lyrics endpoint")
	}
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return nil, fmt.Errorf("failed to parse request body")
	}
	req := &OpenAILyricsGenerationRequest{
		Model:  strings.TrimSpace(gjson.GetBytes(body, "model").String()),
		Prompt: strings.TrimSpace(gjson.GetBytes(body, "prompt").String()),
		Body:   body,
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if !isOpenAIMusicModel(req.Model) {
		return nil, fmt.Errorf("lyrics endpoint requires a lyrics model, got %q", req.Model)
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	return req, nil
}

func buildMiniMaxSpeechURL(base string) string {
	return buildMiniMaxVideoURL(base, "/v1/t2a_v2")
}

func buildMiniMaxMusicURL(base string) string {
	return buildMiniMaxVideoURL(miniMaxMusicEndpointBase(base), "/v1/music_generation")
}

func buildMiniMaxLyricsURL(base string) string {
	return buildMiniMaxVideoURL(miniMaxMusicEndpointBase(base), "/v1/lyrics_generation")
}

func buildMiniMaxSpeechBody(parsed *OpenAIAudioSpeechRequest, model string) ([]byte, string, error) {
	if parsed == nil {
		return nil, "", fmt.Errorf("parsed speech request is required")
	}
	format := normalizeAudioFormat(parsed.ResponseFormat, "mp3")
	voiceID := strings.TrimSpace(parsed.Voice)
	if voiceID == "" {
		voiceID = "English_expressive_narrator"
	}
	body := []byte(`{"model":"","text":"","stream":false,"output_format":"url","voice_setting":{"voice_id":"","speed":1,"vol":1,"pitch":0},"audio_setting":{"sample_rate":32000,"bitrate":128000,"format":"","channel":1}}`)
	body, _ = sjson.SetBytes(body, "model", strings.TrimSpace(model))
	body, _ = sjson.SetBytes(body, "text", strings.TrimSpace(parsed.Input))
	body, _ = sjson.SetBytes(body, "voice_setting.voice_id", voiceID)
	body, _ = sjson.SetBytes(body, "audio_setting.format", format)
	if parsed.Speed > 0 {
		body, _ = sjson.SetBytes(body, "voice_setting.speed", parsed.Speed)
	}
	return body, format, nil
}

func buildMiniMaxMusicBody(parsed *OpenAIMusicGenerationRequest, model string) ([]byte, string, error) {
	return buildMiniMaxMusicBodyWithLyrics(parsed, model, strings.TrimSpace(parsed.Lyrics))
}

func buildMiniMaxMusicBodyWithLyrics(parsed *OpenAIMusicGenerationRequest, model string, lyrics string) ([]byte, string, error) {
	if parsed == nil {
		return nil, "", fmt.Errorf("parsed music request is required")
	}
	format := normalizeAudioFormat(parsed.OutputFormat, "mp3")
	body := []byte(`{"model":"","prompt":"","output_format":"url","audio_setting":{"sample_rate":44100,"bitrate":256000,"format":""}}`)
	body, _ = sjson.SetBytes(body, "model", strings.TrimSpace(model))
	body, _ = sjson.SetBytes(body, "prompt", strings.TrimSpace(parsed.Prompt))
	body, _ = sjson.SetBytes(body, "audio_setting.format", format)
	if lyrics = strings.TrimSpace(lyrics); lyrics != "" {
		body, _ = sjson.SetBytes(body, "lyrics", lyrics)
	}
	if parsed.IsInstrumental != nil {
		body, _ = sjson.SetBytes(body, "is_instrumental", *parsed.IsInstrumental)
	}
	return body, format, nil
}

func buildMiniMaxLyricsBody(prompt string) []byte {
	body := []byte(`{"mode":"write_full_song","prompt":""}`)
	body, _ = sjson.SetBytes(body, "prompt", strings.TrimSpace(prompt))
	return body
}

func extractMiniMaxLyrics(body []byte) (string, error) {
	if baseRespErr := miniMaxBaseRespError(body); baseRespErr != "" {
		return "", errors.New(baseRespErr)
	}
	if lyrics := strings.TrimSpace(gjson.GetBytes(body, "lyrics").String()); lyrics != "" {
		return lyrics, nil
	}
	return "", fmt.Errorf("MiniMax lyrics_generation did not return lyrics; body_preview=%s", miniMaxVideoResponsePreview(body))
}

func normalizeAudioFormat(value string, fallback string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "mp3", "wav", "flac", "pcm":
		return value
	default:
		return fallback
	}
}

func audioMIMEType(format string) string {
	switch normalizeAudioFormat(format, "mp3") {
	case "wav":
		return "audio/wav"
	case "flac":
		return "audio/flac"
	case "pcm":
		return "audio/L16"
	default:
		return "audio/mpeg"
	}
}

func convertMiniMaxAudioResponse(body []byte, format string, model string) (openAIAudioResult, error) {
	if baseRespErr := miniMaxBaseRespError(body); baseRespErr != "" {
		return openAIAudioResult{}, errors.New(baseRespErr)
	}
	result := openAIAudioResult{
		MimeType: audioMIMEType(format),
		Format:   normalizeAudioFormat(format, "mp3"),
		Model:    strings.TrimSpace(model),
	}
	for _, path := range []string{"extra_info.audio_length", "extra_info.music_duration"} {
		if value := gjson.GetBytes(body, path); value.Exists() && value.Type == gjson.Number {
			result.DurationMs = value.Int()
			break
		}
	}
	for _, path := range []string{"data.audio", "audio"} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			if strings.HasPrefix(strings.ToLower(value), "http://") || strings.HasPrefix(strings.ToLower(value), "https://") || strings.HasPrefix(strings.ToLower(value), "data:") {
				result.URL = value
				return result, nil
			}
			decoded, err := hex.DecodeString(value)
			if err != nil {
				return openAIAudioResult{}, fmt.Errorf("decode MiniMax hex audio: %w", err)
			}
			result.AudioBase64 = base64.StdEncoding.EncodeToString(decoded)
			return result, nil
		}
	}
	for _, path := range []string{"data.audio_url", "audio_url", "data.url", "url"} {
		if value := strings.TrimSpace(gjson.GetBytes(body, path).String()); value != "" {
			result.URL = value
			return result, nil
		}
	}
	return openAIAudioResult{}, fmt.Errorf("MiniMax audio response did not return audio; body_preview=%s", miniMaxVideoResponsePreview(body))
}

func buildOpenAIMusicResponse(result openAIAudioResult, createdAt int64) ([]byte, error) {
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	out := []byte(`{"created":0,"data":[]}`)
	out, _ = sjson.SetBytes(out, "created", createdAt)
	if result.Model != "" {
		out, _ = sjson.SetBytes(out, "model", result.Model)
	}
	item := []byte(`{}`)
	if result.URL != "" {
		item, _ = sjson.SetBytes(item, "url", result.URL)
	} else {
		item, _ = sjson.SetBytes(item, "url", "data:"+result.MimeType+";base64,"+result.AudioBase64)
		item, _ = sjson.SetBytes(item, "b64_json", result.AudioBase64)
	}
	item, _ = sjson.SetBytes(item, "mime_type", result.MimeType)
	if result.DurationMs > 0 {
		item, _ = sjson.SetBytes(item, "duration_ms", result.DurationMs)
	}
	out, _ = sjson.SetRawBytes(out, "data.-1", item)
	return out, nil
}

func buildOpenAILyricsResponse(lyrics string, model string, createdAt int64) ([]byte, error) {
	if createdAt <= 0 {
		createdAt = time.Now().Unix()
	}
	out := []byte(`{"created":0,"lyrics":""}`)
	out, _ = sjson.SetBytes(out, "created", createdAt)
	out, _ = sjson.SetBytes(out, "lyrics", strings.TrimSpace(lyrics))
	if model = strings.TrimSpace(model); model != "" {
		out, _ = sjson.SetBytes(out, "model", model)
	}
	return out, nil
}

func (s *OpenAIGatewayService) ForwardAudioSpeech(ctx context.Context, c *gin.Context, account *Account, parsed *OpenAIAudioSpeechRequest, channelMappedModel string) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed speech request is required")
	}
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if !isMiniMaxMediaProvider(account) || !isMiniMaxSpeechModel(upstreamModel) {
		return nil, fmt.Errorf("unsupported speech upstream model %q", upstreamModel)
	}
	forwardBody, format, err := buildMiniMaxSpeechBody(parsed, upstreamModel)
	if err != nil {
		return nil, err
	}
	result, headers, err := s.forwardMiniMaxAudio(ctx, c, account, buildMiniMaxSpeechURL, forwardBody, format, upstreamModel)
	if err != nil {
		return nil, err
	}
	if result.URL != "" {
		responseBody, buildErr := buildOpenAIMusicResponse(result, time.Now().Unix())
		if buildErr != nil {
			return nil, buildErr
		}
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), headers, s.responseHeaderFilter)
		c.Data(http.StatusOK, "application/json; charset=utf-8", responseBody)
		return &OpenAIForwardResult{Model: requestModel, UpstreamModel: upstreamModel, ResponseHeaders: headers.Clone()}, nil
	}
	audioBytes, err := base64.StdEncoding.DecodeString(result.AudioBase64)
	if err != nil {
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), headers, s.responseHeaderFilter)
	c.Header("Content-Type", result.MimeType)
	c.Data(http.StatusOK, result.MimeType, audioBytes)
	return &OpenAIForwardResult{Model: requestModel, UpstreamModel: upstreamModel, ResponseHeaders: headers.Clone()}, nil
}

func (s *OpenAIGatewayService) ForwardMusic(ctx context.Context, c *gin.Context, account *Account, parsed *OpenAIMusicGenerationRequest, channelMappedModel string) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed music request is required")
	}
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if !isMiniMaxMediaProvider(account) || !isMiniMaxMusicModel(upstreamModel) {
		return nil, fmt.Errorf("unsupported music upstream model %q", upstreamModel)
	}
	lyrics := strings.TrimSpace(parsed.Lyrics)
	if lyrics == "" {
		generatedLyrics, err := s.generateMiniMaxLyrics(ctx, c, account, parsed.Prompt)
		if err != nil {
			return nil, err
		}
		lyrics = generatedLyrics
	}
	forwardBody, format, err := buildMiniMaxMusicBodyWithLyrics(parsed, upstreamModel, lyrics)
	if err != nil {
		return nil, err
	}
	result, headers, err := s.forwardMiniMaxAudio(ctx, c, account, buildMiniMaxMusicURL, forwardBody, format, upstreamModel)
	if err != nil {
		return nil, err
	}
	responseBody, err := buildOpenAIMusicResponse(result, time.Now().Unix())
	if err != nil {
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), headers, s.responseHeaderFilter)
	c.Data(http.StatusOK, "application/json; charset=utf-8", responseBody)
	return &OpenAIForwardResult{Model: requestModel, UpstreamModel: upstreamModel, ResponseHeaders: headers.Clone()}, nil
}

func (s *OpenAIGatewayService) ForwardLyrics(ctx context.Context, c *gin.Context, account *Account, parsed *OpenAILyricsGenerationRequest, channelMappedModel string) (*OpenAIForwardResult, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed lyrics request is required")
	}
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	if !isMiniMaxMediaProvider(account) || !isMiniMaxLyricsModel(upstreamModel) {
		return nil, fmt.Errorf("unsupported lyrics upstream model %q", upstreamModel)
	}
	lyrics, err := s.generateMiniMaxLyrics(ctx, c, account, parsed.Prompt)
	if err != nil {
		return nil, err
	}
	responseBody, err := buildOpenAILyricsResponse(lyrics, upstreamModel, time.Now().Unix())
	if err != nil {
		return nil, err
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", responseBody)
	return &OpenAIForwardResult{Model: requestModel, UpstreamModel: upstreamModel}, nil
}

func (s *OpenAIGatewayService) generateMiniMaxLyrics(ctx context.Context, c *gin.Context, account *Account, prompt string) (string, error) {
	if account == nil || account.Type != AccountTypeAPIKey {
		return "", fmt.Errorf("lyrics generation currently supports API key accounts only")
	}
	token := miniMaxMediaAPIKey(account)
	if token == "" {
		return "", fmt.Errorf("api_key not found in credentials")
	}
	baseURL := miniMaxMediaBaseURL(account)
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return "", err
	}
	targetURL := buildMiniMaxLyricsURL(validatedURL)
	if targetBase := absoluteURLBase(targetURL); targetBase != "" {
		if _, err := s.validateUpstreamBaseURL(targetBase); err != nil {
			return "", err
		}
	}
	body := buildMiniMaxLyricsBody(prompt)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	if ua := miniMaxMediaUserAgent(account); ua != "" {
		req.Header.Set("User-Agent", ua)
	}
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return "", fmt.Errorf("MiniMax lyrics_generation request failed: %s", sanitizeUpstreamErrorMessage(err.Error()))
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return "", err
	}
	if resp.StatusCode >= 400 {
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		_, handleErr := s.handleErrorResponse(ctx, resp, c, account, body)
		return "", handleErr
	}
	return extractMiniMaxLyrics(respBody)
}

func (s *OpenAIGatewayService) forwardMiniMaxAudio(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	urlBuilder func(string) string,
	body []byte,
	format string,
	upstreamModel string,
) (openAIAudioResult, http.Header, error) {
	if account == nil || account.Type != AccountTypeAPIKey {
		return openAIAudioResult{}, nil, fmt.Errorf("audio and music endpoints currently support API key accounts only")
	}
	token := miniMaxMediaAPIKey(account)
	if token == "" {
		return openAIAudioResult{}, nil, fmt.Errorf("api_key not found in credentials")
	}
	baseURL := miniMaxMediaBaseURL(account)
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return openAIAudioResult{}, nil, err
	}
	targetURL := urlBuilder(validatedURL)
	if targetBase := absoluteURLBase(targetURL); targetBase != "" {
		if _, err := s.validateUpstreamBaseURL(targetBase); err != nil {
			return openAIAudioResult{}, nil, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return openAIAudioResult{}, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	if ua := miniMaxMediaUserAgent(account); ua != "" {
		req.Header.Set("User-Agent", ua)
	}
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	if err != nil {
		return openAIAudioResult{}, nil, fmt.Errorf("upstream request failed: %s", sanitizeUpstreamErrorMessage(err.Error()))
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return openAIAudioResult{}, resp.Header.Clone(), err
	}
	if resp.StatusCode >= 400 {
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		_, handleErr := s.handleErrorResponse(ctx, resp, c, account, body)
		return openAIAudioResult{}, resp.Header.Clone(), handleErr
	}
	result, err := convertMiniMaxAudioResponse(respBody, format, upstreamModel)
	return result, resp.Header.Clone(), err
}
