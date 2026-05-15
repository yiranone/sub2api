package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIGatewayServiceForwardAudioSpeech_APIKeyMiniMaxMappedModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-4o-mini-tts","input":"hello","voice":"English_expressive_narrator","response_format":"mp3"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/audio/speech", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAIAudioSpeechRequest(c, body)
	require.NoError(t, err)
	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"data":{"audio":"https://cdn.example.com/speech.mp3"},"extra_info":{"audio_length":1234},"base_resp":{"status_code":0,"status_msg":"success"}}`)),
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          10,
		Name:        "minimax-speech",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"gpt-4o-mini-tts": "speech-02-hd",
			},
		},
	}

	result, err := svc.ForwardAudioSpeech(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "speech-02-hd", result.UpstreamModel)
	require.Equal(t, "https://api.minimaxi.com/v1/t2a_v2", upstream.lastReq.URL.String())
	require.Equal(t, "speech-02-hd", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "url", gjson.GetBytes(upstream.lastBody, "output_format").String())
	require.Equal(t, "https://cdn.example.com/speech.mp3", gjson.GetBytes(rec.Body.Bytes(), "data.0.url").String())
	require.Equal(t, int64(1234), gjson.GetBytes(rec.Body.Bytes(), "data.0.duration_ms").Int())
	require.Equal(t, http.StatusOK, rec.Code)
}

func TestOpenAIGatewayServiceForwardMusic_APIKeyMiniMaxMappedModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-music-1","prompt":"short piano theme","output_format":"mp3"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/music/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAIMusicGenerationRequest(c, body)
	require.NoError(t, err)
	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"lyrics":"generated line one\ngenerated line two","base_resp":{"status_code":0,"status_msg":"success"}}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"data":{"audio":"https://cdn.example.com/music.mp3"},"extra_info":{"music_duration":4567},"base_resp":{"status_code":0,"status_msg":"success"}}`)),
			},
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          11,
		Name:        "minimax-music",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
			"model_mapping": map[string]any{
				"gpt-music-1": "music-01",
			},
		},
	}

	result, err := svc.ForwardMusic(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "music-01", result.UpstreamModel)
	require.Len(t, upstream.requests, 2)
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", upstream.requests[0].URL.String())
	require.Equal(t, "write_full_song", gjson.GetBytes(upstream.bodies[0], "mode").String())
	require.Equal(t, "short piano theme", gjson.GetBytes(upstream.bodies[0], "prompt").String())
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", upstream.requests[1].URL.String())
	require.Equal(t, "music-01", gjson.GetBytes(upstream.bodies[1], "model").String())
	require.Equal(t, "url", gjson.GetBytes(upstream.bodies[1], "output_format").String())
	require.Equal(t, "generated line one\ngenerated line two", gjson.GetBytes(upstream.bodies[1], "lyrics").String())
	require.Equal(t, "https://cdn.example.com/music.mp3", gjson.GetBytes(rec.Body.Bytes(), "data.0.url").String())
	require.Equal(t, int64(4567), gjson.GetBytes(rec.Body.Bytes(), "data.0.duration_ms").Int())
}

func TestOpenAIGatewayServiceForwardMusic_UsesExplicitLyrics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-music-1","prompt":"short piano theme","lyrics":"line one\nline two","output_format":"mp3"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/music/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAIMusicGenerationRequest(c, body)
	require.NoError(t, err)
	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"data":{"audio":"https://cdn.example.com/music.mp3"},"base_resp":{"status_code":0,"status_msg":"success"}}`)),
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          11,
		Name:        "minimax-music",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":       "test-api-key",
			"base_url":      "https://api.minimaxi.com",
			"model_mapping": map[string]any{"gpt-music-1": "music-01"},
		},
	}

	result, err := svc.ForwardMusic(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.requests, 1)
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", upstream.requests[0].URL.String())
	require.Equal(t, "line one\nline two", gjson.GetBytes(upstream.bodies[0], "lyrics").String())
}

func TestOpenAIGatewayServiceForwardLyrics_APIKeyMiniMaxMappedModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"write_full_song","prompt":"write lyrics about rain"}`)

	req := httptest.NewRequest(http.MethodPost, "/v1/lyrics/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = req

	svc := &OpenAIGatewayService{cfg: &config.Config{}}
	parsed, err := svc.ParseOpenAILyricsGenerationRequest(c, body)
	require.NoError(t, err)
	upstream := &httpUpstreamRecorder{
		resp: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"lyrics":"line one\nline two","base_resp":{"status_code":0,"status_msg":"success"}}`)),
		},
	}
	svc.httpUpstream = upstream
	account := &Account{
		ID:          13,
		Name:        "minimax-lyrics",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":       "test-api-key",
			"base_url":      "https://api.minimaxi.com",
			"model_mapping": map[string]any{"write_full_song": "write_full_song"},
		},
	}

	result, err := svc.ForwardLyrics(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "write_full_song", result.UpstreamModel)
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", upstream.lastReq.URL.String())
	require.Equal(t, "write_full_song", gjson.GetBytes(upstream.lastBody, "mode").String())
	require.Equal(t, "write lyrics about rain", gjson.GetBytes(upstream.lastBody, "prompt").String())
	require.Equal(t, "line one\nline two", gjson.GetBytes(rec.Body.Bytes(), "lyrics").String())
	require.Equal(t, "write_full_song", gjson.GetBytes(rec.Body.Bytes(), "model").String())
}

func TestBuildMiniMaxMusicURLNormalizesConfiguredEndpoint(t *testing.T) {
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com"))
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com/anthropic"))
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com/anthropic/v1/messages"))
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com/v1/chat/completions"))
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com/v1/text/chatcompletion_v2"))
	require.Equal(t, "https://api.minimaxi.com/v1/music_generation", buildMiniMaxMusicURL("https://api.minimaxi.com/v1/music_generation"))
	require.Equal(t, "https://api.minimax.io/v1/music_generation", buildMiniMaxMusicURL("https://api.minimax.io/v1/chat/completions"))
}

func TestBuildMiniMaxLyricsURLNormalizesConfiguredEndpoint(t *testing.T) {
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", buildMiniMaxLyricsURL("https://api.minimaxi.com"))
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", buildMiniMaxLyricsURL("https://api.minimaxi.com/anthropic/v1/messages"))
	require.Equal(t, "https://api.minimaxi.com/v1/lyrics_generation", buildMiniMaxLyricsURL("https://api.minimaxi.com/v1/lyrics_generation"))
}

func TestConvertMiniMaxAudioResponseHexFallback(t *testing.T) {
	result, err := convertMiniMaxAudioResponse([]byte(`{"data":{"audio":"68656c6c6f"},"extra_info":{"audio_length":1234},"base_resp":{"status_code":0,"status_msg":"success"}}`), "mp3", "speech-02-hd")
	require.NoError(t, err)
	require.Equal(t, "aGVsbG8=", result.AudioBase64)
	require.Equal(t, int64(1234), result.DurationMs)
}

func TestAccountTestServiceRunMiniMaxAudioRequestDownloadsURLPreview(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	upstream := &httpUpstreamRecorder{
		responses: []*http.Response{
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"data":{"audio":"https://cdn.example.com/audio.mp3"},"extra_info":{"audio_length":1234},"base_resp":{"status_code":0,"status_msg":"success"}}`)),
			},
			{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"audio/mpeg"}},
				Body:       io.NopCloser(strings.NewReader("hello")),
			},
		},
	}
	svc := &AccountTestService{httpUpstream: upstream, cfg: &config.Config{}}
	account := &Account{
		ID:          12,
		Name:        "minimax-speech",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "test-api-key",
			"base_url": "https://api.minimaxi.com",
		},
	}

	result, err := svc.runMiniMaxAudioRequest(context.Background(), account, "test-api-key", "https://api.minimaxi.com/v1/t2a_v2", []byte(`{"model":"speech-02-hd","text":"hello"}`), "mp3", "speech-02-hd")
	require.NoError(t, err)
	require.Empty(t, result.URL)
	require.Equal(t, "aGVsbG8=", result.AudioBase64)
	require.Equal(t, "data:audio/mpeg;base64,aGVsbG8=", audioResultDataURL(result))
	require.Len(t, upstream.requests, 2)
	_ = c
}

func TestBuildMiniMaxMusicBodyPreservesExplicitLyrics(t *testing.T) {
	body, _, err := buildMiniMaxMusicBody(&OpenAIMusicGenerationRequest{
		Model:  "gpt-music-1",
		Prompt: "rock ballad",
		Lyrics: "line one\nline two",
	}, "music-01")
	require.NoError(t, err)
	require.Equal(t, "line one\nline two", gjson.GetBytes(body, "lyrics").String())
}
